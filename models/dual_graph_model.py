from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch import nn, Tensor
from torch_geometric.data import Data, Batch

from graphs import (
    ProteinGraphBuilder,
    CorrelationEdgeBuilder,
    PocketGraphBuilder,
    ProteinGraphInputs,
    PocketGraphInputs,
)
from .protein_encoder import ProteinGNNEncoder
from .pocket_encoder import PocketGNNEncoder
from .cross_attention import CrossGraphAttentionModule
from .readout import AttentionPoolingReadout
from .head import PredictionMLP, QuantumFeatureEncoder


@dataclass
class ComplexInputs:
    """
    Per-complex input container for building graphs and running the model.

    This assumes a single frame (e.g., most stable frame) for the pocket graph,
    and optionally multiple frames for residue centroids used in correlation
    edges.
    """

    residue_backbone_coords: Tensor  # (R, 3, 3)
    md_residue_coords: Optional[Tensor]  # (T, R, 3) or None
    pocket_atom_coords: Tensor  # (A, 3)
    pocket_atom_features: Tensor  # (A, F_atom)
    pocket_atom_is_ligand: Tensor  # (A,)
    atom_to_residue: Optional[Tensor] = None  # (A,) or None


class MultiscaleMDGNN(nn.Module):
    """
    Multiscale dual-graph MD-aware GNN for protein–ligand modeling.

    High-level flow per batch:

        For each of num_layers refinement steps:
            1. One protein residue-graph message-passing layer.
            2. Cross-graph residue → atom attention.
            3. One pocket atom-graph message-passing layer.
        Then Top-K + attention pooling and an MLP prediction head.

    This module expects callers to provide already-built batched graphs via
    torch_geometric.data.Batch objects. See the example usage at the bottom of
    this file for a sketch of a typical workflow.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        residue_hidden_dim: int = 32,
        atom_hidden_dim: int = 32,
        num_layers: int = 5,
        protein_layers: Optional[int] = None,
        pocket_layers: Optional[int] = None,
        top_k: int = 16,
        dropout: float = 0.0,
        use_qm_features_for_finetune: bool = False,
        qm_feature_dim: int = 0,
        qm_encoded_dim: int = 64,
    ) -> None:
        super().__init__()
        self.use_qm_features_for_finetune = bool(use_qm_features_for_finetune)
        self.qm_feature_dim = int(qm_feature_dim)

        if protein_layers is not None and pocket_layers is not None:
            if protein_layers != pocket_layers:
                raise ValueError(
                    "protein_layers and pocket_layers must match for interleaved forward"
                )
            num_layers = protein_layers
        elif protein_layers is not None:
            num_layers = protein_layers
        elif pocket_layers is not None:
            num_layers = pocket_layers
        self.num_layers = int(num_layers)

        # Graph builders
        self.protein_builder = ProteinGraphBuilder()
        self.corr_builder = CorrelationEdgeBuilder()
        self.pocket_builder = PocketGraphBuilder()

        # Encoders
        self.protein_encoder = ProteinGNNEncoder(
            in_dim=9,
            hidden_dim=residue_hidden_dim,
            edge_dim=4,
            num_layers=self.num_layers,
            dropout=dropout,
        )
        self.pocket_encoder = PocketGNNEncoder(
            in_dim=atom_feature_dim,
            hidden_dim=atom_hidden_dim,
            edge_dim=9,
            num_layers=self.num_layers,
            dropout=dropout,
        )

        # Cross-graph attention
        self.cross_attention = CrossGraphAttentionModule(
            atom_dim=atom_hidden_dim,
            residue_dim=residue_hidden_dim,
        )

        # Readout and prediction head
        self.readout = AttentionPoolingReadout(hidden_dim=atom_hidden_dim, top_k=top_k)
        self.qm_encoder: Optional[QuantumFeatureEncoder]
        head_in_dim = atom_hidden_dim
        if self.use_qm_features_for_finetune:
            if self.qm_feature_dim <= 0:
                raise ValueError(
                    "qm_feature_dim must be > 0 when use_qm_features_for_finetune=True"
                )
            self.qm_encoder = QuantumFeatureEncoder(
                in_dim=self.qm_feature_dim,
                out_dim=qm_encoded_dim,
                dropout=dropout,
            )
            head_in_dim += qm_encoded_dim
        else:
            self.qm_encoder = None
        self.head = PredictionMLP(in_dim=head_in_dim)

    def build_graphs_from_complex(
        self, complex_inputs: ComplexInputs
    ) -> Dict[str, Data]:
        """
        Build protein and pocket graphs for a single complex.
        """
        protein_inputs = ProteinGraphInputs(
            backbone_coords=complex_inputs.residue_backbone_coords,
            md_residue_coords=complex_inputs.md_residue_coords,
        )
        protein_data = self.protein_builder(protein_inputs)
        if complex_inputs.md_residue_coords is not None:
            protein_data = self.corr_builder(
                protein_data, complex_inputs.md_residue_coords
            )

        pocket_inputs = PocketGraphInputs(
            atom_coords=complex_inputs.pocket_atom_coords,
            atom_features=complex_inputs.pocket_atom_features,
            atom_is_ligand=complex_inputs.pocket_atom_is_ligand,
            atom_to_residue=complex_inputs.atom_to_residue,
        )
        pocket_data = self.pocket_builder(pocket_inputs)

        return {"protein": protein_data, "pocket": pocket_data}

    def forward(
        self,
        batch: Dict[str, Any],
        return_latent: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for a batched set of complexes.

        Args:
            batch: Dictionary containing:
                - 'protein': Batch of residue graphs (torch_geometric.data.Batch).
                - 'pocket': Batch of pocket atom graphs (torch_geometric.data.Batch).
            return_latent: If True, also return pooled embedding Z.

        Returns:
            Dict with:
                - 'y_pred': (B, 1) predictions.
                - optionally 'Z': (B, D) pooled embeddings.
        """
        protein_batch: Batch = batch["protein"]
        pocket_batch: Batch = batch["pocket"]

        H_res = self.protein_encoder.project(protein_batch)  # (R_total, D_r)
        H_atoms = self.pocket_encoder.project(pocket_batch)  # (A_total, D_a)

        protein_edge_index = protein_batch.edge_index
        protein_edge_attr = protein_batch.edge_attr
        pocket_edge_index = pocket_batch.edge_index
        pocket_edge_attr = pocket_batch.edge_attr
        atom_to_residue = getattr(pocket_batch, "atom_to_residue", None)

        for layer_idx in range(self.num_layers):
            H_res = self.protein_encoder.apply_layer(
                layer_idx,
                H_res,
                protein_edge_index,
                protein_edge_attr,
            )
            H_atoms = self.cross_attention(
                atom_h=H_atoms,
                residue_h=H_res,
                atom_batch=pocket_batch.batch,
                residue_batch=protein_batch.batch,
                atom_to_residue=atom_to_residue,
            )
            H_atoms = self.pocket_encoder.apply_layer(
                layer_idx,
                H_atoms,
                pocket_edge_index,
                pocket_edge_attr,
            )

        # Readout over pocket atoms
        Z = self.readout(
            h=H_atoms,
            coords=pocket_batch.pos,
            batch=pocket_batch.batch,
            is_ligand=pocket_batch.is_ligand,
        )

        if self.use_qm_features_for_finetune:
            qm_features = batch.get("qm_features", None)
            if qm_features is None:
                qm_features = torch.zeros(
                    (Z.size(0), self.qm_feature_dim), device=Z.device, dtype=Z.dtype
                )
            elif qm_features.dim() == 1:
                qm_features = qm_features.unsqueeze(0)
            qm_features = qm_features.to(device=Z.device, dtype=Z.dtype)
            qm_encoded = self.qm_encoder(qm_features)
            Z_head = torch.cat([Z, qm_encoded], dim=-1)
        else:
            Z_head = Z

        y_pred = self.head(Z_head)

        out = {"y_pred": y_pred}
        if return_latent:
            out["Z"] = Z
        return out


"""
Example usage sketch (not executable as-is):

    from torch.optim import Adam
    from graphs import ProteinGraphBuilder, CorrelationEdgeBuilder, PocketGraphBuilder, ProteinGraphInputs, PocketGraphInputs
    from training.batch_utils import collate_complexes
    from training.trainer import Trainer

    # Prepare per-complex tensors (pseudo-code)
    complexes = [...]
    protein_builder = ProteinGraphBuilder()
    corr_builder = CorrelationEdgeBuilder()
    pocket_builder = PocketGraphBuilder()

    protein_graphs = []
    pocket_graphs = []
    y_affinity = []

    for c in complexes:
        protein_inputs = ProteinGraphInputs(
            backbone_coords=c.residue_backbone_coords,
            md_residue_coords=c.md_residue_coords,
        )
        p_data = protein_builder(protein_inputs)
        p_data = corr_builder(p_data, c.md_residue_coords)

        pocket_inputs = PocketGraphInputs(
            atom_coords=c.pocket_atom_coords,
            atom_features=c.pocket_atom_features,
            atom_is_ligand=c.pocket_atom_is_ligand,
            atom_to_residue=c.atom_to_residue,
        )
        a_data = pocket_builder(pocket_inputs)

        protein_graphs.append(p_data)
        pocket_graphs.append(a_data)
        y_affinity.append(c.y_affinity)

    labels = {"y_affinity": torch.stack(y_affinity)}  # (B,)
    graph_batch = collate_complexes(protein_graphs, pocket_graphs, labels)

    model = MultiscaleMDGNN(atom_feature_dim=c.pocket_atom_features.size(-1))
    opt = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, opt)

    # Fine-tuning step
    loss_dict = trainer.finetune_step(graph_batch)
    print(loss_dict["loss"])
"""


