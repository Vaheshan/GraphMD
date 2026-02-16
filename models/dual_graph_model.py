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
from .head import PredictionMLP


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

        1. Protein graph message passing on residue graph.
        2. Cross-graph residue → atom attention to inject global context.
        3. Pocket graph message passing on atom graph.
        4. Top-K + attention pooling over selected pocket/ligand atoms.
        5. MLP head to predict a scalar property (e.g., stability or affinity).

    This module expects callers to provide already-built batched graphs via
    torch_geometric.data.Batch objects. See the example usage at the bottom of
    this file for a sketch of a typical workflow.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        residue_hidden_dim: int = 128,
        atom_hidden_dim: int = 128,
        protein_layers: int = 3,
        pocket_layers: int = 5,
        top_k: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Graph builders
        self.protein_builder = ProteinGraphBuilder()
        self.corr_builder = CorrelationEdgeBuilder()
        self.pocket_builder = PocketGraphBuilder()

        # Encoders
        self.protein_encoder = ProteinGNNEncoder(
            in_dim=9,
            hidden_dim=residue_hidden_dim,
            edge_dim=4,
            num_layers=protein_layers,
            dropout=dropout,
        )
        self.pocket_encoder = PocketGNNEncoder(
            in_dim=atom_feature_dim,
            hidden_dim=atom_hidden_dim,
            edge_dim=9,
            num_layers=pocket_layers,
            dropout=dropout,
        )

        # Cross-graph attention
        self.cross_attention = CrossGraphAttentionModule(
            atom_dim=atom_hidden_dim,
            residue_dim=residue_hidden_dim,
        )

        # Readout and prediction head
        self.readout = AttentionPoolingReadout(hidden_dim=atom_hidden_dim, top_k=top_k)
        self.head = PredictionMLP(in_dim=atom_hidden_dim)

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

        # Protein encoder
        H_res = self.protein_encoder(protein_batch)  # (R_total, D_r)

        # Initial pocket encoder (local atom context)
        H_atoms = self.pocket_encoder(pocket_batch)  # (A_total, D_a)

        # Cross-graph attention (residue → atom)
        H_atoms_ctx = self.cross_attention(
            atom_h=H_atoms,
            residue_h=H_res,
            atom_batch=pocket_batch.batch,
            residue_batch=protein_batch.batch,
            atom_to_residue=getattr(pocket_batch, "atom_to_residue", None),
        )

        # Optional second pass of pocket message passing on context-enriched atoms
        pocket_batch_enriched = pocket_batch.clone()
        pocket_batch_enriched.x = H_atoms_ctx
        H_atoms_final = self.pocket_encoder(pocket_batch_enriched)

        # Readout over pocket atoms
        Z = self.readout(
            h=H_atoms_final,
            coords=pocket_batch.pos,
            batch=pocket_batch.batch,
            is_ligand=pocket_batch.is_ligand,
        )

        y_pred = self.head(Z)

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


