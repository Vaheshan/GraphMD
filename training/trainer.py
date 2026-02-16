from typing import Dict, Any, List, Optional

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from models import MultiscaleMDGNN
from .batch_utils import GraphBatch


class Trainer:
    """
    Trainer encapsulating pretraining (frame stability) and fine-tuning (affinity)
    for the MultiscaleMDGNN model.

    This class is intentionally lightweight and makes minimal assumptions about
    the surrounding training loop. It expects callers to prepare GraphBatch
    instances and label tensors according to the methods below.
    """

    def __init__(
        self,
        model: MultiscaleMDGNN,
        optimizer: Optimizer,
        lambda_temp: float = 0.0,
        alpha_multitask: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lambda_temp = float(lambda_temp)
        self.alpha_multitask = float(alpha_multitask)
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.mse_loss = nn.MSELoss()

    def _move_graph_batch(self, batch: GraphBatch) -> GraphBatch:
        batch.protein = batch.protein.to(self.device)
        batch.pocket = batch.pocket.to(self.device)
        labels_device: Dict[str, Any] = {}
        for k, v in batch.labels.items():
            if isinstance(v, torch.Tensor):
                labels_device[k] = v.to(self.device)
            else:
                labels_device[k] = v
        batch.labels = labels_device
        return batch

    def pretrain_step(self, frame_batches: List[GraphBatch]) -> Dict[str, Tensor]:
        """
        Frame stability pretraining.

        Args:
            frame_batches: List of GraphBatch objects, one per selected frame t.
                For each t, frame_batches[t].labels must contain:
                    - 'y_stability': Tensor of shape (B,) or (B, 1) with targets
                      for frame t.

        Returns:
            Dict with:
                - 'loss': total pretraining loss (scalar tensor).
                - 'L_stability': frame-wise MSE loss term.
                - 'L_temp': temporal regularization loss term.
        """
        if len(frame_batches) == 0:
            raise ValueError("pretrain_step requires at least one frame batch.")

        self.model.train()
        self.optimizer.zero_grad()

        preds: List[Tensor] = []
        targets: List[Tensor] = []
        latents: List[Tensor] = []

        for fb in frame_batches:
            fb = self._move_graph_batch(fb)
            out = self.model({"protein": fb.protein, "pocket": fb.pocket}, return_latent=True)
            y_pred = out["y_pred"].view(-1)
            Z = out["Z"]  # (B, D)
            y_true = fb.labels["y_stability"].view(-1)

            preds.append(y_pred)
            targets.append(y_true)
            latents.append(Z)

        y_pred_all = torch.stack(preds, dim=1)  # (B, T)
        y_true_all = torch.stack(targets, dim=1)  # (B, T)

        # Stability loss: MSE across all frames
        L_stability = self.mse_loss(y_pred_all, y_true_all)

        # Temporal regularization over latent embeddings
        L_temp = torch.tensor(0.0, device=self.device)
        if self.lambda_temp > 0.0 and len(latents) > 1:
            Z_stack = torch.stack(latents, dim=1)  # (B, T, D)
            Z_t = Z_stack[:, :-1, :]
            Z_tp1 = Z_stack[:, 1:, :]
            L_temp = torch.mean((Z_t - Z_tp1) ** 2)

        loss = L_stability + self.lambda_temp * L_temp
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.detach(),
            "L_stability": L_stability.detach(),
            "L_temp": L_temp.detach(),
        }

    def finetune_step(self, batch: GraphBatch, multitask: bool = False) -> Dict[str, Tensor]:
        """
        Binding affinity fine-tuning (optionally multitask with stability).

        Args:
            batch: GraphBatch built for the most stable frame per complex, with:
                - labels['y_affinity']: (B,) or (B, 1) affinity targets.
                - optional labels['y_stability']: stability scores for the same frame.
            multitask: If True, include stability loss with weight alpha_multitask.

        Returns:
            Dict with:
                - 'loss': total fine-tuning loss.
                - 'L_affinity': affinity MSE loss term.
                - 'L_stability': optional stability loss term (0 if not used).
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch = self._move_graph_batch(batch)
        out = self.model({"protein": batch.protein, "pocket": batch.pocket}, return_latent=False)

        y_pred_aff = out["y_pred"].view(-1)
        y_true_aff = batch.labels["y_affinity"].view(-1)

        L_affinity = self.mse_loss(y_pred_aff, y_true_aff)
        L_stability = torch.tensor(0.0, device=self.device)

        if multitask and "y_stability" in batch.labels:
            y_true_stab = batch.labels["y_stability"].view(-1)
            # Reuse the same prediction as a proxy or alternatively call the model
            # again if a different head is desired. Here we keep it simple.
            L_stability = self.mse_loss(y_pred_aff, y_true_stab)

        loss = L_affinity + self.alpha_multitask * L_stability
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.detach(),
            "L_affinity": L_affinity.detach(),
            "L_stability": L_stability.detach(),
        }

