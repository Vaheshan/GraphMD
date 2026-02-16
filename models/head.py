from typing import List

from torch import nn, Tensor


class PredictionMLP(nn.Module):
    """
    Simple configurable MLP prediction head.

    Maps pooled graph embedding Z to a scalar prediction.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        d_in = in_dim
        for d_hidden in hidden_dims:
            layers.append(nn.Linear(d_in, d_hidden))
            layers.append(nn.ReLU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(d_hidden))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            d_in = d_hidden
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (B, D) pooled graph embedding.

        Returns:
            y: (B, 1) predictions.
        """
        return self.net(z)

