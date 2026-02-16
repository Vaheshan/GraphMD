from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers = []
    d_in = in_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(d_in, hidden_dim))
        layers.append(nn.ReLU())
        d_in = hidden_dim
    layers.append(nn.Linear(d_in, out_dim))
    return nn.Sequential(*layers)


class PocketMessageLayer(MessagePassing):
    """
    Message passing layer for pocket atom graphs using edge-aware MLP messages.

    Message:
        m_a = sum_{b in N(a)} MLP_msg([h_a, h_b, e_ab])
    Update:
        h_a_next = MLP_update([h_a, m_a]) + h_a  (residual)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        msg_hidden_dim: Optional[int] = None,
        update_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr="add")
        msg_hidden_dim = msg_hidden_dim or hidden_dim
        update_hidden_dim = update_hidden_dim or hidden_dim

        self.mlp_msg = _build_mlp(
            in_dim=2 * hidden_dim + edge_dim,
            hidden_dim=msg_hidden_dim,
            out_dim=hidden_dim,
        )
        self.mlp_update = _build_mlp(
            in_dim=hidden_dim + hidden_dim,
            hidden_dim=update_hidden_dim,
            out_dim=hidden_dim,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        m = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        h_cat = torch.cat([x, m], dim=-1)
        h_new = self.mlp_update(h_cat)
        h_new = self.dropout(h_new)
        h_new = self.norm(h_new + x)
        return h_new

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        m_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_msg(m_in)


class PocketGNNEncoder(nn.Module):
    """
    Encoder for pocket atom graphs.

    Expects Data objects from PocketGraphBuilder:
        data.x: (A, F_atom)
        data.edge_index: (2, E)
        data.edge_attr: (E, F_edge)
        data.is_ligand: (A,) boolean mask
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        edge_dim: int = 9,
        num_layers: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                PocketMessageLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, data: Data) -> Tensor:
        """
        Args:
            data: torch_geometric.data.Data with x, edge_index, edge_attr.

        Returns:
            Atom embeddings of shape (A, hidden_dim).
        """
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return x

