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


class ProteinMessageLayer(MessagePassing):
    """
    Message passing layer for residue graphs using edge-aware MLP messages.

    Message:
        m_r = sum_{s in N(r)} MLP_msg([h_r, h_s, e_rs])
    Update:
        h_r_next = MLP_update([h_r, m_r]) + h_r  (residual)
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
        # x: (R, D), edge_index: (2, E), edge_attr: (E, edge_dim)
        m = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        h_cat = torch.cat([x, m], dim=-1)
        h_new = self.mlp_update(h_cat)
        h_new = self.dropout(h_new)
        h_new = self.norm(h_new + x)
        return h_new

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # x_i, x_j: (E, D), edge_attr: (E, edge_dim)
        m_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_msg(m_in)


class ProteinGNNEncoder(nn.Module):
    """
    Encoder for protein residue graphs.

    Expects Data objects from ProteinGraphBuilder/CorrelationEdgeBuilder:
        data.x: (R, F_res)
        data.edge_index: (2, E)
        data.edge_attr: (E, F_edge)
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 128,
        edge_dim: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                ProteinMessageLayer(
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
            Residue embeddings of shape (R, hidden_dim).
        """
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return x

