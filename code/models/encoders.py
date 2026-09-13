"""Feature encoders used by DCR-GRN."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GATConv, GCNConv, JumpingKnowledge, TransformerConv
from torch_geometric.utils import degree


class PatchySANPooling(nn.Module):
    """Convert variable-size enclosing subgraphs into fixed-size vectors."""

    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pooled = []
        for graph_id in batch.unique():
            indices = (batch == graph_id).nonzero(as_tuple=True)[0]
            subgraph_features = x[indices]
            importance = subgraph_features.norm(p=2, dim=1)
            top_indices = torch.topk(
                importance,
                k=min(self.k, subgraph_features.size(0)),
                largest=True,
            ).indices
            patch = subgraph_features[top_indices]
            if patch.size(0) < self.k:
                padding = torch.zeros(
                    self.k - patch.size(0),
                    x.size(1),
                    device=x.device,
                    dtype=x.dtype,
                )
                patch = torch.cat([patch, padding], dim=0)
            pooled.append(patch.reshape(-1))
        return torch.stack(pooled)


class LocalSubgraphEncoder(nn.Module):
    """Encode two-hop DRNL subgraphs with GCN and PatchySAN pooling."""

    def __init__(
        self,
        train_dataset,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        pooling_ratio: float = 0.2,
    ):
        super().__init__()
        self.convs = ModuleList([GCNConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        if pooling_ratio < 1:
            sizes = sorted(data.num_nodes for data in train_dataset)
            quantile_index = int(math.ceil(pooling_ratio * len(sizes))) - 1
            pooling_size = max(10, sizes[quantile_index])
        else:
            pooling_size = int(pooling_ratio)

        self.pool = PatchySANPooling(pooling_size)
        self.projection = nn.Sequential(
            Linear(hidden_channels * pooling_size, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
        return self.projection(self.pool(x, data.batch))


class ExpressionGraphEncoder(nn.Module):
    """Encode the expression KNN graph using GAT and LSTM-based JK aggregation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
    ):
        super().__init__()
        self.convs = ModuleList([GATConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels))
        self.jumping_knowledge = JumpingKnowledge(
            mode="lstm",
            channels=hidden_channels,
            num_layers=num_layers,
        )
        self.output_projection = Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        layer_outputs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            layer_outputs.append(x)
        return self.output_projection(self.jumping_knowledge(layer_outputs))


class GlobalTopologyEncoder(nn.Module):
    """Encode the directed prior GRN with PCConv-style low-degree correction."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, heads: int = 8):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.conv1 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.self1 = nn.Linear(hidden_dim, hidden_dim)
        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate1 = nn.Linear(hidden_dim + 1, 1)
        self.gate2 = nn.Linear(hidden_dim + 1, 1)
        self.low_degree_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.low_degree_scale = nn.Parameter(torch.tensor(0.10))
        self._auxiliary_loss = None

    @staticmethod
    def _degree_features(edge_index, num_nodes, device):
        total_degree = degree(
            edge_index[0], num_nodes=num_nodes, dtype=torch.float
        ).to(device)
        total_degree = total_degree + degree(
            edge_index[1], num_nodes=num_nodes, dtype=torch.float
        ).to(device)
        log_degree = torch.log1p(total_degree).view(-1, 1)
        log_degree = (log_degree - log_degree.mean()) / log_degree.std().clamp(min=1e-6)
        return total_degree, log_degree

    @staticmethod
    def _filter_layer(x, edge_index, conv, self_linear, gate, norm, log_degree):
        neighbor_features = conv(x, edge_index)
        self_features = self_linear(x)
        mixing_weight = torch.sigmoid(gate(torch.cat([x, log_degree], dim=-1)))
        filtered = mixing_weight * neighbor_features + (1.0 - mixing_weight) * self_features
        return norm(filtered + x)

    def _low_degree_correction(self, x, total_degree, log_degree):
        nonzero_degree = total_degree[total_degree > 0]
        if nonzero_degree.numel() > 0:
            threshold = torch.clamp(torch.median(nonzero_degree) / 3.0, min=1.0)
        else:
            threshold = torch.tensor(1.0, device=x.device)
        low_degree_weight = torch.sigmoid((threshold - total_degree).view(-1, 1))
        correction = self.low_degree_mlp(torch.cat([x, log_degree], dim=-1))
        return x + self.low_degree_scale * low_degree_weight * correction, low_degree_weight

    @staticmethod
    def _edge_reconstruction_loss(x, edge_index, low_degree_weight):
        if edge_index.numel() == 0:
            return x.new_tensor(0.0)
        source, target = edge_index
        mask = (low_degree_weight[source, 0] > 0.5) | (
            low_degree_weight[target, 0] > 0.5
        )
        selected = torch.nonzero(mask, as_tuple=False).view(-1)[:4096]
        if selected.numel() == 0:
            return x.new_tensor(0.0)
        source = source[selected]
        target = target[selected]
        negative_target = torch.roll(target, shifts=1)
        scale = x.size(-1) ** 0.5
        positive_score = (x[source] * x[target]).sum(dim=-1) / scale
        negative_score = (x[source] * x[negative_target]).sum(dim=-1) / scale
        return 0.01 * (
            F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()
        )

    def forward(self, x, edge_index, edge_attr=None):
        del edge_attr
        self._auxiliary_loss = x.new_tensor(0.0)
        total_degree, log_degree = self._degree_features(
            edge_index, x.size(0), x.device
        )
        x = F.relu(self.input_projection(x))
        x = self._filter_layer(
            x,
            edge_index,
            self.conv1,
            self.self1,
            self.gate1,
            self.norm1,
            log_degree,
        )
        x, low_degree_weight = self._low_degree_correction(
            x, total_degree, log_degree
        )
        x = self._filter_layer(
            x,
            edge_index,
            self.conv2,
            self.self2,
            self.gate2,
            self.norm2,
            log_degree,
        )
        if self.training:
            self._auxiliary_loss = self._edge_reconstruction_loss(
                x, edge_index, low_degree_weight
            )
        return x

    def auxiliary_loss(self):
        if self._auxiliary_loss is None:
            return self.low_degree_scale.new_tensor(0.0)
        return self._auxiliary_loss


class IdentityExpressionEncoder(nn.Module):
    """Preserve TF and target expression identities without graph propagation."""

    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.network(x)


class DirectedRoleEncoder(nn.Module):
    """Represent regulators and targets with separate directed message passing."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 32):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.outgoing_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.outgoing_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.incoming_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.incoming_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.outgoing_norm = nn.LayerNorm(hidden_dim)
        self.incoming_norm = nn.LayerNorm(hidden_dim)
        self.direction_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.direction_weight)
        self.tf_projection = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, out_dim)
        )
        self.target_projection = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, out_dim)
        )

    def forward(self, x, edge_index):
        base = F.relu(self.input_projection(x))
        outgoing = F.relu(self.outgoing_conv1(base, edge_index))
        outgoing = F.relu(self.outgoing_conv2(outgoing, edge_index))
        outgoing = self.outgoing_norm(outgoing + base)

        reverse_edge_index = edge_index.flip(0)
        incoming = F.relu(self.incoming_conv1(base, reverse_edge_index))
        incoming = F.relu(self.incoming_conv2(incoming, reverse_edge_index))
        incoming = self.incoming_norm(incoming + base)
        return outgoing, incoming

    def directed_score(self, outgoing, incoming):
        return ((outgoing @ self.direction_weight) * incoming).sum(dim=-1)

    def project_tf_target(self, outgoing, incoming):
        return self.tf_projection(outgoing), self.target_projection(incoming)
