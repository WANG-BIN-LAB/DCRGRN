"""DCR-GRN model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

from .decomposition import SharedPrivateFactorizer
from .encoders import (
    DirectedRoleEncoder,
    ExpressionGraphEncoder,
    GlobalTopologyEncoder,
    IdentityExpressionEncoder,
    LocalSubgraphEncoder,
)
from .fusion import CrossDomainCIBFusion, conditional_bottleneck_loss


SUPPORTED_VARIANTS = {
    "backbone",
    "dcrgrn",
}


class DCRGRN(nn.Module):
    """Dual-domain complementary representation learning for GRN inference."""

    def __init__(
        self,
        train_dataset,
        grn_size: int,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 32,
        num_layers: int = 2,
        variant: str = "dcrgrn",
    ):
        super().__init__()
        variant = str(variant).lower()
        if variant not in SUPPORTED_VARIANTS:
            choices = ", ".join(sorted(SUPPORTED_VARIANTS))
            raise ValueError(f"Unknown variant {variant!r}. Choose one of: {choices}")
        self.variant = variant
        self.uses_dual_domain_learning = variant != "backbone"

        self.local_encoder = LocalSubgraphEncoder(
            train_dataset,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            pooling_ratio=0.2,
        )
        self.knn_encoder = ExpressionGraphEncoder(
            in_channels=32,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )
        self.global_encoder = GlobalTopologyEncoder(
            input_dim=grn_size,
            hidden_dim=128,
            heads=8,
        )
        self.global_projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
        )
        self.identity_encoder = IdentityExpressionEncoder(grn_size, out_channels)
        self.directed_role_encoder = DirectedRoleEncoder(
            grn_size,
            hidden_dim=128,
            out_dim=out_channels,
        )
        self.directed_residual_gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
        )
        nn.init.zeros_(self.directed_residual_gate[-1].weight)
        nn.init.constant_(self.directed_residual_gate[-1].bias, -2.0)

        if self.uses_dual_domain_learning:
            self.local_head = nn.Linear(out_channels, 1)
            self.knn_head = nn.Linear(out_channels, 1)
            self.global_head = nn.Linear(out_channels, 1)
            self.expression_factorizer = SharedPrivateFactorizer(
                out_channels, condition_dim=0, dropout=0.1
            )
            self.topology_factorizer = SharedPrivateFactorizer(
                out_channels, condition_dim=4, dropout=0.1
            )
            self.degree_regularizer_head = nn.Sequential(
                nn.Linear(out_channels * 2 + 4, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 1),
            )
            self.cross_domain_fusion = CrossDomainCIBFusion(
                dim=out_channels,
                hidden_dim=out_channels,
                dropout=0.1,
            )
        else:
            self.local_evidence = nn.Linear(out_channels, 2)
            self.knn_evidence = nn.Linear(out_channels, 2)
            self.global_evidence = nn.Linear(out_channels, 2)
            self.identity_evidence = nn.Linear(out_channels, 2)

        self.last_details = None
        self.last_evidence = None

    @staticmethod
    def _to_undirected(edge_index):
        reverse = edge_index.flip(0)
        return torch.unique(torch.cat([edge_index, reverse], dim=1), dim=1)

    @staticmethod
    def _pairwise_sum(node_embeddings, target_nodes):
        selected = node_embeddings[target_nodes]
        return selected[0::2] + selected[1::2]

    def _encode_views(self, data, grn_data, knn_graph):
        knn_nodes = self.knn_encoder(knn_graph.x, knn_graph.edge_index)
        directed_edges = getattr(
            grn_data, "edge_index_directed", grn_data.edge_index
        )
        undirected_edges = self._to_undirected(directed_edges)
        global_nodes = self.global_encoder(
            grn_data.x,
            undirected_edges,
            edge_attr=getattr(grn_data, "edge_attr", None),
        )
        global_nodes = self.global_projection(global_nodes)

        local_pair = self.local_encoder(data)
        knn_pair = self._pairwise_sum(knn_nodes, data.target_nodes)
        global_pair = self._pairwise_sum(global_nodes, data.target_nodes)

        outgoing, incoming = self.directed_role_encoder(
            grn_data.x, directed_edges
        )
        pair_nodes = data.target_nodes.view(-1, 2)
        outgoing_tf = outgoing[pair_nodes[:, 0]]
        incoming_target = incoming[pair_nodes[:, 1]]
        tf_role, target_role = self.directed_role_encoder.project_tf_target(
            outgoing_tf, incoming_target
        )
        directed_score = self.directed_role_encoder.directed_score(
            outgoing_tf, incoming_target
        )
        direction_weight = torch.sigmoid(directed_score).unsqueeze(-1)
        directed_pair = direction_weight * tf_role + (1.0 - direction_weight) * target_role
        residual_weight = torch.sigmoid(
            self.directed_residual_gate(torch.cat([global_pair, directed_pair], dim=-1))
        )
        global_pair = (
            (1.0 - residual_weight) * global_pair
            + residual_weight * directed_pair
        )

        identity_nodes = self.identity_encoder(grn_data.x)
        identity_pair = self._pairwise_sum(identity_nodes, data.target_nodes)
        return {
            "local": local_pair,
            "knn": knn_pair,
            "global": global_pair,
            "identity": identity_pair,
            "directed_edges": directed_edges,
        }

    def forward(self, data, grn_data, knn_graph):
        views = self._encode_views(data, grn_data, knn_graph)
        if not self.uses_dual_domain_learning:
            return self._forward_backbone(views)
        return self._forward_dcr(data, views)

    def _forward_backbone(self, views):
        local_alpha = F.softplus(self.local_evidence(views["local"])) + 1.0
        knn_alpha = F.softplus(self.knn_evidence(views["knn"])) + 1.0
        global_alpha = F.softplus(self.global_evidence(views["global"])) + 1.0
        identity_alpha = F.softplus(self.identity_evidence(views["identity"])) + 1.0
        fused_alpha = self._combine_evidence(
            [local_alpha, knn_alpha, global_alpha, identity_alpha]
        )
        self.last_evidence = [
            global_alpha,
            local_alpha,
            knn_alpha,
            identity_alpha,
            fused_alpha,
        ]
        return (
            self._alpha_to_logit(global_alpha),
            self._alpha_to_logit(local_alpha),
            self._alpha_to_logit(knn_alpha),
            self._alpha_to_logit(fused_alpha),
        )

    def _forward_dcr(self, data, views):
        structure = self._structure_features(data, views["directed_edges"])
        expression = self.expression_factorizer(
            views["identity"], views["knn"], aggregate="attention"
        )
        topology = self.topology_factorizer(
            views["local"],
            views["global"],
            condition=structure,
            aggregate="sum",
        )

        degree_regularizer_weight = torch.sigmoid(
            self.degree_regularizer_head(
                torch.cat(
                    [
                        topology["shared"],
                        topology["private_b"],
                        structure,
                    ],
                    dim=-1,
                )
            )
        )
        low_degree_ratio = self._low_degree_ratio(structure)

        fusion = self.cross_domain_fusion(
            expression,
            topology,
            structure,
        )
        self.last_details = {
            "expression": expression,
            "topology": topology,
            "fusion": fusion,
            "degree_regularizer_weight": degree_regularizer_weight,
            "low_degree_ratio": low_degree_ratio,
            "structure": structure,
            "pair_nodes": data.target_nodes.view(-1, 2),
        }
        return (
            self.global_head(views["global"]),
            self.local_head(views["local"]),
            self.knn_head(views["knn"]),
            fusion["logit"],
        )

    @staticmethod
    def _structure_features(data, directed_edge_index):
        pair_nodes = data.target_nodes.view(-1, 2).to(directed_edge_index.device)
        num_nodes = int(data.target_nodes.max().item()) + 1
        if hasattr(data, "sub_nodes") and data.sub_nodes.numel() > 0:
            num_nodes = max(num_nodes, int(data.sub_nodes.max().item()) + 1)
        if directed_edge_index.numel() > 0:
            num_nodes = max(num_nodes, int(directed_edge_index.max().item()) + 1)

        out_degree = degree(
            directed_edge_index[0], num_nodes=num_nodes, dtype=torch.float
        )
        in_degree = degree(
            directed_edge_index[1], num_nodes=num_nodes, dtype=torch.float
        )
        tf_nodes = pair_nodes[:, 0].long()
        target_nodes = pair_nodes[:, 1].long()
        log_out_degree = torch.log1p(out_degree[tf_nodes]).view(-1, 1)
        log_in_degree = torch.log1p(in_degree[target_nodes]).view(-1, 1)

        if hasattr(data, "batch"):
            subgraph_size = torch.bincount(
                data.batch, minlength=pair_nodes.size(0)
            ).to(log_out_degree.device, dtype=torch.float).view(-1, 1)
        else:
            subgraph_size = torch.ones_like(log_out_degree)
        log_subgraph_size = torch.log1p(subgraph_size)

        if num_nodes <= 2000 and directed_edge_index.numel() > 0:
            adjacency = torch.zeros(
                (num_nodes, num_nodes),
                device=directed_edge_index.device,
                dtype=torch.float,
            )
            adjacency[
                directed_edge_index[0].long(), directed_edge_index[1].long()
            ] = 1.0
            two_hop_support = (
                adjacency[tf_nodes] * adjacency[:, target_nodes].t()
            ).sum(dim=-1, keepdim=True)
        else:
            two_hop_support = torch.zeros_like(log_out_degree)

        return torch.cat(
            [
                log_out_degree,
                log_in_degree,
                log_subgraph_size,
                torch.log1p(two_hop_support),
            ],
            dim=-1,
        )

    @staticmethod
    def _low_degree_ratio(structure, tau=2.0):
        out_degree = torch.expm1(structure[:, 0:1]).clamp(min=0.0)
        in_degree = torch.expm1(structure[:, 1:2]).clamp(min=0.0)
        return 0.5 * (
            tau / (out_degree + tau) + tau / (in_degree + tau)
        )

    def auxiliary_loss(self):
        return self.global_encoder.auxiliary_loss()

    def decomposition_loss(
        self,
        labels,
        lambda_reconstruction=0.05,
        lambda_orthogonality=0.01,
        lambda_shared=0.01,
        lambda_contrastive=0.02,
        lambda_routing=0.005,
    ):
        if not self.uses_dual_domain_learning or self.last_details is None:
            return next(self.parameters()).new_tensor(0.0)
        expression = self.last_details["expression"]
        topology = self.last_details["topology"]
        fusion = self.last_details["fusion"]
        reconstruction = (
            expression["reconstruction_loss"]
            + topology["reconstruction_loss"]
        )
        orthogonality = (
            expression["orthogonality_loss"]
            + topology["orthogonality_loss"]
        )
        shared_consistency = (
            expression["shared_consistency_loss"]
            + topology["shared_consistency_loss"]
        )
        contrastive = self._supervised_contrastive_loss(
            fusion["z_expression"] + fusion["z_topology"], labels
        )
        routing = F.l1_loss(
            self.last_details["degree_regularizer_weight"],
            self.last_details["low_degree_ratio"].detach(),
        )
        return (
            lambda_reconstruction * reconstruction
            + lambda_orthogonality * orthogonality
            + lambda_shared * shared_consistency
            + lambda_contrastive * contrastive
            + lambda_routing * routing
        )

    def bottleneck_loss(
        self,
        labels,
        lambda_kl=5e-4,
        lambda_gain=0.01,
        lambda_entropy=0.001,
    ):
        if not self.uses_dual_domain_learning or self.last_details is None:
            return next(self.parameters()).new_tensor(0.0)
        return conditional_bottleneck_loss(
            self.last_details["fusion"],
            labels,
            lambda_kl=lambda_kl,
            lambda_gain=lambda_gain,
            lambda_entropy=lambda_entropy,
        )

    @staticmethod
    def _supervised_contrastive_loss(features, labels, temperature=0.2):
        labels = labels.view(-1).long()
        if features.size(0) < 3 or labels.unique().numel() < 2:
            return features.new_tensor(0.0)
        normalized = F.normalize(features, dim=-1)
        logits = normalized @ normalized.t() / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(
            normalized.size(0), device=normalized.device, dtype=torch.bool
        )
        positive_mask = (labels[:, None] == labels[None, :]) & ~self_mask
        valid = positive_mask.sum(dim=1) > 0
        if not valid.any():
            return features.new_tensor(0.0)
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_probability = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8)
        )
        loss = -(
            log_probability * positive_mask.to(log_probability.dtype)
        ).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
        return loss[valid].mean()

    @staticmethod
    def _alpha_to_logit(alpha):
        return torch.log(alpha[:, 1:2] / (alpha[:, 0:1] + 1e-8) + 1e-8)

    @staticmethod
    def _belief_and_uncertainty(alpha):
        evidence = alpha - 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        return evidence / (strength + 1e-8), 2.0 / (strength + 1e-8)

    @classmethod
    def _combine_two_evidence(cls, alpha_a, alpha_b):
        belief_a, uncertainty_a = cls._belief_and_uncertainty(alpha_a)
        belief_b, uncertainty_b = cls._belief_and_uncertainty(alpha_b)
        conflict = (
            belief_a[:, 0:1] * belief_b[:, 1:2]
            + belief_a[:, 1:2] * belief_b[:, 0:1]
        )
        denominator = (1.0 - conflict).clamp(min=1e-6)
        belief = (
            belief_a * belief_b
            + belief_a * uncertainty_b
            + belief_b * uncertainty_a
        ) / denominator
        uncertainty = uncertainty_a * uncertainty_b / denominator
        strength = 2.0 / (uncertainty + 1e-8)
        return belief * strength + 1.0

    @classmethod
    def _combine_evidence(cls, alphas):
        combined = alphas[0]
        for alpha in alphas[1:]:
            combined = cls._combine_two_evidence(combined, alpha)
        return combined

    @staticmethod
    def _dirichlet_kl_to_uniform(alpha):
        uniform = torch.ones_like(alpha)
        alpha_strength = alpha.sum(dim=1, keepdim=True)
        uniform_strength = uniform.sum(dim=1, keepdim=True)
        log_beta_alpha = torch.lgamma(alpha_strength) - torch.lgamma(alpha).sum(
            dim=1, keepdim=True
        )
        log_beta_uniform = torch.lgamma(uniform).sum(
            dim=1, keepdim=True
        ) - torch.lgamma(uniform_strength)
        return (
            (alpha - uniform)
            * (torch.digamma(alpha) - torch.digamma(alpha_strength))
        ).sum(dim=1) + (log_beta_alpha + log_beta_uniform).squeeze(1)

    def evidential_loss(self, labels, epoch=None):
        if self.last_evidence is None:
            return next(self.parameters()).new_tensor(0.0)
        labels = labels.view(-1).long()
        one_hot = F.one_hot(labels, num_classes=2).to(
            dtype=self.last_evidence[0].dtype,
            device=self.last_evidence[0].device,
        )
        annealing = 1.0 if epoch is None else min(1.0, float(epoch) / 10.0)
        total = self.last_evidence[0].new_tensor(0.0)
        for alpha in self.last_evidence:
            strength = alpha.sum(dim=1, keepdim=True)
            cross_entropy = (
                one_hot * (torch.digamma(strength) - torch.digamma(alpha))
            ).sum(dim=1)
            adjusted = one_hot + (1.0 - one_hot) * alpha
            total = total + (
                cross_entropy
                + 0.001
                * annealing
                * self._dirichlet_kl_to_uniform(adjusted)
            ).mean()
        return total
