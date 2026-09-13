"""Conditional information bottleneck and cross-domain fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalBottleneck(nn.Module):
    """Map a component to a diagonal Gaussian latent variable."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.statistics = nn.Linear(input_dim, latent_dim * 2)

    def forward(self, x):
        mean, log_variance = self.statistics(x).chunk(2, dim=-1)
        log_variance = log_variance.clamp(min=-6.0, max=2.0)
        if self.training:
            noise = torch.randn_like(mean)
            latent = mean + noise * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        kl_per_dimension = -0.5 * (
            1.0 + log_variance - mean.pow(2) - log_variance.exp()
        )
        return {"latent": latent, "kl": kl_per_dimension.sum(dim=-1).mean()}


class CrossDomainCIBFusion(nn.Module):
    """Build expression/topology latents and decode a regulatory edge logit."""

    def __init__(self, dim: int = 32, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.shared_expression_bottleneck = ConditionalBottleneck(dim, dim)
        self.shared_topology_bottleneck = ConditionalBottleneck(dim, dim)
        self.identity_private_bottleneck = ConditionalBottleneck(dim * 2, dim)
        self.knn_private_bottleneck = ConditionalBottleneck(dim * 2, dim)
        self.local_private_bottleneck = ConditionalBottleneck(dim * 2 + 4, dim)
        self.global_private_bottleneck = ConditionalBottleneck(dim * 2 + 4, dim)

        self.expression_router = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.topology_router = nn.Sequential(
            nn.Linear(dim * 3 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.shared_head = self._prediction_head(dim * 4, hidden_dim, dropout)
        self.decoder = self._prediction_head(dim * 5 + 4, hidden_dim, dropout)

    @staticmethod
    def _prediction_head(input_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, expression, topology, structure):
        shared_expression = self.shared_expression_bottleneck(expression["shared"])
        shared_topology = self.shared_topology_bottleneck(topology["shared"])
        identity_private = self.identity_private_bottleneck(
            torch.cat(
                [expression["private_a"], shared_expression["latent"]], dim=-1
            )
        )
        knn_private = self.knn_private_bottleneck(
            torch.cat(
                [expression["private_b"], shared_expression["latent"]], dim=-1
            )
        )
        local_private = self.local_private_bottleneck(
            torch.cat(
                [topology["private_a"], shared_topology["latent"], structure],
                dim=-1,
            )
        )
        global_private = self.global_private_bottleneck(
            torch.cat(
                [topology["private_b"], shared_topology["latent"], structure],
                dim=-1,
            )
        )

        c_expression = shared_expression["latent"]
        c_topology = shared_topology["latent"]
        u_identity = identity_private["latent"]
        u_knn = knn_private["latent"]
        u_local = local_private["latent"]
        u_global = global_private["latent"]

        expression_weights = torch.softmax(
            self.expression_router(
                torch.cat([c_expression, u_identity, u_knn], dim=-1)
            ),
            dim=-1,
        )
        topology_weights = torch.softmax(
            self.topology_router(
                torch.cat([c_topology, u_local, u_global, structure], dim=-1)
            ),
            dim=-1,
        )
        z_expression = (
            c_expression
            + expression_weights[:, 0:1] * u_identity
            + expression_weights[:, 1:2] * u_knn
        )
        z_topology = (
            c_topology
            + topology_weights[:, 0:1] * u_local
            + topology_weights[:, 1:2] * u_global
        )

        shared_logit = self.shared_head(
            torch.cat(
                [
                    c_expression,
                    c_topology,
                    torch.abs(c_expression - c_topology),
                    c_expression * c_topology,
                ],
                dim=-1,
            )
        )
        edge_logit = self.decoder(
            torch.cat(
                [
                    z_expression,
                    z_topology,
                    torch.abs(z_expression - z_topology),
                    z_expression * z_topology,
                    c_expression * c_topology,
                    structure,
                ],
                dim=-1,
            )
        )

        posteriors = [
            shared_expression,
            shared_topology,
            identity_private,
            knn_private,
            local_private,
            global_private,
        ]
        return {
            "logit": edge_logit,
            "shared_logit": shared_logit,
            "z_expression": z_expression,
            "z_topology": z_topology,
            "shared_expression": c_expression,
            "shared_topology": c_topology,
            "expression_weights": expression_weights,
            "topology_weights": topology_weights,
            "kl": sum(posterior["kl"] for posterior in posteriors),
        }


def conditional_bottleneck_loss(
    fusion,
    labels,
    lambda_kl=5e-4,
    lambda_gain=0.01,
    lambda_entropy=0.001,
):
    """Regularize bottleneck capacity, private utility, and routing diversity."""

    full_bce = F.binary_cross_entropy_with_logits(
        fusion["logit"].view(-1), labels, reduction="none"
    )
    shared_bce = F.binary_cross_entropy_with_logits(
        fusion["shared_logit"].view(-1), labels, reduction="none"
    )
    loss = lambda_kl * fusion["kl"]
    loss = loss + lambda_gain * F.softplus(full_bce - shared_bce).mean()
    for weights in (fusion["expression_weights"], fusion["topology_weights"]):
        entropy = -(
            weights * torch.log(weights.clamp(min=1e-8))
        ).sum(dim=-1).mean()
        loss = loss - lambda_entropy * entropy
    return loss
