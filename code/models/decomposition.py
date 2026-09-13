"""Shared-private representation factorization for DCR-GRN."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPrivateFactorizer(nn.Module):
    """Factor two candidate-edge views into shared and orthogonal private parts."""

    def __init__(self, dim: int, condition_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        self.projection_a = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.projection_b = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.mask_network = nn.Sequential(
            nn.Linear(dim * 4 + condition_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.reconstruction_a = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.reconstruction_b = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.private_key = nn.Linear(dim, dim)
        self.private_query = nn.Linear(dim, dim)

    @staticmethod
    def _orthogonalize(private, shared):
        denominator = (shared * shared).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        projection = (private * shared).sum(dim=-1, keepdim=True) / denominator
        return private - projection * shared

    @staticmethod
    def _squared_cosine(a, b):
        return F.cosine_similarity(a, b, dim=-1).pow(2).mean()

    def forward(self, view_a, view_b, condition=None, aggregate="attention"):
        projected_a = self.projection_a(view_a)
        projected_b = self.projection_b(view_b)
        mask_inputs = [
            projected_a,
            projected_b,
            torch.abs(projected_a - projected_b),
            projected_a * projected_b,
        ]
        if self.condition_dim > 0:
            if condition is None:
                condition = projected_a.new_zeros(
                    projected_a.size(0), self.condition_dim
                )
            mask_inputs.append(condition)

        mask = self.mask_network(torch.cat(mask_inputs, dim=-1))
        shared = mask * 0.5 * (projected_a + projected_b)
        private_a = self._orthogonalize((1.0 - mask) * projected_a, shared)
        private_b = self._orthogonalize((1.0 - mask) * projected_b, shared)

        reconstructed_a = self.reconstruction_a(torch.cat([shared, private_a], dim=-1))
        reconstructed_b = self.reconstruction_b(torch.cat([shared, private_b], dim=-1))
        reconstruction_loss = F.mse_loss(reconstructed_a, projected_a)
        reconstruction_loss = reconstruction_loss + F.mse_loss(
            reconstructed_b, projected_b
        )
        orthogonality_loss = self._squared_cosine(shared, private_a)
        orthogonality_loss = orthogonality_loss + self._squared_cosine(
            shared, private_b
        )
        orthogonality_loss = orthogonality_loss + self._squared_cosine(
            private_a, private_b
        )
        shared_consistency_loss = (
            1.0
            - F.cosine_similarity(
                mask * projected_a, mask * projected_b, dim=-1
            )
        ).mean()

        if aggregate == "attention":
            query = self.private_query(shared).unsqueeze(1)
            private_tokens = torch.stack([private_a, private_b], dim=1)
            keys = self.private_key(private_tokens)
            scores = (query * keys).sum(dim=-1) / math.sqrt(float(self.dim))
            gates = torch.softmax(scores, dim=-1)
            output = shared + (gates.unsqueeze(-1) * private_tokens).sum(dim=1)
        elif aggregate == "sum":
            gates = None
            output = shared + private_a + private_b
        else:
            raise ValueError(f"Unsupported aggregation mode: {aggregate}")

        return {
            "projected_a": projected_a,
            "projected_b": projected_b,
            "shared": shared,
            "private_a": private_a,
            "private_b": private_b,
            "output": output,
            "mask": mask,
            "gates": gates,
            "reconstruction_loss": reconstruction_loss,
            "orthogonality_loss": orthogonality_loss,
            "shared_consistency_loss": shared_consistency_loss,
        }
