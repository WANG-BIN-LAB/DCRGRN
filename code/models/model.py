import math
import os
import random
from itertools import combinations, product
from torch_geometric.utils import to_undirected, degree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Conv1d, MaxPool1d, Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv, global_sort_pool, GATConv, MessagePassing, GATv2Conv, global_mean_pool, \
    JumpingKnowledge, TransformerConv
from torch_geometric.nn import GCN, GAT
from torch.nn import MultiheadAttention
from torch_geometric.utils import add_remaining_self_loops
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PatchySANPooling(torch.nn.Module):
    """
    Selects top-k nodes based on node importance (L2 norm) to create a fixed-size representation for subgraphs.
    """
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x, batch):

        pooled = []
        for i in batch.unique():
            idx = (batch == i).nonzero(as_tuple=True)[0]
            x_sub = x[idx]
            degrees = x_sub.norm(p=2, dim=1)# Calculate node importance based on feature norm
            topk = torch.topk(degrees, k=min(self.k, x_sub.size(0)), largest=True).indices
            patch = x_sub[topk]
            # Padding if subgraph is smaller than k
            if patch.size(0) < self.k:
                pad = torch.zeros(self.k - patch.size(0), x.size(1), device=x.device)
                patch = torch.cat([patch, pad], dim=0)
            pooled.append(patch.view(-1))
        return torch.stack(pooled)



class SubgraphEncoder(nn.Module):
    """
        View 1: Encodes local subgraphs using GCN layers and PatchySAN pooling.
        Used to capture local topological patterns around target links.
    """
    def __init__(self,train_dataset, in_channels, hidden_channels, out_channels, num_layers,k):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Determine k for pooling based on dataset statistics
        if k < 1:
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        self.pool = PatchySANPooling(self.k)  # new!
        self.proj_head = nn.Sequential(
            Linear(hidden_channels * k, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels)
        )
    def encode(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.pool(x, batch)
        return self.proj_head(x)
    def forward(self, data):
        z1 = self.encode(data.x, data.edge_index, data.batch)
        return z1

class MultiScaleGNN(torch.nn.Module):
    """
    View 2: Multi-scale GNN using Jumping Knowledge (JK).
    Captures information from different neighborhood ranges (hops) on the KNN graph.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels))

        # JK-Net aggregates features from all layers
        self.jk = JumpingKnowledge(mode='lstm', channels=hidden_channels, num_layers=num_layers)
        self.out_proj = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)
        x = self.jk(xs)  # [N, hidden]
        return self.out_proj(x)



class GRNTransformer(nn.Module):
    """
    View 3 GRN Transformer.

    Scheme 10 implementation:
    - edgeconf / evidential_edgeconf: TransformerConv consumes scalar edge_attr
      (co-expression confidence when no native score exists).
    - A netType FiLM modulation h = gamma(type) * h + beta(type) is applied
      after each GRN layer, matching the proposal's prior-type embedding.
    """
    def __init__(self, input_dim, hidden_dim=128, heads=4, variant='baseline', net_type='Specific'):
        super().__init__()
        self.variant = str(variant)
        self.edgeconf_enabled = self.variant in ('edgeconf', 'evidential_edgeconf')
        edge_dim = 1 if self.edgeconf_enabled else None
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.conv1 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.type_to_id = {'Specific': 0, 'Non-Specific': 1, 'STRING': 2, 'LOF_GOF': 3, 'LOF/GOF': 3, 'LOF-GOF': 3}
        type_id = self.type_to_id.get(str(net_type), 0)
        self.register_buffer('net_type_id', torch.tensor([type_id], dtype=torch.long), persistent=False)
        if self.edgeconf_enabled:
            self.type_emb = nn.Embedding(4, 16)
            self.film1 = nn.Linear(16, hidden_dim * 2)
            self.film2 = nn.Linear(16, hidden_dim * 2)

    def _film(self, x, film):
        t = self.type_emb(self.net_type_id.to(x.device)).view(1, -1)
        gamma, beta = film(t).chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma)
        beta = torch.tanh(beta)
        return gamma * x + beta

    def _conv(self, conv, x, edge_index, edge_attr):
        if self.edgeconf_enabled and edge_attr is not None:
            return conv(x, edge_index, edge_attr=edge_attr)
        return conv(x, edge_index)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.linear(x))
        h1 = self._conv(self.conv1, x, edge_index, edge_attr)
        x = self.norm1(h1 + x)
        if self.edgeconf_enabled:
            x = self._film(x, self.film1)
        h2 = self._conv(self.conv2, x, edge_index, edge_attr)
        x = self.norm2(h2 + x)
        if self.edgeconf_enabled:
            x = self._film(x, self.film2)
        return x

class AdaptiveGRNTransformer(nn.Module):
    """Heterophily/low-degree aware GRN branch.

    pcconv_grn: each layer separates self filtering and neighbor aggregation,
    then learns a node-wise gate from the current feature and log-degree. This
    follows the heterophily-aware idea that a node should decide whether to
    trust itself or its neighbors instead of always smoothing.

    lowdeg_grn: adds a low-degree expert residual and a deterministic
    self-supervised edge reconstruction term on edges incident to low-degree
    nodes, so isolated/near-isolated genes get a trainable correction rather
    than being washed out by sparse message passing.
    """
    def __init__(self, input_dim, hidden_dim=128, heads=8, variant='pcconv_grn', aux_weight=0.02):
        super().__init__()
        self.variant = variant
        self.use_pcconv = variant in ('pcconv_grn', 'pcconv_lowdeg')
        self.use_lowdeg = variant in ('lowdeg_grn', 'pcconv_lowdeg')
        self.aux_weight = aux_weight
        # Keep common modules in the same order as the original GRNTransformer
        # so shared initializations remain as comparable as possible.
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.conv1 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.self1 = nn.Linear(hidden_dim, hidden_dim)
        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate1 = nn.Linear(hidden_dim + 1, 1)
        self.gate2 = nn.Linear(hidden_dim + 1, 1)
        self.low_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.low_scale = nn.Parameter(torch.tensor(0.10))
        self._aux_loss = None

    def _degree_feature(self, edge_index, num_nodes, device):
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float).to(device) + \
              degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float).to(device)
        logd = torch.log1p(deg).view(-1, 1)
        logd = (logd - logd.mean()) / (logd.std().clamp(min=1e-6))
        return deg, logd

    def _layer(self, x, edge_index, conv, self_lin, gate, norm, logd):
        neigh = conv(x, edge_index)
        if self.use_pcconv:
            self_part = self_lin(x)
            g = torch.sigmoid(gate(torch.cat([x, logd], dim=-1)))
            h = g * neigh + (1.0 - g) * self_part
        else:
            h = neigh
        return norm(h + x)

    def _apply_lowdeg(self, x, deg, logd):
        if not self.use_lowdeg:
            return x, None
        nz = deg[deg > 0]
        if nz.numel() > 0:
            thr = torch.clamp(torch.median(nz) / 3.0, min=1.0)
        else:
            thr = torch.tensor(1.0, device=x.device)
        low_weight = torch.sigmoid((thr - deg).view(-1, 1))
        correction = self.low_mlp(torch.cat([x, logd], dim=-1))
        return x + self.low_scale * low_weight * correction, low_weight

    def _edge_aux(self, x, edge_index, low_weight):
        if (not self.use_lowdeg) or low_weight is None or edge_index.numel() == 0:
            return x.new_tensor(0.0)
        src, dst = edge_index[0], edge_index[1]
        mask = ((low_weight[src, 0] > 0.50) | (low_weight[dst, 0] > 0.50))
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if idx.numel() == 0:
            return x.new_tensor(0.0)
        idx = idx[:4096]
        s, d = src[idx], dst[idx]
        # deterministic pseudo-negative: rotate destinations inside this edge set.
        nd = torch.roll(d, shifts=1)
        pos = (x[s] * x[d]).sum(dim=-1) / (x.size(-1) ** 0.5)
        neg = (x[s] * x[nd]).sum(dim=-1) / (x.size(-1) ** 0.5)
        return self.aux_weight * (F.softplus(-pos).mean() + F.softplus(neg).mean())

    def forward(self, x, edge_index, edge_attr=None):
        # edge_attr is accepted for compatibility with edge-confidence GRNTransformer calls;
        # low-degree/PCConv filtering intentionally ignores it.
        self._aux_loss = x.new_tensor(0.0)
        deg, logd = self._degree_feature(edge_index, x.size(0), x.device)
        x = F.relu(self.linear(x))
        x = self._layer(x, edge_index, self.conv1, self.self1, self.gate1, self.norm1, logd)
        x, low_weight = self._apply_lowdeg(x, deg, logd)
        x = self._layer(x, edge_index, self.conv2, self.self2, self.gate2, self.norm2, logd)
        if self.use_lowdeg and self.training:
            self._aux_loss = self._edge_aux(x, edge_index, low_weight)
        return x

    def auxiliary_loss(self):
        return self._aux_loss if self._aux_loss is not None else self.low_scale.new_tensor(0.0)


class IdentityEncoder(nn.Module):
    """Topology-free identity-preserving branch.

    It bypasses edge_index and keeps direct gene expression identity, which is
    complementary to the low-degree/PCConv GRN branch and evidential fusion.
    """
    def __init__(self, input_dim, out_dim, hidden_dim=128, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DualRoleEncoder(nn.Module):
    """Directed TF/Target dual-role encoder for GRN branch augmentation."""
    def __init__(self, input_dim, hidden_dim=128, out_dim=32, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.out_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.in_gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.in_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dir_weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.dir_weight)
        self.tf_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, out_dim))
        self.target_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.relu(self.in_proj(x))
        eo = F.relu(self.out_gcn1(h, edge_index))
        eo = F.relu(self.out_gcn2(eo, edge_index))
        eo = self.out_norm(eo + h)
        rev = edge_index.flip(0)
        ei = F.relu(self.in_gcn1(h, rev))
        ei = F.relu(self.in_gcn2(ei, rev))
        ei = self.in_norm(ei + h)
        emb = self.out_proj(torch.cat([eo, ei], dim=-1))
        return eo, ei, emb

    def directed_score(self, eo, ei):
        w = eo @ self.dir_weight
        return (w * ei).sum(dim=-1)

    def project_tf_target(self, eo, ei):
        return self.tf_head(eo), self.target_head(ei)


class SPORESharedPrivateDecomposer(nn.Module):
    """Candidate-edge shared/private decomposition for a pair of views."""

    def __init__(self, dim, cond_dim=0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.proj_a = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.proj_b = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.mask_mlp = nn.Sequential(
            nn.Linear(dim * 4 + cond_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.rec_a = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.rec_b = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.private_key = nn.Linear(dim, dim)
        self.private_query = nn.Linear(dim, dim)

    def _orthogonalize(self, u, c):
        denom = (c * c).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return u - ((u * c).sum(dim=-1, keepdim=True) / denom) * c

    def _cos2(self, a, b):
        return F.cosine_similarity(a, b, dim=-1).pow(2).mean()

    def forward(self, a, b, cond=None, mode='attention'):
        ah = self.proj_a(a)
        bh = self.proj_b(b)
        feat = [ah, bh, torch.abs(ah - bh), ah * bh]
        if self.cond_dim > 0:
            if cond is None:
                cond = ah.new_zeros(ah.size(0), self.cond_dim)
            feat.append(cond)
        mask = self.mask_mlp(torch.cat(feat, dim=-1))
        shared = mask * 0.5 * (ah + bh)
        ua = self._orthogonalize((1.0 - mask) * ah, shared)
        ub = self._orthogonalize((1.0 - mask) * bh, shared)

        rec_a = self.rec_a(torch.cat([shared, ua], dim=-1))
        rec_b = self.rec_b(torch.cat([shared, ub], dim=-1))
        rec_loss = F.mse_loss(rec_a, ah) + F.mse_loss(rec_b, bh)
        orth_loss = self._cos2(shared, ua) + self._cos2(shared, ub) + self._cos2(ua, ub)
        share_a = mask * ah
        share_b = mask * bh
        share_loss = (1.0 - F.cosine_similarity(share_a, share_b, dim=-1)).mean()

        if mode == 'attention':
            query = self.private_query(shared).unsqueeze(1)
            tokens = torch.stack([ua, ub], dim=1)
            keys = self.private_key(tokens)
            score = (query * keys).sum(dim=-1) / math.sqrt(float(self.dim))
            att = torch.softmax(score, dim=-1)
            out = shared + (att.unsqueeze(-1) * tokens).sum(dim=1)
            gates = att
        else:
            out = shared + ua + ub
            gates = None

        return {
            'a_hat': ah,
            'b_hat': bh,
            'shared': shared,
            'u_a': ua,
            'u_b': ub,
            'out': out,
            'mask': mask,
            'gates': gates,
            'rec_loss': rec_loss,
            'orth_loss': orth_loss,
            'share_loss': share_loss,
        }


class SPOREFusion(nn.Module):
    """Orthogonal synergy-aware evidential fusion with DS anchoring."""

    def __init__(self, dim=32, rank=4, lambda_max=0.2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.lambda_max = float(lambda_max)
        self.align_E = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.align_T = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.low_E = nn.Linear(dim, rank)
        self.low_T = nn.Linear(dim, rank)
        self.syn_proj = nn.Sequential(nn.Linear(rank, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.private_key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim * 2, dim)
        self.beta = nn.Parameter(torch.tensor(-1.0))
        self.gamma = nn.Parameter(torch.tensor(-1.0))
        self.evi_spore = nn.Linear(dim, 2)
        self.lambda_head = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def _orthogonalize_to(self, u, basis):
        out = u
        for b in basis:
            denom = (b * b).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            out = out - ((out * b).sum(dim=-1, keepdim=True) / denom) * b
        return out

    def forward(self, z_E, z_T, private_tokens, alpha_ds):
        e = self.align_E(z_E)
        t = self.align_T(z_T)
        cF = 0.5 * (e + t)
        sET = self.syn_proj(self.low_E(e) * self.low_T(t))
        ortho_tokens = torch.stack([self._orthogonalize_to(u, [cF, sET]) for u in private_tokens], dim=1)
        q = self.query(torch.cat([cF, sET], dim=-1)).unsqueeze(1)
        k = self.private_key(ortho_tokens)
        att = torch.softmax((q * k).sum(dim=-1) / math.sqrt(float(self.dim)), dim=-1)
        private_mix = (att.unsqueeze(-1) * ortho_tokens).sum(dim=1)
        z = cF + torch.sigmoid(self.beta) * sET + torch.sigmoid(self.gamma) * private_mix

        alpha_spore = F.softplus(self.evi_spore(z)) + 1.0
        e_ds = alpha_ds - 1.0
        e_spore = alpha_spore - 1.0
        lam = self.lambda_max * torch.sigmoid(self.lambda_head(torch.cat([cF, sET], dim=-1)))
        alpha_final = (1.0 - lam) * e_ds + lam * e_spore + 1.0
        return {
            'alpha_spore': alpha_spore,
            'alpha_final': alpha_final,
            'lambda': lam,
            'cF': cF,
            'sET': sET,
            'private_tokens': ortho_tokens,
            'private_att': att,
        }


class SPORENFFusion(nn.Module):
    """SPORE non-evidential fusion: shared/private/synergy features -> logit."""

    def __init__(self, dim=32, rank=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.align_E = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.align_T = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.low_E = nn.Linear(dim, rank)
        self.low_T = nn.Linear(dim, rank)
        self.syn_proj = nn.Sequential(
            nn.Linear(rank, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.private_key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim * 2, dim)
        self.beta = nn.Parameter(torch.tensor(-1.0))
        self.gamma = nn.Parameter(torch.tensor(-1.0))
        self.out_norm = nn.LayerNorm(dim)
        self.pred_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def _orthogonalize_to(self, u, basis):
        out = u
        for b in basis:
            denom = (b * b).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            out = out - ((out * b).sum(dim=-1, keepdim=True) / denom) * b
        return out

    def forward(self, z_E, z_T, private_tokens):
        e = self.align_E(z_E)
        t = self.align_T(z_T)
        cF = 0.5 * (e + t)
        sET = self.syn_proj(self.low_E(e) * self.low_T(t))
        ortho_tokens = torch.stack([self._orthogonalize_to(u, [cF, sET]) for u in private_tokens], dim=1)
        q = self.query(torch.cat([cF, sET], dim=-1)).unsqueeze(1)
        k = self.private_key(ortho_tokens)
        att = torch.softmax((q * k).sum(dim=-1) / math.sqrt(float(self.dim)), dim=-1)
        private_mix = (att.unsqueeze(-1) * ortho_tokens).sum(dim=1)
        z = self.out_norm(cF + torch.sigmoid(self.beta) * sET + torch.sigmoid(self.gamma) * private_mix)
        logit = self.pred_head(z)
        return {
            'logit': logit,
            'z': z,
            'cF': cF,
            'sET': sET,
            'private_tokens': ortho_tokens,
            'private_att': att,
        }


def _entmax15_bisect(logits, dim=-1, n_iter=25, eps=1e-6):
    """Numerically stable entmax-1.5 gate used by CURO.

    This keeps the "sparse but differentiable routing" behavior required by the
    CURO scheme without adding an external dependency. It falls back naturally
    to sparse probability vectors whose entries sum to one.
    """
    X = logits - logits.max(dim=dim, keepdim=True).values
    X = X / 2.0
    max_val = X.max(dim=dim, keepdim=True).values
    tau_lo = max_val - 1.0
    tau_hi = max_val
    for _ in range(n_iter):
        tau_m = (tau_lo + tau_hi) / 2.0
        p_m = torch.clamp(X - tau_m, min=0.0).pow(2)
        f_m = p_m.sum(dim=dim, keepdim=True) - 1.0
        tau_lo = torch.where(f_m > 0, tau_m, tau_lo)
        tau_hi = torch.where(f_m <= 0, tau_m, tau_hi)
    p = torch.clamp(X - tau_hi, min=0.0).pow(2)
    return p / p.sum(dim=dim, keepdim=True).clamp(min=eps)


class SPOREUOTFusion(nn.Module):
    """Global Unbalanced Optimal-Transport regulatory fusion.

    The module converts pair-level expression/topology states into a small
    mini-batch TF-by-target transport plan. The transport mass acts as a
    listwise regulatory prior and is combined with a residual classifier.
    """
    def __init__(self, dim=32, hidden=32, sinkhorn_iter=6, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.sinkhorn_iter = sinkhorn_iter
        self.feat = nn.Sequential(
            nn.Linear(dim * 4 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.affinity = nn.Linear(hidden, 1)
        self.supply = nn.Linear(hidden, 1)
        self.demand = nn.Linear(hidden, 1)
        self.residual = nn.Linear(hidden, 1)
        self.mass_scale = nn.Parameter(torch.tensor(0.5))
        self.tau = nn.Parameter(torch.tensor(0.0))

    def _scatter_mean(self, values, inv, n):
        out = values.new_zeros(n, values.size(-1))
        cnt = values.new_zeros(n, 1)
        out.index_add_(0, inv, values)
        cnt.index_add_(0, inv, torch.ones_like(values[:, :1]))
        return out / cnt.clamp(min=1.0)

    def forward(self, z_E, z_T, degree_cond, pair_nodes):
        h = self.feat(torch.cat([z_E, z_T, torch.abs(z_E - z_T), z_E * z_T, degree_cond], dim=-1))
        raw_aff = self.affinity(h).view(-1)
        res = self.residual(h)

        tf_unique, tf_inv = torch.unique(pair_nodes[:, 0].long(), sorted=True, return_inverse=True)
        tg_unique, tg_inv = torch.unique(pair_nodes[:, 1].long(), sorted=True, return_inverse=True)
        n_tf, n_tg = tf_unique.numel(), tg_unique.numel()
        if n_tf == 0 or n_tg == 0:
            return {'logit': res, 'z': h, 'transport_logit': res.detach() * 0.0}

        tf_h = self._scatter_mean(h, tf_inv, n_tf)
        tg_h = self._scatter_mean(h, tg_inv, n_tg)
        a = F.softplus(self.supply(tf_h).view(-1)) + 1e-4
        b = F.softplus(self.demand(tg_h).view(-1)) + 1e-4
        a = a / a.sum().clamp(min=1e-6)
        b = b / b.sum().clamp(min=1e-6)

        score = h.new_full((n_tf, n_tg), -20.0)
        score[tf_inv, tg_inv] = raw_aff
        tau = F.softplus(self.tau) + 0.35
        K = torch.exp((score - score.max()).clamp(min=-30.0, max=20.0) / tau) + 1e-8
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        # Unbalanced exponents keep low-degree TF/TG from being over-constrained.
        unbalance = 0.75
        for _ in range(self.sinkhorn_iter):
            u = (a / (K @ v + 1e-8)).pow(unbalance)
            v = (b / (K.t() @ u + 1e-8)).pow(unbalance)
        plan = u[:, None] * K * v[None, :]
        pair_mass = plan[tf_inv, tg_inv].clamp(min=1e-8)
        mass_logit = torch.log(pair_mass) - torch.log(pair_mass.mean().detach().clamp(min=1e-8))
        logit = res + torch.sigmoid(self.mass_scale) * mass_logit.view(-1, 1)
        return {
            'logit': logit,
            'z': h,
            'transport_logit': mass_logit.view(-1, 1),
            'raw_affinity': raw_aff.view(-1, 1),
            'tf_inv': tf_inv,
            'tg_inv': tg_inv,
        }


class SPOREEPIDFusion(nn.Module):
    """Edge-level variational information-atom decomposition.

    Four small atom heads model expression-only, topology-only, shared and
    synergistic regulatory evidence. A learned atom gate composes the final
    score while auxiliary losses keep every atom predictive but non-collapsed.
    """
    def __init__(self, dim=32, hidden=32, dropout=0.1):
        super().__init__()
        self.e_proj = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.t_proj = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.shared_proj = nn.Sequential(nn.Linear(dim * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.syn_proj = nn.Sequential(nn.Linear(dim * 4, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.atom_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(4)])
        self.atom_gate = nn.Sequential(
            nn.Linear(hidden * 4 + 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )
        self.final_bias = nn.Linear(hidden, 1)

    def forward(self, z_E, z_T, degree_cond, pair_nodes=None):
        e = self.e_proj(z_E)
        t = self.t_proj(z_T)
        c = self.shared_proj(torch.cat([z_E, z_T], dim=-1))
        s = self.syn_proj(torch.cat([z_E, z_T, torch.abs(z_E - z_T), z_E * z_T], dim=-1))
        atoms = [e, t, c, s]
        atom_logits = torch.cat([head(atom) for head, atom in zip(self.atom_heads, atoms)], dim=-1)
        gate_logits = self.atom_gate(torch.cat(atoms + [degree_cond], dim=-1))
        weights = torch.softmax(gate_logits, dim=-1)
        final = (weights * atom_logits).sum(dim=-1, keepdim=True) + 0.1 * self.final_bias(c)
        return {
            'logit': final,
            'z': c + s,
            'atom_logits': atom_logits,
            'atom_weights': weights,
            'atoms': atoms,
        }


class SPORECUROFusion(nn.Module):
    """Counterfactual utility-routed regulatory fusion.

    The router compares expression-only, topology-only and joint counterfactual
    scores before sparsely selecting residual experts with entmax-1.5.
    """
    def __init__(self, dim=32, hidden=32, num_experts=4, dropout=0.1):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(dim * 4 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cf_e = nn.Linear(dim, 1)
        self.cf_t = nn.Linear(dim, 1)
        self.cf_joint = nn.Linear(dim * 2, 1)
        self.router = nn.Sequential(
            nn.Linear(hidden + 3 + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
            for _ in range(num_experts)
        ])

    def forward(self, z_E, z_T, degree_cond, pair_nodes=None):
        h = self.base(torch.cat([z_E, z_T, torch.abs(z_E - z_T), z_E * z_T, degree_cond], dim=-1))
        le = self.cf_e(z_E)
        lt = self.cf_t(z_T)
        lj = self.cf_joint(torch.cat([z_E, z_T], dim=-1))
        util = torch.cat([
            torch.abs(lj - le),
            torch.abs(lj - lt),
            lj - torch.maximum(le, lt),
        ], dim=-1)
        route_logits = self.router(torch.cat([h, util, degree_cond], dim=-1))
        route = _entmax15_bisect(route_logits, dim=-1)
        expert_logits = torch.cat([expert(h) for expert in self.experts], dim=-1)
        final = (route * expert_logits).sum(dim=-1, keepdim=True)
        return {
            'logit': final,
            'z': h,
            'route': route,
            'route_logits': route_logits,
            'expert_logits': expert_logits,
            'cf_logits': torch.cat([le, lt, lj], dim=-1),
        }


class SPOREMCGFusion(nn.Module):
    """Mixed-curvature regulatory geometry.

    Euclidean interaction captures local linear compatibility while a
    Poincare-ball component captures hierarchical TF-target organization.
    """
    def __init__(self, dim=32, hidden=32, dropout=0.1):
        super().__init__()
        self.e_euc = nn.Linear(dim, hidden)
        self.t_euc = nn.Linear(dim, hidden)
        self.e_hyp = nn.Linear(dim, hidden)
        self.t_hyp = nn.Linear(dim, hidden)
        self.curve_head = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.mix_head = nn.Sequential(nn.Linear(hidden * 2 + 4, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.residual = nn.Sequential(nn.Linear(dim * 4 + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _ball_project(self, x, c):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        radius = (1.0 / torch.sqrt(c)).clamp(max=10.0)
        return torch.tanh(torch.sqrt(c) * norm) * x / (torch.sqrt(c) * norm).clamp(min=1e-8) * radius * 0.95

    def _poincare_dist(self, x, y, c):
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        diff2 = ((x - y) * (x - y)).sum(dim=-1, keepdim=True)
        denom = (1.0 - c * x2).clamp(min=1e-5) * (1.0 - c * y2).clamp(min=1e-5)
        z = 1.0 + 2.0 * c * diff2 / denom
        return torch.acosh(z.clamp(min=1.0 + 1e-5)) / torch.sqrt(c).clamp(min=1e-5)

    def forward(self, z_E, z_T, degree_cond, pair_nodes=None):
        ee = F.normalize(self.e_euc(z_E), dim=-1)
        tt = F.normalize(self.t_euc(z_T), dim=-1)
        euclid_score = (ee * tt).sum(dim=-1, keepdim=True)
        c = F.softplus(self.curve_head(degree_cond)) + 0.05
        eh = self._ball_project(torch.tanh(self.e_hyp(z_E)), c)
        th = self._ball_project(torch.tanh(self.t_hyp(z_T)), c)
        hyp_score = -self._poincare_dist(eh, th, c)
        mix = torch.sigmoid(self.mix_head(torch.cat([ee, tt, degree_cond], dim=-1)))
        res = self.residual(torch.cat([z_E, z_T, torch.abs(z_E - z_T), z_E * z_T, degree_cond], dim=-1))
        logit = res + self.scale * (mix * euclid_score + (1.0 - mix) * hyp_score)
        return {
            'logit': logit,
            'z': mix * ee + (1.0 - mix) * tt,
            'euclid_score': euclid_score,
            'hyp_score': hyp_score,
            'curvature': c,
            'mix': mix,
        }


class SPORECIBFusion(nn.Module):
    """Task-conditioned information-bottleneck SPORE fusion.

    This keeps the existing SPORE shared/private decomposition, but decodes the
    expression and topology domain outputs through stochastic low-capacity
    shared/private variables. Private variables are useful only when they reduce
    the conditional prediction loss beyond the shared baseline.
    """
    def __init__(self, dim=32, hidden=32, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.cE_stats = nn.Linear(dim, dim * 2)
        self.cT_stats = nn.Linear(dim, dim * 2)
        self.uid_stats = nn.Linear(dim * 2, dim * 2)
        self.uknn_stats = nn.Linear(dim * 2, dim * 2)
        self.uL_stats = nn.Linear(dim * 2 + 4, dim * 2)
        self.uG_stats = nn.Linear(dim * 2 + 4, dim * 2)
        self.expr_gate = nn.Sequential(nn.Linear(dim * 3, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.topo_gate = nn.Sequential(nn.Linear(dim * 3 + 4, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.expr_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.topo_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.shared_head = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.full_head = nn.Sequential(
            # [zE, zT, |zE-zT|, zE*zT, cE*cT, degree_cond]
            nn.Linear(dim * 5 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Ablation decoder: deterministic, parameter-matched fusion without stochastic CIB sampling/KL.
        self.noib_head = nn.Sequential(
            nn.Linear(dim * 5 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Conditional topology prompt used by EdgePrompt/TopoPure variants.
        # Appended after the original CIB modules so that retained weights that
        # still match are loaded without changing earlier initialization order.
        self.edge_prompt = nn.Sequential(
            nn.Linear(4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.edge_prompt_gate = nn.Sequential(
            nn.Linear(dim + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.topo_pure_head = nn.Sequential(
            nn.Linear(dim * 4 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.topo_residual_scale = nn.Parameter(torch.tensor(-2.2))
        # Next-five CIB decoders. They are appended after existing CIB modules
        # to preserve initialization order of earlier retained/CIB parameters.
        self.crsb_e_proj = nn.Linear(dim, dim)
        self.crsb_t_proj = nn.Linear(dim, dim)
        self.crsb_velocity = nn.Sequential(
            nn.Linear(dim * 2 + 8 + 1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.crsb_bridge_proj = nn.Sequential(
            nn.Linear(dim * 2 + 8, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.crsb_norm = nn.LayerNorm(dim)
        self.crsb_head = nn.Sequential(
            nn.Linear(dim * 3 + 5, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.crsb_temp = nn.Parameter(torch.tensor(0.0))
        self.crsb_steps = 4
        self.uti_i1 = nn.Sequential(nn.Linear(dim * 2, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.uti_e2 = nn.Linear(dim, dim)
        self.uti_t2 = nn.Linear(dim, dim)
        self.uti_e3 = nn.Linear(dim, dim)
        self.uti_t3 = nn.Linear(dim, dim)
        self.uti_j3 = nn.Linear(dim * 2, dim)
        self.uti_rank_out = nn.Sequential(nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))
        self.uti_order = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.uti_head = nn.Sequential(
            nn.Linear(dim * 3 + 5, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.ugrt_feat = nn.Sequential(
            nn.Linear(dim * 4 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.ugrt_cost = nn.Linear(hidden, 1)
        self.ugrt_supply = nn.Linear(hidden + 1, 1)
        self.ugrt_demand = nn.Linear(hidden + 1, 1)
        self.ugrt_mass_scale = nn.Parameter(torch.tensor(0.25))
        self.ugrt_tau = nn.Parameter(torch.tensor(0.0))
        self.dehg_feat = nn.Sequential(
            nn.Linear(dim * 4 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dehg_tf_msg = nn.Linear(hidden, hidden)
        self.dehg_tg_msg = nn.Linear(hidden, hidden)
        self.dehg_motif_msg = nn.Linear(hidden, hidden)
        self.dehg_gate = nn.Sequential(nn.Linear(hidden * 4 + 4, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.dehg_head = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.dehg_scale = nn.Parameter(torch.tensor(-2.8))
        self.per_strength = nn.Sequential(
            nn.Linear(dim * 4 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.per_alpha = nn.Parameter(torch.tensor(3.0))
        self.per_bias = nn.Parameter(torch.tensor(-1.0))
        # DualControl-CIB: scalar difficulty controller
        # r_ij = sigmoid(w_C*C + w_U*U + w_D*D + w_V*V + b).
        # The features are detached in the loss so the predictor cannot create
        # artificial uncertainty/conflict to escape compression.
        self.dual_w = nn.Parameter(torch.tensor([1.0, 0.5, 1.0, 1.0]))
        self.dual_b = nn.Parameter(torch.tensor(-2.0))
        self.variant = 'spore_cib'

    def _sample(self, stats):
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(min=-6.0, max=2.0)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu
        kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl = kl_dim.sum(dim=-1).mean()
        return z, kl, mu, logvar, kl_dim

    def _binary_entropy(self, p):
        p = p.clamp(min=1e-8, max=1.0 - 1e-8)
        return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / math.log(2.0)

    def _scatter_mean(self, values, inv, n):
        out = values.new_zeros(n, values.size(-1))
        cnt = values.new_zeros(n, 1)
        out.index_add_(0, inv, values)
        cnt.index_add_(0, inv, torch.ones_like(values[:, :1]))
        return out / cnt.clamp(min=1.0)

    def _ugrt_decode(self, zE, zT, base_logit, expr_logit, topo_logit, degree_cond, pair_nodes):
        if pair_nodes is None or pair_nodes.numel() == 0:
            return {'logit': base_logit, 'transport_logit': base_logit.detach() * 0.0}
        h = self.ugrt_feat(torch.cat([zE, zT, torch.abs(zE - zT), zE * zT, degree_cond], dim=-1))
        pE = torch.sigmoid(expr_logit.detach())
        pT = torch.sigmoid(topo_logit.detach())
        unc = (torch.abs(pE - pT) + 0.5 * (self._binary_entropy(pE) + self._binary_entropy(pT))).view(-1)
        raw_score = self.ugrt_cost(h).view(-1) - 0.50 * unc.detach()
        tf_unique, tf_inv = torch.unique(pair_nodes[:, 0].long(), sorted=True, return_inverse=True)
        tg_unique, tg_inv = torch.unique(pair_nodes[:, 1].long(), sorted=True, return_inverse=True)
        n_tf, n_tg = tf_unique.numel(), tg_unique.numel()
        if n_tf == 0 or n_tg == 0:
            return {'logit': base_logit, 'transport_logit': base_logit.detach() * 0.0}
        tf_h = self._scatter_mean(h, tf_inv, n_tf)
        tg_h = self._scatter_mean(h, tg_inv, n_tg)
        tf_deg = self._scatter_mean(degree_cond[:, 0:1], tf_inv, n_tf)
        tg_deg = self._scatter_mean(degree_cond[:, 1:2], tg_inv, n_tg)
        a = F.softplus(self.ugrt_supply(torch.cat([tf_h, tf_deg], dim=-1)).view(-1)) + 1e-4
        b = F.softplus(self.ugrt_demand(torch.cat([tg_h, tg_deg], dim=-1)).view(-1)) + 1e-4
        a = a / a.sum().clamp(min=1e-6)
        b = b / b.sum().clamp(min=1e-6)
        score = h.new_full((n_tf, n_tg), -30.0)
        score[tf_inv, tg_inv] = raw_score
        tau = F.softplus(self.ugrt_tau) + 0.35
        K = torch.exp((score - score.max()).clamp(min=-35.0, max=20.0) / tau) + 1e-8
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        unbalance = 0.70
        for _ in range(8):
            u = (a / (K @ v + 1e-8)).pow(unbalance)
            v = (b / (K.t() @ u + 1e-8)).pow(unbalance)
        plan = u[:, None] * K * v[None, :]
        mass = plan[tf_inv, tg_inv].clamp(min=1e-8)
        mass_logit = torch.log(mass) - torch.log(mass.mean().detach().clamp(min=1e-8))
        # Keep a small safe base component so batch transport is not forced to be
        # the only globally comparable score, avoiding the previous GRT failure.
        logit = 0.75 * base_logit + torch.sigmoid(self.ugrt_mass_scale) * mass_logit.view(-1, 1)
        return {
            'logit': logit,
            'transport_logit': mass_logit.view(-1, 1),
            'raw_affinity': raw_score.view(-1, 1),
            'tf_inv': tf_inv,
            'tg_inv': tg_inv,
            'supply': a,
            'demand': b,
            'mass': mass.view(-1, 1),
        }

    def _dehg_decode(self, zE, zT, base_logit, degree_cond, pair_nodes):
        h = self.dehg_feat(torch.cat([zE, zT, torch.abs(zE - zT), zE * zT, degree_cond], dim=-1))
        if pair_nodes is None or pair_nodes.numel() == 0:
            return {'logit': base_logit, 'hyper_residual': h.detach() * 0.0}
        tf_unique, tf_inv = torch.unique(pair_nodes[:, 0].long(), sorted=True, return_inverse=True)
        tg_unique, tg_inv = torch.unique(pair_nodes[:, 1].long(), sorted=True, return_inverse=True)
        n_tf, n_tg = tf_unique.numel(), tg_unique.numel()
        tf_msg = self.dehg_tf_msg(self._scatter_mean(h, tf_inv, n_tf))[tf_inv]
        tg_msg = self.dehg_tg_msg(self._scatter_mean(h, tg_inv, n_tg))[tg_inv]
        # Lightweight dynamic motif proxy: use local-subgraph support and common
        # neighbor support already stored in degree_cond as a state-dependent
        # soft hyperedge, instead of rebuilding Local encoder internals.
        motif_key = torch.round((degree_cond[:, 2:3] + degree_cond[:, 3:4]).detach() * 2.0) / 2.0
        _, motif_inv = torch.unique(motif_key.view(-1), sorted=True, return_inverse=True)
        motif_msg = self.dehg_motif_msg(self._scatter_mean(h, motif_inv, int(motif_inv.max().item()) + 1))[motif_inv]
        gate = torch.softmax(self.dehg_gate(torch.cat([h, tf_msg, tg_msg, motif_msg, degree_cond], dim=-1)), dim=-1)
        coop = gate[:, 2:3] * motif_msg
        comp = gate[:, 0:1] * tf_msg + gate[:, 1:2] * tg_msg
        residual = coop - comp
        h2 = h + torch.sigmoid(self.dehg_scale) * residual
        logit = base_logit + self.dehg_head(h2)
        return {
            'logit': logit,
            'hyper_residual': residual,
            'hyper_gate': gate,
            'tf_inv': tf_inv,
            'tg_inv': tg_inv,
            'motif_inv': motif_inv,
        }

    def forward(self, expr_pack, topo_pack, z_T, degree_cond, pair_nodes=None):
        variant = str(getattr(self, 'variant', 'spore_cib')).lower()
        degree_cond_eff = degree_cond.new_zeros(degree_cond.shape) if variant in ('spore_cib_wo_degreecond', 'spore_cib_wo_topodeg') else degree_cond

        if variant == 'spore_cib_wo_ib':
            # Deterministic bottleneck control: same domain inputs and comparable decoder,
            # but no stochastic sampling, no KL, and no shared/private utility term.
            cE = expr_pack['shared']
            cT = topo_pack['shared']
            uid = expr_pack['u_a']
            uknn = expr_pack['u_b']
            uL = topo_pack['u_a']
            uG = topo_pack['u_b']
            k_cE = k_cT = k_uid = k_uknn = k_uL = k_uG = cE.new_tensor(0.0)
            mu_cE, mu_cT, mu_uid, mu_uknn, mu_uL, mu_uG = cE, cT, uid, uknn, uL, uG
            kd_cE = kd_cT = kd_uid = kd_uknn = kd_uL = kd_uG = cE.new_zeros(cE.size(0), cE.size(1))
        else:
            cE, k_cE, mu_cE, _, kd_cE = self._sample(self.cE_stats(expr_pack['shared']))
            cT, k_cT, mu_cT, _, kd_cT = self._sample(self.cT_stats(topo_pack['shared']))
            uid, k_uid, mu_uid, _, kd_uid = self._sample(self.uid_stats(torch.cat([expr_pack['u_a'], cE], dim=-1)))
            uknn, k_uknn, mu_uknn, _, kd_uknn = self._sample(self.uknn_stats(torch.cat([expr_pack['u_b'], cE], dim=-1)))
            uL, k_uL, mu_uL, _, kd_uL = self._sample(self.uL_stats(torch.cat([topo_pack['u_a'], cT, degree_cond_eff], dim=-1)))
            uG, k_uG, mu_uG, _, kd_uG = self._sample(self.uG_stats(torch.cat([topo_pack['u_b'], cT, degree_cond_eff], dim=-1)))

        # Domain-level ablations. They preserve dimensionality but remove the claimed decomposition.
        if variant == 'spore_cib_wo_edecomp':
            cE = 0.5 * (cE + uid + uknn)
            uid = uid.new_zeros(uid.shape)
            uknn = uknn.new_zeros(uknn.shape)
        if variant in ('spore_cib_wo_tdecomp', 'spore_cib_wo_topodeg'):
            cT = 0.5 * (cT + uL + uG)
            uL = uL.new_zeros(uL.shape)
            uG = uG.new_zeros(uG.shape)

        gE = torch.softmax(self.expr_gate(torch.cat([cE, uid, uknn], dim=-1)), dim=-1)
        gT = torch.softmax(self.topo_gate(torch.cat([cT, uL, uG, degree_cond_eff], dim=-1)), dim=-1)
        zE = cE + gE[:, 0:1] * uid + gE[:, 1:2] * uknn
        zT = cT + gT[:, 0:1] * uL + gT[:, 1:2] * uG
        prompt_gate = None
        prompt_vec = None
        if variant in ('spore_cib_edgeprompt', 'spore_cib_topopure'):
            # EdgePrompt: use only pre-branch structural signals, not dataset ID.
            # Low-degree / low-common-neighbor pairs get more topology adaptation,
            # while high-support pairs keep the original CIB representation.
            out_d = torch.expm1(degree_cond[:, 0:1]).clamp(min=0.0)
            in_d = torch.expm1(degree_cond[:, 1:2]).clamp(min=0.0)
            low_ratio = 0.5 * (2.0 / (out_d + 2.0) + 2.0 / (in_d + 2.0))
            prompt_vec = torch.tanh(self.edge_prompt(degree_cond_eff))
            prompt_gate = torch.sigmoid(self.edge_prompt_gate(torch.cat([zT, degree_cond_eff], dim=-1)))
            zT = zT + 0.15 * low_ratio.detach() * prompt_gate * prompt_vec
        expr_logit = self.expr_head(zE)
        topo_logit = self.topo_head(zT)
        shared_logit = self.shared_head(torch.cat([cE, cT, torch.abs(cE - cT), cE * cT], dim=-1))
        full_logit = self.full_head(torch.cat([zE, zT, torch.abs(zE - zT), zE * zT, cE * cT, degree_cond_eff], dim=-1))
        if variant == 'spore_cib_wo_ib':
            full_logit = self.noib_head(torch.cat([zE, zT, torch.abs(zE - zT), zE * zT, cE * cT, degree_cond_eff], dim=-1))
        topo_pure_logit = self.topo_pure_head(torch.cat([zT, cT, uL, uG, degree_cond_eff], dim=-1))
        if variant == 'spore_cib_topopure':
            # Keep expression/topology CIB as the main decision boundary and add
            # only a small topology-pure residual, preventing the old failure mode
            # where topology rewrites the expression domain.
            full_logit = full_logit + torch.sigmoid(self.topo_residual_scale) * topo_pure_logit
        if variant == 'spore_cib_crsb':
            x0 = self.crsb_e_proj(zE)
            x1 = self.crsb_t_proj(zT)
            pE = torch.sigmoid(expr_logit.detach())
            pT = torch.sigmoid(topo_logit.detach())
            entE = self._binary_entropy(pE)
            entT = self._binary_entropy(pT)
            conflict = torch.abs(pE - pT)
            cos = F.cosine_similarity(zE.detach(), zT.detach(), dim=-1, eps=1e-8).unsqueeze(-1)
            bridge_cond = torch.cat([degree_cond_eff, conflict, cos, entE, entT], dim=-1)
            state = x0
            energy = state.new_zeros(state.size(0), 1)
            steps = max(1, int(getattr(self, 'crsb_steps', 4)))
            for step in range(steps):
                t = state.new_full((state.size(0), 1), float(step) / float(steps))
                target_pull = (x1 - state) / float(steps - step)
                v = torch.tanh(self.crsb_velocity(torch.cat([state, x1, bridge_cond, t], dim=-1)))
                delta = 0.5 * target_pull + 0.5 * v / float(steps)
                state = state + delta
                energy = energy + delta.pow(2).mean(dim=-1, keepdim=True)
            z_bridge = self.crsb_norm(0.5 * state + 0.5 * self.crsb_bridge_proj(torch.cat([x0, x1, bridge_cond], dim=-1)))
            bridge_logit = self.crsb_head(torch.cat([z_bridge, zE, zT, energy, degree_cond_eff], dim=-1))
            risk_gate = torch.exp(-energy / self.crsb_temp.exp().clamp(min=0.05, max=10.0)).clamp(0.0, 1.0)
            full_logit = risk_gate * bridge_logit + (1.0 - risk_gate) * full_logit
            bridge_state = z_bridge
        else:
            energy = None
            risk_gate = None
            bridge_state = None
        if variant == 'spore_cib_uti':
            pE = torch.sigmoid(expr_logit.detach())
            pT = torch.sigmoid(topo_logit.detach())
            entE = self._binary_entropy(pE)
            entT = self._binary_entropy(pT)
            post_var = torch.stack([lv.exp().mean(dim=-1, keepdim=True) for lv in []], dim=0).mean() if False else full_logit.detach().new_zeros(full_logit.size(0), 1)
            uncertainty = (0.35 * entE + 0.35 * entT + 0.20 * torch.abs(pE - pT) + 0.10 * post_var).detach()
            i1 = self.uti_i1(torch.cat([zE, zT], dim=-1))
            i2 = self.uti_e2(zE) * self.uti_t2(zT)
            e3 = self.uti_e3(zE)
            t3 = self.uti_t3(zT)
            j3 = self.uti_j3(torch.cat([zE, zT], dim=-1))
            i3 = self.uti_rank_out(e3 * t3 * j3)
            order_logits = self.uti_order(torch.cat([uncertainty, degree_cond_eff], dim=-1))
            order_w = torch.softmax(order_logits, dim=-1)
            z_inter = order_w[:, 0:1] * i1 + order_w[:, 1:2] * i2 + order_w[:, 2:3] * i3
            full_logit = self.uti_head(torch.cat([z_inter, zE, zT, uncertainty, degree_cond_eff], dim=-1))
            uti_order_w = order_w
            uti_interaction = z_inter
            uti_uncertainty = uncertainty
        else:
            uti_order_w = None
            uti_interaction = None
            uti_uncertainty = None
        if variant == 'spore_cib_ugrt':
            ugrt = self._ugrt_decode(zE, zT, full_logit, expr_logit, topo_logit, degree_cond_eff, pair_nodes)
            full_logit = ugrt['logit']
        else:
            ugrt = {}
        if variant == 'spore_cib_dehg':
            dehg = self._dehg_decode(zE, zT, full_logit, degree_cond_eff, pair_nodes)
            full_logit = dehg['logit']
        else:
            dehg = {}
        if variant == 'spore_cib_per':
            # PER-CIB needs real perturbation supervision for the full method.
            # Without it, this branch is only a signed executable edge head
            # screened by edge labels; train1/report must mark it as PER-static.
            a_signed = torch.tanh(self.per_strength(torch.cat([zE, zT, torch.abs(zE - zT), zE * zT, degree_cond_eff], dim=-1)))
            per_logit = self.per_alpha.abs().clamp(max=8.0) * torch.abs(a_signed) + self.per_bias
            full_logit = per_logit
        else:
            a_signed = None
        return {
            'logit': full_logit,
            'z': zE + zT,
            'expr_logit': expr_logit,
            'topo_logit': topo_logit,
            'topo_pure_logit': topo_pure_logit,
            'prompt_gate': prompt_gate,
            'prompt_vec': prompt_vec,
            'shared_logit': shared_logit,
            'zE': zE,
            'zT': zT,
            'cE': cE,
            'cT': cT,
            'private_tokens': [uid, uknn, uL, uG],
            'mu_tokens': [mu_cE, mu_cT, mu_uid, mu_uknn, mu_uL, mu_uG],
            'expr_gate': gE,
            'topo_gate': gT,
            'dual_w': self.dual_w,
            'dual_b': self.dual_b,
            'crsb_energy': energy,
            'crsb_gate': risk_gate,
            'crsb_bridge': bridge_state,
            'uti_order_w': uti_order_w,
            'uti_interaction': uti_interaction,
            'uti_uncertainty': uti_uncertainty,
            'ugrt': ugrt,
            'dehg': dehg,
            'per_signed_A': a_signed,
            'ib_kl': k_cE + k_cT + k_uid + k_uknn + k_uL + k_uG,
            'ib_kl_dim': torch.cat([kd_cE, kd_cT, kd_uid, kd_uknn, kd_uL, kd_uG], dim=-1),
        }


class SPOREHCSPFusion(nn.Module):
    """Hierarchical context-conditioned shared/private SPORE fusion.

    The full cross-cell-type HCSP design needs a joint multi-dataset loader.
    This screening implementation keeps the same functional decomposition
    inside one dataset: global program + sparse subset programs + residual
    cell-state program, all conditioned by pair/dataset structural context.
    """
    def __init__(self, dim=32, hidden=32, num_subset=3, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_subset = num_subset
        self.global_prog = nn.Sequential(nn.Linear(dim * 2, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.subset_prog = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 4 + 4, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim))
            for _ in range(num_subset)
        ])
        self.ctx_router = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, num_subset))
        self.residual_prog = nn.Sequential(nn.Linear(dim * 4 + 4, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.pred = nn.Sequential(nn.Linear(dim * 3 + 4, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _sparsemax(self, logits, dim=-1):
        z = logits - logits.mean(dim=dim, keepdim=True)
        zs = torch.sort(z, dim=dim, descending=True).values
        range_vals = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
        view = [1] * z.dim()
        view[dim] = -1
        range_vals = range_vals.view(*view)
        bound = 1 + range_vals * zs
        cumsum = zs.cumsum(dim)
        is_gt = bound > cumsum
        k = is_gt.sum(dim=dim, keepdim=True).clamp(min=1)
        tau = (cumsum.gather(dim, k - 1) - 1) / k.to(z.dtype)
        return torch.clamp(z - tau, min=0.0)

    def forward(self, z_E, z_T, degree_cond, pair_nodes=None):
        base = torch.cat([z_E, z_T, torch.abs(z_E - z_T), z_E * z_T, degree_cond], dim=-1)
        g = self.global_prog(torch.cat([z_E, z_T], dim=-1))
        subset_tokens = torch.stack([prog(base) for prog in self.subset_prog], dim=1)
        ctx_batch = torch.cat([
            degree_cond,
            degree_cond.mean(dim=0, keepdim=True).expand_as(degree_cond),
        ], dim=-1)
        alpha = self._sparsemax(self.ctx_router(ctx_batch), dim=-1)
        h = (alpha.unsqueeze(-1) * subset_tokens).sum(dim=1)
        r = self.residual_prog(base)
        z = g + h + r
        logit = self.pred(torch.cat([z, g, h, degree_cond], dim=-1))
        return {
            'logit': logit,
            'z': z,
            'global_prog': g,
            'subset_tokens': subset_tokens,
            'subset_alpha': alpha,
            'residual_prog': r,
        }


class ATFGRN(torch.nn.Module):
    """
    Main Model: Adaptive Multi-View Fusion Network.
    Integrates Local Subgraph, KNN Structure, and Global GRN views using Attention.
    """
    def __init__(self, train_dataset, grn_size, in_channels, hidden_channels, out_channels, num_layers, contrastive_margin=1.0, variant='baseline', net_type='Specific'):
        super(ATFGRN, self).__init__()
        self.variant = str(variant)
        self.spore_new5_variants = (
            'spore_uot', 'spore_epid', 'spore_curo', 'spore_cur', 'spore_mcg',
            'spore_cib', 'spore_cib_safecap', 'spore_cib_rankguard',
            'spore_cib_stablemask', 'spore_cib_scondrank', 'spore_cib_puauc',
            'spore_cib_edgeprompt', 'spore_cib_confdec', 'spore_cib_topopure',
            'spore_cib_dualcontrol', 'spore_cib_gradalign', 'spore_cib_stabledecomp',
            'spore_cib_crsb', 'spore_cib_uti', 'spore_cib_ugrt', 'spore_cib_dehg',
            'spore_cib_per',
            'spore_cib_wo_edecomp', 'spore_cib_wo_tdecomp',
            'spore_cib_wo_degreecond', 'spore_cib_wo_topodeg', 'spore_cib_wo_ib',
            'spore_hcsp'
        )
        self.spore_enabled = self.variant in ('spore_grn', 'spore', 'spore_nf') or self.variant in self.spore_new5_variants
        self.spore_nf_enabled = self.variant in ('spore_nf',) or self.variant in self.spore_new5_variants
        self.use_identity_path = (
            self.variant in ('evidential_pcconv_lowdeg_idpath', 'idpath_evidential_pcconv_lowdeg')
            or ('idpath' in self.variant and 'evidential' in self.variant)
            or self.spore_enabled
        )
        self.evidential_enabled = self.variant in (
            'evidential', 'evidential_edgeconf', 'evidential_lowdeg',
            'evidential_pcconv_lowdeg', 'evidential_pcconv_lowdeg_idpath',
            'idpath_evidential_pcconv_lowdeg', 'directed_idpath_evidential_pcconv_lowdeg',
            'spore_grn', 'spore'
        )
        self.use_directed_role = self.variant in ('directed_idpath_evidential_pcconv_lowdeg', 'spore_grn', 'spore', 'spore_nf') or self.variant in self.spore_new5_variants

        # 1. Subgraph Encoder (Local View)
        self.gclencoder = SubgraphEncoder(train_dataset,in_channels, hidden_channels, out_channels, num_layers=num_layers,k=0.2)
        # 2. Multi-Scale GNN (KNN Graph View)
        self.Mgcn = MultiScaleGNN(in_channels=32, hidden_channels=hidden_channels, out_channels=out_channels,
                                  num_layers=num_layers)
        # 3. GRN Transformer (Global View)
        self.mlp = nn.Sequential(
            nn.Linear(grn_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),

        )


        if self.variant == 'evidential_lowdeg':
            self.gcn = AdaptiveGRNTransformer(input_dim=grn_size, hidden_dim=128, heads=8, variant='lowdeg_grn', aux_weight=0.01)
        elif self.variant in (
            'evidential_pcconv_lowdeg', 'evidential_pcconv_lowdeg_idpath',
            'idpath_evidential_pcconv_lowdeg', 'directed_idpath_evidential_pcconv_lowdeg',
            'spore_grn', 'spore', 'spore_nf',
            'spore_uot', 'spore_epid', 'spore_curo', 'spore_cur', 'spore_mcg',
            'spore_cib', 'spore_cib_safecap', 'spore_cib_rankguard',
            'spore_cib_stablemask', 'spore_cib_scondrank', 'spore_cib_puauc',
            'spore_cib_edgeprompt', 'spore_cib_confdec', 'spore_cib_topopure',
            'spore_cib_dualcontrol', 'spore_cib_gradalign', 'spore_cib_stabledecomp',
            'spore_cib_crsb', 'spore_cib_uti', 'spore_cib_ugrt', 'spore_cib_dehg',
            'spore_cib_per',
            'spore_cib_wo_edecomp', 'spore_cib_wo_tdecomp',
            'spore_cib_wo_degreecond', 'spore_cib_wo_topodeg', 'spore_cib_wo_ib',
            'spore_hcsp'
        ):
            self.gcn = AdaptiveGRNTransformer(input_dim=grn_size, hidden_dim=128, heads=8, variant='pcconv_lowdeg', aux_weight=0.01)
        else:
            self.gcn = GRNTransformer(input_dim=grn_size, hidden_dim=128, heads=8, variant=self.variant, net_type=net_type)

        self.lin1 = Linear(out_channels, 1)
        self.lin2 = Linear(out_channels, 1)
        self.lin_g = Linear(out_channels, 1)
        self.lin_grn = Linear(128*2, 32)
        self.lin3 = Linear(out_channels, 1)
        self.lin4  = nn.Sequential(

            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),

        )
        # Fusion layer for final prediction
        self.fuse_mlp  = nn.Sequential(
            nn.Linear(out_channels * (4 if self.use_identity_path else 3), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )
        # Attention Mechanism Parameters
        self.att_1 = Linear(out_channels, 16)
        self.att_2 = Linear(out_channels, 16)
        self.att_grn = Linear(out_channels, 16)
        self.query = Linear(16, 1)

        # Scheme 3: Evidential / trusted multi-view fusion.
        # Each branch outputs Dirichlet evidence for binary classes; the fused
        # prediction is obtained by Dempster-Shafer evidence combination.
        if self.evidential_enabled:
            self.evi_1 = Linear(out_channels, 2)
            self.evi_2 = Linear(out_channels, 2)
            self.evi_g = Linear(out_channels, 2)

        # Identity-path modules are declared after the original modules so that
        # existing evidential/GRN layers keep their initialization order.
        if self.use_identity_path:
            self.identity_encoder = IdentityEncoder(grn_size, out_channels)
            self.att_id = Linear(out_channels, 16)
            self.lin_id = Linear(out_channels, 1)
            if self.evidential_enabled:
                self.evi_id = Linear(out_channels, 2)
        else:
            self.identity_encoder = None

        if self.use_directed_role:
            self.dual_role = DualRoleEncoder(grn_size, hidden_dim=128, out_dim=out_channels)
            self.directed_residual_gate = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 1),
            )
            nn.init.zeros_(self.directed_residual_gate[-1].weight)
            nn.init.constant_(self.directed_residual_gate[-1].bias, -2.0)
        else:
            self.dual_role = None
            self.directed_residual_gate = None

        if self.spore_enabled:
            self.spore_expr = SPORESharedPrivateDecomposer(out_channels, cond_dim=0, dropout=0.1)
            self.spore_topo = SPORESharedPrivateDecomposer(out_channels, cond_dim=4, dropout=0.1)
            self.spore_topo_gate_L = nn.Sequential(nn.Linear(out_channels * 2 + 4, out_channels), nn.ReLU(), nn.Linear(out_channels, 1))
            self.spore_topo_gate_G = nn.Sequential(nn.Linear(out_channels * 2 + 4, out_channels), nn.ReLU(), nn.Linear(out_channels, 1))
            self.spore_fusion = SPOREFusion(dim=out_channels, rank=4, lambda_max=0.2, dropout=0.1)
            self.spore_nf_fusion = SPORENFFusion(dim=out_channels, rank=4, dropout=0.1)
            self.spore_uot_fusion = SPOREUOTFusion(dim=out_channels, hidden=out_channels, sinkhorn_iter=6, dropout=0.1)
            self.spore_epid_fusion = SPOREEPIDFusion(dim=out_channels, hidden=out_channels, dropout=0.1)
            self.spore_curo_fusion = SPORECUROFusion(dim=out_channels, hidden=out_channels, num_experts=4, dropout=0.1)
            self.spore_mcg_fusion = SPOREMCGFusion(dim=out_channels, hidden=out_channels, dropout=0.1)
            self.spore_cib_fusion = SPORECIBFusion(dim=out_channels, hidden=out_channels, dropout=0.1)
            self.spore_cib_fusion.variant = self.variant
            self.spore_hcsp_fusion = SPOREHCSPFusion(dim=out_channels, hidden=out_channels, num_subset=3, dropout=0.1)
        else:
            self.spore_expr = None
            self.spore_topo = None
            self.spore_topo_gate_L = None
            self.spore_topo_gate_G = None
            self.spore_fusion = None
            self.spore_nf_fusion = None
            self.spore_uot_fusion = None
            self.spore_epid_fusion = None
            self.spore_curo_fusion = None
            self.spore_mcg_fusion = None
            self.spore_cib_fusion = None
            self.spore_hcsp_fusion = None


    def undirected(self,edge_index):
        """Helper to ensure edges are bidirectional."""
        reversed_edge = edge_index.flip(0)
        edge_index_undirected = torch.cat([edge_index, reversed_edge], dim=1)

        edge_index_undirected = torch.unique(edge_index_undirected, dim=1)
        return edge_index_undirected


    def auxiliary_loss(self):
        if hasattr(self.gcn, 'auxiliary_loss'):
            return self.gcn.auxiliary_loss()
        return next(self.parameters()).new_tensor(0.0)


    def forward(self, data,grn_data, knn_graph, compute_contrastive=False):
        # --- View 1: KNN Graph (Feature Similarity) ---
        emb_2 = self.Mgcn(x=knn_graph.x, edge_index=knn_graph.edge_index)
        directed_edge_index = getattr(grn_data, 'edge_index_directed', grn_data.edge_index)
        graph_edge_index = self.undirected(directed_edge_index)

        # --- View 3: Global GRN (Explicit Structure) ---
        emb_grn = self.gcn(x=grn_data.x, edge_index=graph_edge_index, edge_attr=getattr(grn_data, 'edge_attr', None)) # 32
        emb_grn = self.lin4(emb_grn)
        emb_grn = emb_grn[data.target_nodes]
        emb_2 = emb_2[data.target_nodes]
        # --- View 1: Local Subgraph (Topological Context) ---
        emb_1= self.gclencoder(data)

        # Aggregate TF and Target representations (Summation)
        N = emb_2.size(0)
        index_1 = torch.range(0, N-1, 2).to(torch.long)
        index_2 = torch.range(1, N, 2).to(torch.long)
        emb_2 = emb_2[index_1] + emb_2[index_2]
        emb_grn = emb_grn[index_1] + emb_grn[index_2]

        if self.use_directed_role:
            target_nodes = data.target_nodes
            eo, ei, _ = self.dual_role(x=grn_data.x, edge_index=directed_edge_index)
            eop = eo[target_nodes][index_1]
            eip = ei[target_nodes][index_2]
            e_tf, e_target = self.dual_role.project_tf_target(eop, eip)
            score = self.dual_role.directed_score(eop, eip)
            direction_gate = torch.sigmoid(score).unsqueeze(-1)
            directed_pair = direction_gate * e_tf + (1.0 - direction_gate) * e_target
            residual_gate = torch.sigmoid(self.directed_residual_gate(torch.cat([emb_grn, directed_pair], dim=-1)))
            emb_grn = (1.0 - residual_gate) * emb_grn + residual_gate * directed_pair

        emb_id = None
        if self.use_identity_path:
            id_nodes = self.identity_encoder(grn_data.x)[data.target_nodes]
            emb_id = id_nodes[index_1] + id_nodes[index_2]

        # --- Attention-based Feature Fusion ---
        # Calculate attention scores (alpha) for each view
        att_1 = self.query(F.tanh(self.att_1(emb_1)))
        att_2 = self.query(F.tanh(self.att_2(emb_2)))
        att_grn = self.query(F.tanh(self.att_grn(emb_grn)))
        att_parts = [att_1, att_2, att_grn]
        if self.use_identity_path:
            att_parts.append(self.query(F.tanh(self.att_id(emb_id))))
        alpha = F.softmax(torch.cat(att_parts, dim=-1), dim=-1)
        alpha_t = alpha[:, 0:1]
        alpha_f = alpha[:, 1:2]
        alpha_g = alpha[:, 2:3]
        if self.use_identity_path:
            alpha_id = alpha[:, 3:4]
            x = torch.cat([alpha_t * emb_1, alpha_f * emb_2, alpha_g * emb_grn, alpha_id * emb_id], dim=-1)
        else:
            x = torch.cat([alpha_t * emb_1, alpha_f * emb_2, alpha_g * emb_grn], dim=-1)

        # --- Outputs ---
        if self.evidential_enabled:
            alpha_1 = F.softplus(self.evi_1(emb_1)) + 1.0
            alpha_2 = F.softplus(self.evi_2(emb_2)) + 1.0
            alpha_g = F.softplus(self.evi_g(emb_grn)) + 1.0
            evidence_parts = [alpha_1, alpha_2, alpha_g]
            last_alphas = [alpha_g, alpha_1, alpha_2]
            if self.use_identity_path:
                alpha_id_evi = F.softplus(self.evi_id(emb_id)) + 1.0
                evidence_parts.append(alpha_id_evi)
                last_alphas.append(alpha_id_evi)
            alpha_f = self._combine_evidence_list(evidence_parts)
            if self.spore_enabled and self.use_identity_path:
                spore_cond = self._spore_degree_features(data, directed_edge_index, index_1, index_2)
                expr_pack = self.spore_expr(emb_id, emb_2, mode='attention')
                topo_pack = self.spore_topo(emb_1, emb_grn, cond=spore_cond, mode='sum')
                topo_cond_eff = spore_cond
                if self.variant == 'spore_cib_wo_topodeg':
                    # Large-module ablation: remove topology shared/private decomposition
                    # and degree-conditioned low-degree routing together.
                    topo_cond_eff = spore_cond.new_zeros(spore_cond.shape)
                    topo_pack = dict(topo_pack)
                    topo_shared = 0.5 * (topo_pack['shared'] + topo_pack['u_a'] + topo_pack['u_b'])
                    topo_pack['shared'] = topo_shared
                    topo_pack['u_a'] = topo_shared.new_zeros(topo_shared.shape)
                    topo_pack['u_b'] = topo_shared.new_zeros(topo_shared.shape)
                low_ratio = self._spore_low_degree_ratio(topo_cond_eff)
                gate_L = torch.sigmoid(self.spore_topo_gate_L(torch.cat([topo_pack['shared'], topo_pack['u_a'], topo_cond_eff], dim=-1)))
                gate_G = torch.sigmoid(self.spore_topo_gate_G(torch.cat([topo_pack['shared'], topo_pack['u_b'], topo_cond_eff], dim=-1)))
                z_T = topo_pack['shared'] + gate_L * topo_pack['u_a'] + gate_G * topo_pack['u_b']
                alpha_E = self._combine_two_evidence(alpha_id_evi, alpha_2)
                alpha_T = self._combine_two_evidence(alpha_1, alpha_g)
                alpha_ds = self._combine_two_evidence(alpha_E, alpha_T)
                spore_out = self.spore_fusion(
                    expr_pack['out'],
                    z_T,
                    [expr_pack['u_a'], expr_pack['u_b'], topo_pack['u_a'], topo_pack['u_b']],
                    alpha_ds,
                )
                alpha_f = spore_out['alpha_final']
                self.last_spore = {
                    'expr': expr_pack,
                    'topo': topo_pack,
                    'fusion': spore_out,
                    'gate_L': gate_L,
                    'gate_G': gate_G,
                    'low_ratio': low_ratio,
                    'alpha_E': alpha_E,
                    'alpha_T': alpha_T,
                    'alpha_ds': alpha_ds,
                }
            self.last_alphas = last_alphas + [alpha_f]
            output_1 = self._alpha_to_logit(alpha_1)
            output_2 = self._alpha_to_logit(alpha_2)
            output_g = self._alpha_to_logit(alpha_g)
            output_3 = self._alpha_to_logit(alpha_f)
        else:
            # Individual view outputs (likely for auxiliary loss)
            output_1 = self.lin1(emb_1)
            output_2 = self.lin2(emb_2)
            output_g = self.lin_g(emb_grn)
            if self.use_identity_path:
                self.output_id = self.lin_id(emb_id)

            # Final fused prediction
            if self.spore_nf_enabled and self.use_identity_path:
                spore_cond = self._spore_degree_features(data, directed_edge_index, index_1, index_2)
                pair_nodes = data.target_nodes.view(-1, 2).to(spore_cond.device)
                expr_pack = self.spore_expr(emb_id, emb_2, mode='attention')
                topo_pack = self.spore_topo(emb_1, emb_grn, cond=spore_cond, mode='sum')
                topo_cond_eff = spore_cond
                if self.variant == 'spore_cib_wo_topodeg':
                    # Large-module ablation: remove topology shared/private decomposition
                    # and degree-conditioned low-degree routing together.
                    topo_cond_eff = spore_cond.new_zeros(spore_cond.shape)
                    topo_pack = dict(topo_pack)
                    topo_shared = 0.5 * (topo_pack['shared'] + topo_pack['u_a'] + topo_pack['u_b'])
                    topo_pack['shared'] = topo_shared
                    topo_pack['u_a'] = topo_shared.new_zeros(topo_shared.shape)
                    topo_pack['u_b'] = topo_shared.new_zeros(topo_shared.shape)
                low_ratio = self._spore_low_degree_ratio(topo_cond_eff)
                gate_L = torch.sigmoid(self.spore_topo_gate_L(torch.cat([topo_pack['shared'], topo_pack['u_a'], topo_cond_eff], dim=-1)))
                gate_G = torch.sigmoid(self.spore_topo_gate_G(torch.cat([topo_pack['shared'], topo_pack['u_b'], topo_cond_eff], dim=-1)))
                z_T = topo_pack['shared'] + gate_L * topo_pack['u_a'] + gate_G * topo_pack['u_b']
                private_tokens = [expr_pack['u_a'], expr_pack['u_b'], topo_pack['u_a'], topo_pack['u_b']]
                if self.variant == 'spore_uot':
                    spore_out = self.spore_uot_fusion(expr_pack['out'], z_T, spore_cond, pair_nodes)
                elif self.variant == 'spore_epid':
                    spore_out = self.spore_epid_fusion(expr_pack['out'], z_T, spore_cond, pair_nodes)
                elif self.variant in ('spore_curo', 'spore_cur'):
                    spore_out = self.spore_curo_fusion(expr_pack['out'], z_T, spore_cond, pair_nodes)
                elif self.variant == 'spore_mcg':
                    spore_out = self.spore_mcg_fusion(expr_pack['out'], z_T, spore_cond, pair_nodes)
                elif self.variant in (
                    'spore_cib', 'spore_cib_safecap', 'spore_cib_rankguard',
                    'spore_cib_stablemask', 'spore_cib_scondrank',
                    'spore_cib_puauc', 'spore_cib_edgeprompt',
                    'spore_cib_confdec', 'spore_cib_topopure',
                    'spore_cib_dualcontrol', 'spore_cib_gradalign',
                    'spore_cib_stabledecomp', 'spore_cib_crsb',
                    'spore_cib_uti', 'spore_cib_ugrt', 'spore_cib_dehg',
                    'spore_cib_per', 'spore_cib_wo_edecomp', 'spore_cib_wo_tdecomp',
                    'spore_cib_wo_degreecond', 'spore_cib_wo_topodeg', 'spore_cib_wo_ib'
                ):
                    spore_out = self.spore_cib_fusion(expr_pack, topo_pack, z_T, topo_cond_eff, pair_nodes)
                elif self.variant == 'spore_hcsp':
                    spore_out = self.spore_hcsp_fusion(expr_pack['out'], z_T, spore_cond, pair_nodes)
                else:
                    spore_out = self.spore_nf_fusion(
                        expr_pack['out'],
                        z_T,
                        private_tokens,
                    )
                output_3 = spore_out['logit']
                self.last_spore = {
                    'expr': expr_pack,
                    'topo': topo_pack,
                    'fusion': spore_out,
                    'gate_L': gate_L,
                    'gate_G': gate_G,
                    'low_ratio': low_ratio,
                    'degree_cond': topo_cond_eff,
                    'pair_nodes': pair_nodes,
                }
            else:
                output_3 = self.fuse_mlp(x)

        return output_g,output_1, output_2, output_3

    def _spore_degree_features(self, data, directed_edge_index, index_1=None, index_2=None):
        pair_nodes = data.target_nodes.view(-1, 2).to(directed_edge_index.device)
        num_nodes = grn_num_nodes = int(data.target_nodes.max().item()) + 1
        if hasattr(data, 'sub_nodes') and data.sub_nodes.numel() > 0:
            grn_num_nodes = max(grn_num_nodes, int(data.sub_nodes.max().item()) + 1)
        if directed_edge_index.numel() > 0:
            grn_num_nodes = max(grn_num_nodes, int(directed_edge_index.max().item()) + 1)
        out_deg = degree(directed_edge_index[0], num_nodes=grn_num_nodes, dtype=torch.float)
        in_deg = degree(directed_edge_index[1], num_nodes=grn_num_nodes, dtype=torch.float)
        tf = pair_nodes[:, 0].long()
        tg = pair_nodes[:, 1].long()
        log_out = torch.log1p(out_deg[tf]).view(-1, 1)
        log_in = torch.log1p(in_deg[tg]).view(-1, 1)
        if hasattr(data, 'batch'):
            sub_nodes = torch.bincount(data.batch, minlength=pair_nodes.size(0)).to(log_out.device).float().view(-1, 1)
        else:
            sub_nodes = torch.ones_like(log_out)
        log_sub_nodes = torch.log1p(sub_nodes)
        # Directed two-hop support: TF out-neighbors intersect Target in-neighbors.
        if grn_num_nodes <= 2000 and directed_edge_index.numel() > 0:
            adj = torch.zeros((grn_num_nodes, grn_num_nodes), device=directed_edge_index.device, dtype=torch.float)
            adj[directed_edge_index[0].long(), directed_edge_index[1].long()] = 1.0
            common = (adj[tf] * adj[:, tg].t()).sum(dim=-1, keepdim=True)
        else:
            common = torch.zeros_like(log_out)
        log_common = torch.log1p(common)
        return torch.cat([log_out, log_in, log_sub_nodes, log_common], dim=-1)

    def _spore_low_degree_ratio(self, degree_cond, tau=2.0):
        out_d = torch.expm1(degree_cond[:, 0:1]).clamp(min=0.0)
        in_d = torch.expm1(degree_cond[:, 1:2]).clamp(min=0.0)
        return 0.5 * (tau / (out_d + tau) + tau / (in_d + tau))

    def _alpha_to_logit(self, alpha):
        return torch.log(alpha[:, 1:2] / (alpha[:, 0:1] + 1e-8) + 1e-8)

    def _alpha_to_belief_uncertainty(self, alpha):
        evidence = alpha - 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        belief = evidence / (S + 1e-8)
        uncertainty = 2.0 / (S + 1e-8)
        return belief, uncertainty

    def _combine_two_evidence(self, alpha_a, alpha_b):
        b_a, u_a = self._alpha_to_belief_uncertainty(alpha_a)
        b_b, u_b = self._alpha_to_belief_uncertainty(alpha_b)
        conflict = b_a[:, 0:1] * b_b[:, 1:2] + b_a[:, 1:2] * b_b[:, 0:1]
        denom = (1.0 - conflict).clamp(min=1e-6)
        b = (b_a * b_b + b_a * u_b + b_b * u_a) / denom
        u = (u_a * u_b) / denom
        S = 2.0 / (u + 1e-8)
        evidence = b * S
        return evidence + 1.0

    def _combine_evidence_list(self, alphas):
        out = alphas[0]
        for alpha in alphas[1:]:
            out = self._combine_two_evidence(out, alpha)
        return out

    def _kl_dirichlet_to_uniform(self, alpha):
        beta = torch.ones_like(alpha)
        S_alpha = alpha.sum(dim=1, keepdim=True)
        S_beta = beta.sum(dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        lnB_uni = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = ((alpha - beta) * (dg1 - dg0)).sum(dim=1, keepdim=True) + lnB + lnB_uni
        return kl.squeeze(1)

    def _evidential_ce_loss(self, alpha, y, epoch=None):
        y = y.view(-1).long()
        y_onehot = F.one_hot(y, num_classes=2).to(alpha.dtype).to(alpha.device)
        S = alpha.sum(dim=1, keepdim=True)
        ce = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
        anneal = 1.0 if epoch is None else min(1.0, float(epoch) / 10.0)
        return (ce + 0.001 * anneal * self._kl_dirichlet_to_uniform(alpha_tilde)).mean()

    def compute_evidential_loss(self, y, epoch=None):
        return sum(self._evidential_ce_loss(alpha, y, epoch=epoch) for alpha in self.last_alphas)

    def _spore_supcon_loss(self, features, y, temperature=0.2):
        y = y.view(-1).long()
        if features.size(0) < 3 or y.unique().numel() < 2:
            return features.new_tensor(0.0)
        z = F.normalize(features, dim=-1)
        logits = (z @ z.t()) / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
        pos_mask = (y[:, None] == y[None, :]) & (~self_mask)
        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return features.new_tensor(0.0)
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        return loss[valid].mean()

    def compute_spore_loss(
        self,
        y,
        lambda_rec=0.05,
        lambda_orth=0.01,
        lambda_share=0.01,
        lambda_syn=0.02,
        lambda_degree=0.005,
    ):
        if not self.spore_enabled or not hasattr(self, 'last_spore'):
            return next(self.parameters()).new_tensor(0.0)
        expr = self.last_spore['expr']
        topo = self.last_spore['topo']
        fusion = self.last_spore['fusion']
        rec = expr['rec_loss'] + topo['rec_loss']
        orth = expr['orth_loss'] + topo['orth_loss']
        share = expr['share_loss'] + topo['share_loss']
        syn_feat = fusion['sET'] if 'sET' in fusion else fusion.get('z', expr['out'])
        syn = self._spore_supcon_loss(syn_feat, y)
        degree_loss = F.l1_loss(self.last_spore['gate_G'], self.last_spore['low_ratio'].detach())
        return (
            lambda_rec * rec
            + lambda_orth * orth
            + lambda_share * share
            + lambda_syn * syn
            + lambda_degree * degree_loss
        )

    def compute_stablemask_loss(self, first_spore, second_spore, logits_a, logits_b, y, args=None):
        """Stochastic decomposition consistency for StableMask-CIB.

        The same candidate edges are forwarded twice with independent dropout
        and VIB noise. Shared components and shared/private masks are encouraged
        to be stable, while private-token variance/covariance terms prevent the
        decomposition from collapsing into a constant mask or zero private space.
        """
        lam = float(getattr(args, 'cib_stable_lambda', 0.05)) if args is not None else 0.05
        lam_ema = float(getattr(args, 'cib_stable_pred_lambda', 0.02)) if args is not None else 0.02
        gamma = float(getattr(args, 'cib_stable_private_gamma', 0.50)) if args is not None else 0.50
        if lam <= 0:
            return logits_a.new_tensor(0.0)

        f1 = first_spore.get('fusion', {})
        f2 = second_spore.get('fusion', {})
        e1, e2 = first_spore.get('expr', {}), second_spore.get('expr', {})
        t1, t2 = first_spore.get('topo', {}), second_spore.get('topo', {})

        def sym_mse(a, b):
            return 0.5 * (F.mse_loss(a, b.detach()) + F.mse_loss(b, a.detach()))

        loss = logits_a.new_tensor(0.0)
        if all(k in f1 and k in f2 for k in ('cE', 'cT')):
            loss = loss + sym_mse(f1['cE'], f2['cE']) + sym_mse(f1['cT'], f2['cT'])
        if 'mu_tokens' in f1 and 'mu_tokens' in f2:
            for mu1, mu2 in zip(f1['mu_tokens'], f2['mu_tokens']):
                loss = loss + sym_mse(mu1, mu2)
        if 'mask' in e1 and 'mask' in e2:
            loss = loss + sym_mse(e1['mask'], e2['mask'])
        if 'mask' in t1 and 'mask' in t2:
            loss = loss + sym_mse(t1['mask'], t2['mask'])

        p1 = torch.sigmoid(logits_a.view(-1)).clamp(min=1e-6, max=1.0 - 1e-6)
        p2 = torch.sigmoid(logits_b.view(-1)).clamp(min=1e-6, max=1.0 - 1e-6)
        m = (0.5 * (p1 + p2)).clamp(min=1e-6, max=1.0 - 1e-6)
        kl1 = p1 * torch.log(p1 / m) + (1.0 - p1) * torch.log((1.0 - p1) / (1.0 - m))
        kl2 = p2 * torch.log(p2 / m) + (1.0 - p2) * torch.log((1.0 - p2) / (1.0 - m))
        pred_cons = 0.5 * (kl1 + kl2)
        y = y.to(torch.float).view(-1)
        hard_neg = ((p1.detach() > torch.quantile(p1.detach(), 0.90)) & (y <= 0.5)).float()
        conflict = torch.zeros_like(p1)
        if 'expr_logit' in f1 and 'topo_logit' in f1:
            conflict = torch.abs(torch.sigmoid(f1['expr_logit'].detach().view(-1)) - torch.sigmoid(f1['topo_logit'].detach().view(-1)))
        cons_w = 1.0 + 1.0 * y + 0.5 * hard_neg + 0.5 * conflict
        loss = loss + lam_ema * (pred_cons * cons_w).mean()

        priv = f1.get('private_tokens', [])
        if priv:
            var_loss = logits_a.new_tensor(0.0)
            for token in priv:
                if token.size(0) > 1:
                    std = token.std(dim=0)
                    var_loss = var_loss + torch.relu(gamma - std).pow(2).mean()
            loss = loss + 0.10 * var_loss / max(len(priv), 1)
        if len(priv) >= 4 and priv[0].size(0) > 1:
            cov_loss = logits_a.new_tensor(0.0)
            for a, b in ((priv[0], priv[1]), (priv[2], priv[3])):
                aa = a - a.mean(dim=0, keepdim=True)
                bb = b - b.mean(dim=0, keepdim=True)
                cov = aa.t().matmul(bb) / max(a.size(0) - 1, 1)
                cov_loss = cov_loss + cov.pow(2).mean()
            loss = loss + 0.10 * cov_loss
        return lam * loss

    def _cib_binary_entropy(self, p):
        p = p.clamp(min=1e-8, max=1.0 - 1e-8)
        return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / math.log(2.0)

    def _cib_difficulty_score(self, y, args=None):
        """DualControl-CIB candidate-edge difficulty r_ij.

        Features follow the provided design:
        conflict C, uncertainty U, low-degree D and rank-violation V.
        The features are detached; only the scalar controller weights are
        learnable, preventing the predictor from manufacturing uncertainty.
        """
        fusion = self.last_spore.get('fusion', {})
        scores = fusion.get('logit', None)
        if scores is None:
            return None, {}
        scores = scores.view(-1)
        y = y.to(torch.float).view(-1)
        pE = torch.sigmoid(fusion.get('expr_logit', fusion['logit']).view(-1)).detach()
        pT = torch.sigmoid(fusion.get('topo_logit', fusion['logit']).view(-1)).detach()
        conflict = torch.abs(pE - pT)
        uncertainty = 0.5 * (self._cib_binary_entropy(pE) + self._cib_binary_entropy(pT))

        degree_cond = self.last_spore.get('degree_cond', None)
        if degree_cond is not None:
            deg_sum = degree_cond[:, 0] + degree_cond[:, 1]
            low_degree = (1.0 - deg_sum / deg_sum.max().clamp(min=1.0)).clamp(0.0, 1.0).detach()
        else:
            low_degree = torch.zeros_like(conflict)

        margin = float(getattr(args, 'cib_rank_margin', 0.10)) if args is not None else 0.10
        violation = torch.zeros_like(scores)
        pos_idx = torch.nonzero(y > 0.5, as_tuple=False).view(-1)
        neg_idx = torch.nonzero(y <= 0.5, as_tuple=False).view(-1)
        pair_nodes = self.last_spore.get('pair_nodes', None)
        if pos_idx.numel() > 0 and neg_idx.numel() > 0:
            if pair_nodes is not None:
                tf_nodes = pair_nodes[:, 0]
                for pi in pos_idx:
                    same_tf_neg = neg_idx[tf_nodes[neg_idx] == tf_nodes[pi]]
                    pool = same_tf_neg if same_tf_neg.numel() > 0 else neg_idx
                    max_neg = scores[pool].detach().max()
                    violation[pi] = torch.relu(margin + max_neg - scores[pi].detach())
            else:
                max_neg = scores[neg_idx].detach().max()
                violation[pos_idx] = torch.relu(margin + max_neg - scores[pos_idx].detach())
        violation = violation.clamp(min=0.0, max=5.0)

        feats = torch.stack([conflict, uncertainty, low_degree, violation], dim=-1).detach()
        w = F.softplus(fusion.get('dual_w', self.spore_cib_fusion.dual_w))
        b = fusion.get('dual_b', self.spore_cib_fusion.dual_b)
        r = torch.sigmoid((feats * w.view(1, -1)).sum(dim=-1) + b)
        self.last_spore['dual_r'] = r
        self.last_spore['dual_features'] = feats
        return r, {
            'conflict': conflict,
            'uncertainty': uncertainty,
            'low_degree': low_degree,
            'violation': violation,
        }

    def compute_cib_dual_rank_loss(self, y, args=None, r=None):
        """Degree-matched difficulty-weighted ranking loss for DualControl-CIB."""
        fusion = self.last_spore.get('fusion', {})
        if 'logit' not in fusion:
            return next(self.parameters()).new_tensor(0.0)
        scores = fusion['logit'].view(-1)
        y = y.to(torch.float).view(-1)
        pos_mask = y > 0.5
        neg_mask = ~pos_mask
        if not (pos_mask.any() and neg_mask.any()):
            return scores.new_tensor(0.0)
        margin = float(getattr(args, 'cib_rank_margin', 0.10)) if args is not None else 0.10
        tau = float(getattr(args, 'cib_rank_tau', 0.50)) if args is not None else 0.50
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        if r is None:
            r, _ = self._cib_difficulty_score(y, args=args)
        pos_r = r[pos_mask] if r is not None else torch.ones_like(pos_scores)

        # 40% high-score hard negatives, 40% degree-matched hard negatives,
        # 20% medium negatives, implemented as soft mixture weights.
        n_neg = neg_scores.numel()
        rank_order = torch.argsort(neg_scores.detach(), descending=True)
        hard_mask = torch.zeros_like(neg_scores)
        med_mask = torch.zeros_like(neg_scores)
        k_hard = max(1, int(math.ceil(0.40 * float(n_neg))))
        hard_mask[rank_order[:k_hard]] = 1.0
        lo = int(math.floor(0.40 * float(n_neg)))
        hi = max(lo + 1, int(math.ceil(0.70 * float(n_neg))))
        med_mask[rank_order[lo:hi]] = 1.0

        degree_cond = self.last_spore.get('degree_cond', None)
        if degree_cond is not None:
            in_d = torch.expm1(degree_cond[:, 1]).clamp(min=0.0)
            pos_in = in_d[pos_mask]
            neg_in = in_d[neg_mask]
            scale = neg_in.std().detach().clamp(min=1.0)
            degree_match = torch.exp(-torch.abs(neg_in[:, None] - pos_in[None, :]) / scale)
        else:
            degree_match = torch.ones(neg_scores.numel(), pos_scores.numel(), device=scores.device)
        base_neg_w = (0.40 * hard_mask[:, None] + 0.20 * med_mask[:, None] + 0.40 * degree_match).clamp(min=0.05)
        pair = F.softplus((neg_scores[:, None] - pos_scores[None, :] + margin) / tau)
        return (pair * base_neg_w * pos_r.detach()[None, :].clamp(min=0.02)).mean()

    def compute_cib_rank_raw_loss(self, y, args=None):
        """Raw RankGuard-style loss used by GradAlign-RankGuard-CIB."""
        fusion = self.last_spore.get('fusion', {})
        if 'logit' not in fusion:
            return next(self.parameters()).new_tensor(0.0)
        scores = fusion['logit'].view(-1)
        y = y.to(torch.float).view(-1)
        pos_mask = y > 0.5
        neg_mask = ~pos_mask
        if not (pos_mask.any() and neg_mask.any()):
            return scores.new_tensor(0.0)
        margin = float(getattr(args, 'cib_rank_margin', 0.10)) if args is not None else 0.10
        tau = float(getattr(args, 'cib_rank_tau', 0.50)) if args is not None else 0.50
        eta = float(getattr(args, 'cib_rank_eta', 0.50)) if args is not None else 0.50
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        k = min(neg_scores.numel(), max(1, int(math.ceil(0.35 * float(neg_scores.numel())))))
        hard_neg = neg_scores.topk(k=k, largest=True).values
        rank_loss = F.softplus((hard_neg[:, None] - pos_scores[None, :] + margin) / tau).mean()

        pair_nodes = self.last_spore.get('pair_nodes', None)
        if pair_nodes is not None:
            tf_nodes = pair_nodes[:, 0]
            list_losses, weights = [], []
            for tf_id in torch.unique(tf_nodes):
                m = tf_nodes == tf_id
                yy = y[m]
                if yy.numel() < 2 or yy.max() <= 0.5 or yy.min() > 0.5:
                    continue
                ss = scores[m]
                denom = torch.logsumexp(ss / tau, dim=0)
                list_losses.append((-(ss[yy > 0.5] / tau - denom)).mean())
                n_pos = (yy > 0.5).float().sum()
                n_neg = (yy <= 0.5).float().sum()
                weights.append((torch.log1p(n_neg) / torch.log1p(n_pos).clamp(min=1.0)).clamp(1.0, 4.0))
            if list_losses:
                w = torch.stack(weights)
                w = w / w.mean().clamp(min=1e-6)
                rank_loss = rank_loss + eta * (torch.stack(list_losses) * w).mean()
        return rank_loss

    def gradalign_parameters(self):
        """Shared CIB/SPORE parameters coordinated by GradAlign."""
        modules = [self.spore_expr, self.spore_topo, self.spore_topo_gate_L, self.spore_topo_gate_G, self.spore_cib_fusion]
        params = []
        for module in modules:
            if module is None:
                continue
            params.extend([p for p in module.parameters() if p.requires_grad])
        # Keep order stable while removing duplicates.
        seen, uniq = set(), []
        for p in params:
            if id(p) not in seen:
                uniq.append(p)
                seen.add(id(p))
        return uniq

    def compute_variant_loss(self, y, args=None):
        if not self.spore_nf_enabled or not hasattr(self, 'last_spore'):
            return next(self.parameters()).new_tensor(0.0)
        fusion = self.last_spore.get('fusion', {})
        y = y.to(torch.float).view(-1)
        device_loss = y.new_tensor(0.0)

        if self.variant == 'spore_uot':
            lam_list = float(getattr(args, 'uot_lambda_list', 0.02)) if args is not None else 0.02
            lam_aff = float(getattr(args, 'uot_lambda_affinity', 0.002)) if args is not None else 0.002
            score = fusion.get('transport_logit', None)
            tf_inv = fusion.get('tf_inv', None)
            if score is None or tf_inv is None:
                return device_loss
            score = score.view(-1)
            list_losses = []
            for tf_id in torch.unique(tf_inv):
                mask = tf_inv == tf_id
                yy = y[mask]
                ss = score[mask]
                if yy.numel() < 2 or yy.max() <= 0 or yy.min() >= 1:
                    continue
                pos = ss[yy > 0.5]
                neg = ss[yy <= 0.5]
                list_losses.append(F.softplus(neg[:, None] - pos[None, :] + 0.10).mean())
            list_loss = torch.stack(list_losses).mean() if list_losses else device_loss
            aff = fusion.get('raw_affinity', score).view(-1)
            return lam_list * list_loss + lam_aff * aff.pow(2).mean()

        if self.variant == 'spore_epid':
            lam_atom = float(getattr(args, 'epid_lambda_atom', 0.02)) if args is not None else 0.02
            lam_entropy = float(getattr(args, 'epid_lambda_entropy', 0.002)) if args is not None else 0.002
            lam_div = float(getattr(args, 'epid_lambda_div', 0.002)) if args is not None else 0.002
            atom_logits = fusion.get('atom_logits', None)
            atom_weights = fusion.get('atom_weights', None)
            atoms = fusion.get('atoms', None)
            loss = device_loss
            if atom_logits is not None:
                loss = loss + lam_atom * F.binary_cross_entropy_with_logits(
                    atom_logits, y[:, None].expand_as(atom_logits)
                )
            if atom_weights is not None:
                entropy = -(atom_weights * torch.log(atom_weights.clamp(min=1e-8))).sum(dim=-1).mean()
                loss = loss - lam_entropy * entropy
            if atoms is not None:
                z = [F.normalize(a, dim=-1) for a in atoms]
                div = device_loss
                cnt = 0
                for i in range(len(z)):
                    for j in range(i + 1, len(z)):
                        div = div + F.cosine_similarity(z[i], z[j], dim=-1).pow(2).mean()
                        cnt += 1
                loss = loss + lam_div * div / max(cnt, 1)
            return loss

        if self.variant in ('spore_curo', 'spore_cur'):
            lam_expert = float(getattr(args, 'curo_lambda_expert', 0.01)) if args is not None else 0.01
            lam_balance = float(getattr(args, 'curo_lambda_balance', 0.003)) if args is not None else 0.003
            lam_margin = float(getattr(args, 'curo_lambda_margin', 0.005)) if args is not None else 0.005
            expert_logits = fusion.get('expert_logits', None)
            route = fusion.get('route', None)
            cf = fusion.get('cf_logits', None)
            loss = device_loss
            if expert_logits is not None:
                loss = loss + lam_expert * F.binary_cross_entropy_with_logits(
                    expert_logits, y[:, None].expand_as(expert_logits)
                )
            if route is not None:
                mean_route = route.mean(dim=0)
                uniform = torch.full_like(mean_route, 1.0 / mean_route.numel())
                loss = loss + lam_balance * F.mse_loss(mean_route, uniform)
            if cf is not None:
                le, lt, lj = cf[:, 0], cf[:, 1], cf[:, 2]
                loss = loss + lam_margin * F.softplus(torch.maximum(le, lt) - lj + 0.05).mean()
            return loss

        if self.variant == 'spore_mcg':
            lam_curve = float(getattr(args, 'mcg_lambda_curve', 0.001)) if args is not None else 0.001
            lam_mix = float(getattr(args, 'mcg_lambda_mix', 0.001)) if args is not None else 0.001
            curv = fusion.get('curvature', None)
            mix = fusion.get('mix', None)
            loss = device_loss
            if curv is not None:
                loss = loss + lam_curve * (torch.log1p(curv).pow(2)).mean()
            if mix is not None:
                # Keep both geometries alive during screening; final pruning can
                # be decided after the six-dataset result table.
                loss = loss + lam_mix * (mix - 0.5).pow(2).mean()
            return loss

        if self.variant in (
            'spore_cib', 'spore_cib_safecap', 'spore_cib_rankguard',
            'spore_cib_stablemask', 'spore_cib_scondrank',
            'spore_cib_puauc', 'spore_cib_edgeprompt',
            'spore_cib_confdec', 'spore_cib_topopure',
            'spore_cib_dualcontrol', 'spore_cib_gradalign',
            'spore_cib_stabledecomp', 'spore_cib_crsb',
            'spore_cib_uti', 'spore_cib_ugrt', 'spore_cib_dehg',
            'spore_cib_per', 'spore_cib_wo_edecomp', 'spore_cib_wo_tdecomp',
            'spore_cib_wo_degreecond', 'spore_cib_wo_topodeg', 'spore_cib_wo_ib'
        ):
            lam_kl = float(getattr(args, 'cib_lambda_kl', 0.0005)) if args is not None else 0.0005
            lam_gain = float(getattr(args, 'cib_lambda_gain', 0.01)) if args is not None else 0.01
            lam_entropy = float(getattr(args, 'cib_lambda_entropy', 0.001)) if args is not None else 0.001
            loss = device_loss
            dual_r = None
            if self.variant == 'spore_cib_dualcontrol' and 'ib_kl_dim' in fusion:
                dual_r, _ = self._cib_difficulty_score(y, args=args)
                freebits = float(getattr(args, 'cib_safecap_freebits', 0.05)) if args is not None else 0.05
                rho_beta = float(getattr(args, 'cib_dual_beta_rho', 0.60)) if args is not None else 0.60
                rmax = float(getattr(args, 'cib_dual_gate_max', 0.35)) if args is not None else 0.35
                lam_gate = float(getattr(args, 'cib_dual_gate_lambda', 0.01)) if args is not None else 0.01
                safe_kl = torch.relu(fusion['ib_kl_dim'] - freebits).sum(dim=-1)
                beta = (1.0 - rho_beta * dual_r).clamp(min=1.0 - rho_beta, max=1.0)
                loss = loss + lam_kl * (beta * safe_kl).mean()
                loss = loss + lam_gate * torch.relu(dual_r.mean() - rmax).pow(2)
            elif self.variant == 'spore_cib_safecap' and 'ib_kl_dim' in fusion:
                kl_dim = fusion['ib_kl_dim']
                freebits = float(getattr(args, 'cib_safecap_freebits', 0.05)) if args is not None else 0.05
                beta_min = float(getattr(args, 'cib_safecap_beta_min', 0.20)) if args is not None else 0.20
                beta_max = float(getattr(args, 'cib_safecap_beta_max', 1.00)) if args is not None else 1.00
                pE = torch.sigmoid(fusion.get('expr_logit', fusion['logit']).view(-1)).detach()
                pT = torch.sigmoid(fusion.get('topo_logit', fusion['logit']).view(-1)).detach()
                entE = -(pE * torch.log(pE.clamp(min=1e-8)) + (1.0 - pE) * torch.log((1.0 - pE).clamp(min=1e-8))) / math.log(2.0)
                entT = -(pT * torch.log(pT.clamp(min=1e-8)) + (1.0 - pT) * torch.log((1.0 - pT).clamp(min=1e-8))) / math.log(2.0)
                uncertainty = 0.5 * (entE + entT)
                conflict = torch.abs(pE - pT)
                degree_cond = self.last_spore.get('degree_cond', None)
                if degree_cond is not None:
                    deg_sum = degree_cond[:, 0] + degree_cond[:, 1]
                    low_degree = 1.0 - deg_sum / deg_sum.max().clamp(min=1.0)
                    low_degree = low_degree.clamp(min=0.0, max=1.0).detach()
                else:
                    low_degree = torch.zeros_like(uncertainty)
                r = (0.35 * uncertainty + 0.35 * conflict + 0.30 * low_degree).clamp(min=0.0, max=1.0).detach()
                beta = beta_max - (beta_max - beta_min) * r
                safe_kl = torch.relu(kl_dim - freebits).sum(dim=-1)
                loss = loss + lam_kl * (beta * safe_kl).mean()
            elif self.variant != 'spore_cib_wo_ib' and 'ib_kl' in fusion:
                loss = loss + lam_kl * fusion['ib_kl']
            if self.variant != 'spore_cib_wo_ib' and 'shared_logit' in fusion and 'logit' in fusion:
                full_bce = F.binary_cross_entropy_with_logits(fusion['logit'].view(-1), y, reduction='none')
                shared_bce = F.binary_cross_entropy_with_logits(fusion['shared_logit'].view(-1), y, reduction='none')
                # Private variables should provide conditional utility beyond
                # the shared baseline; use a soft hinge to avoid destabilizing
                # already-good shared predictions.
                loss = loss + lam_gain * F.softplus(full_bce - shared_bce).mean()
            for key in ('expr_gate', 'topo_gate'):
                gate = fusion.get(key, None)
                if gate is not None:
                    ent = -(gate * torch.log(gate.clamp(min=1e-8))).sum(dim=-1).mean()
                    loss = loss - lam_entropy * ent
            if self.variant == 'spore_cib_dualcontrol' and 'logit' in fusion:
                lam_rank = float(getattr(args, 'cib_dual_rank_lambda', 0.08)) if args is not None else 0.08
                loss = loss + lam_rank * self.compute_cib_dual_rank_loss(y, args=args, r=dual_r)
            if self.variant in ('spore_cib_rankguard', 'spore_cib_scondrank', 'spore_cib_puauc') and 'logit' in fusion:
                lam_rank = float(getattr(args, 'cib_rank_lambda', 0.10)) if args is not None else 0.10
                if self.variant == 'spore_cib_scondrank':
                    lam_rank = float(getattr(args, 'cib_scond_rank_lambda', lam_rank)) if args is not None else lam_rank
                if self.variant == 'spore_cib_puauc':
                    lam_rank = float(getattr(args, 'cib_puauc_lambda', 0.06)) if args is not None else 0.06
                margin = float(getattr(args, 'cib_rank_margin', 0.10)) if args is not None else 0.10
                tau = float(getattr(args, 'cib_rank_tau', 0.50)) if args is not None else 0.50
                eta = float(getattr(args, 'cib_rank_eta', 0.50)) if args is not None else 0.50
                alpha_low = float(getattr(args, 'cib_rank_low_alpha', 0.35)) if args is not None else 0.35
                scores = fusion['logit'].view(-1)
                pos_mask = y > 0.5
                neg_mask = ~pos_mask
                rank_loss = device_loss
                if pos_mask.any() and neg_mask.any():
                    pos_scores = scores[pos_mask]
                    neg_scores = scores[neg_mask]
                    k = min(neg_scores.numel(), max(1, int(math.ceil(0.35 * float(neg_scores.numel())))))
                    hard_neg = neg_scores.topk(k=k, largest=True).values
                    degree_cond = self.last_spore.get('degree_cond', None)
                    if degree_cond is not None:
                        out_d = torch.expm1(degree_cond[:, 0]).clamp(min=0.0)
                        in_d = torch.expm1(degree_cond[:, 1]).clamp(min=0.0)
                        low_pair = 0.5 * (2.0 / (out_d + 2.0) + 2.0 / (in_d + 2.0))
                        sub = degree_cond[:, 2]
                        common = degree_cond[:, 3]
                        low_sub = 1.0 - sub / sub.max().clamp(min=1.0)
                        low_common = 1.0 - common / common.max().clamp(min=1.0)
                        pE = torch.sigmoid(fusion.get('expr_logit', fusion['logit']).view(-1)).detach()
                        pT = torch.sigmoid(fusion.get('topo_logit', fusion['logit']).view(-1)).detach()
                        conflict = torch.abs(pE - pT)
                        trigger = (0.40 * low_pair + 0.20 * low_sub + 0.20 * low_common + 0.20 * conflict).clamp(0.0, 1.0)
                        if self.variant == 'spore_cib_scondrank':
                            low_w = trigger[pos_mask].detach().clamp(min=0.05)
                        else:
                            low_w = 1.0 + alpha_low * low_pair[pos_mask].detach()
                        if self.variant == 'spore_cib_puauc':
                            neg_support = (0.5 * (pE + pT) * (0.5 + 0.5 * common / common.max().clamp(min=1.0))).clamp(0.0, 0.95)
                            neg_rel = (1.0 - neg_support[neg_mask]).detach()
                            hard_idx = neg_scores.topk(k=k, largest=True).indices
                            hard_w = neg_rel[hard_idx].clamp(min=0.10)
                        else:
                            hard_w = torch.ones_like(hard_neg)
                    else:
                        low_w = torch.ones_like(pos_scores)
                        hard_w = torch.ones_like(hard_neg)
                    pair = F.softplus((hard_neg[:, None] - pos_scores[None, :] + margin) / tau)
                    rank_loss = rank_loss + (pair * hard_w[:, None] * low_w[None, :]).mean()
                    pair_nodes = self.last_spore.get('pair_nodes', None)
                    if pair_nodes is not None:
                        tf_nodes = pair_nodes[:, 0]
                        list_losses = []
                        for tf_id in torch.unique(tf_nodes):
                            m = tf_nodes == tf_id
                            yy = y[m]
                            if yy.numel() < 2 or yy.max() <= 0.5 or yy.min() > 0.5:
                                continue
                            ss = scores[m]
                            denom = torch.logsumexp(ss / tau, dim=0)
                            list_losses.append((-(ss[yy > 0.5] / tau - denom)).mean())
                        if list_losses:
                            rank_loss = rank_loss + eta * torch.stack(list_losses).mean()
                loss = loss + lam_rank * rank_loss
            if self.variant == 'spore_cib_confdec' and all(k in fusion for k in ('expr_logit', 'topo_logit', 'logit')):
                lam_agree = float(getattr(args, 'cib_conf_lambda_agree', 0.015)) if args is not None else 0.015
                lam_res = float(getattr(args, 'cib_conf_lambda_residual', 0.003)) if args is not None else 0.003
                pE = torch.sigmoid(fusion['expr_logit'].view(-1))
                pT = torch.sigmoid(fusion['topo_logit'].view(-1))
                conflict = torch.abs(pE.detach() - pT.detach()).clamp(0.0, 1.0)
                agree_w = (1.0 - conflict).pow(2)
                agree = (pE - pT).pow(2)
                loss = loss + lam_agree * (agree_w * agree).mean()
                if 'zE' in fusion and 'zT' in fusion:
                    cos = F.cosine_similarity(fusion['zE'], fusion['zT'], dim=-1).pow(2)
                    loss = loss + lam_res * (conflict * cos).mean()
            if self.variant == 'spore_cib_edgeprompt':
                prompt_gate = fusion.get('prompt_gate', None)
                prompt_vec = fusion.get('prompt_vec', None)
                if prompt_gate is not None and prompt_vec is not None:
                    lam_prompt = float(getattr(args, 'cib_edgeprompt_lambda', 0.001)) if args is not None else 0.001
                    loss = loss + lam_prompt * (prompt_gate.pow(2).mean() + 0.1 * prompt_vec.pow(2).mean())
            if self.variant == 'spore_cib_topopure' and 'topo_pure_logit' in fusion:
                lam_topo = float(getattr(args, 'cib_topopure_aux_lambda', 0.01)) if args is not None else 0.01
                loss = loss + lam_topo * F.binary_cross_entropy_with_logits(fusion['topo_pure_logit'].view(-1), y)
            if self.variant == 'spore_cib_crsb':
                lam_bridge = float(getattr(args, 'crsb_lambda_bridge', 0.003)) if args is not None else 0.003
                lam_risk = float(getattr(args, 'crsb_lambda_risk', 0.002)) if args is not None else 0.002
                energy = fusion.get('crsb_energy', None)
                gate = fusion.get('crsb_gate', None)
                if energy is not None:
                    loss = loss + lam_bridge * energy.mean()
                if energy is not None and gate is not None:
                    # High transport risk should not dominate the final state.
                    loss = loss + lam_risk * (gate * energy.detach()).mean()
            if self.variant == 'spore_cib_uti':
                lam_order = float(getattr(args, 'uti_lambda_order', 0.003)) if args is not None else 0.003
                lam_norm = float(getattr(args, 'uti_lambda_norm', 0.001)) if args is not None else 0.001
                order_w = fusion.get('uti_order_w', None)
                uncertainty = fusion.get('uti_uncertainty', None)
                inter = fusion.get('uti_interaction', None)
                if order_w is not None and uncertainty is not None:
                    # Higher uncertainty should prefer lower-order terms; this
                    # prevents multiplicative noise amplification on mDC-like
                    # low-support edges.
                    expected_order = order_w[:, 0] + 2.0 * order_w[:, 1] + 3.0 * order_w[:, 2]
                    target_order = (3.0 - 1.5 * uncertainty.view(-1)).clamp(min=1.0, max=3.0)
                    loss = loss + lam_order * F.mse_loss(expected_order, target_order.detach())
                if inter is not None:
                    loss = loss + lam_norm * inter.pow(2).mean()
            if self.variant == 'spore_cib_ugrt':
                lam_list = float(getattr(args, 'ugrt_lambda_list', 0.02)) if args is not None else 0.02
                lam_mass = float(getattr(args, 'ugrt_lambda_mass', 0.002)) if args is not None else 0.002
                u = fusion.get('ugrt', {})
                score = u.get('transport_logit', None)
                tf_inv = u.get('tf_inv', None)
                if score is not None and tf_inv is not None:
                    score = score.view(-1)
                    list_losses = []
                    for tf_id in torch.unique(tf_inv):
                        mask = tf_inv == tf_id
                        yy = y[mask]
                        ss = score[mask]
                        if yy.numel() < 2 or yy.max() <= 0.5 or yy.min() > 0.5:
                            continue
                        pos = ss[yy > 0.5]
                        neg = ss[yy <= 0.5]
                        list_losses.append(F.softplus(neg[:, None] - pos[None, :] + 0.10).mean())
                    if list_losses:
                        loss = loss + lam_list * torch.stack(list_losses).mean()
                mass = u.get('mass', None)
                if mass is not None:
                    loss = loss + lam_mass * torch.log1p(mass).pow(2).mean()
            if self.variant == 'spore_cib_dehg':
                lam_split = float(getattr(args, 'dehg_lambda_split', 0.002)) if args is not None else 0.002
                lam_sparse = float(getattr(args, 'dehg_lambda_sparse', 0.001)) if args is not None else 0.001
                d = fusion.get('dehg', {})
                gate = d.get('hyper_gate', None)
                residual = d.get('hyper_residual', None)
                if gate is not None:
                    ent = -(gate * torch.log(gate.clamp(min=1e-8))).sum(dim=-1).mean()
                    loss = loss + lam_split * ent
                if residual is not None:
                    loss = loss + lam_sparse * residual.pow(2).mean()
            if self.variant == 'spore_cib_per':
                lam_stable = float(getattr(args, 'per_lambda_stable', 0.001)) if args is not None else 0.001
                signed = fusion.get('per_signed_A', None)
                if signed is not None:
                    # No perturbation supervision is available in the benchmark
                    # path; this keeps the executable signed matrix bounded.
                    loss = loss + lam_stable * signed.pow(2).mean()
            return loss

        if self.variant == 'spore_hcsp':
            lam_div = float(getattr(args, 'hcsp_lambda_div', 0.002)) if args is not None else 0.002
            lam_cross = float(getattr(args, 'hcsp_lambda_crossorth', 0.003)) if args is not None else 0.003
            lam_res = float(getattr(args, 'hcsp_lambda_residual', 0.0005)) if args is not None else 0.0005
            loss = device_loss
            subset = fusion.get('subset_tokens', None)
            g = fusion.get('global_prog', None)
            r = fusion.get('residual_prog', None)
            if subset is not None:
                z = F.normalize(subset, dim=-1)
                div = device_loss
                cnt = 0
                for i in range(z.size(1)):
                    for j in range(i + 1, z.size(1)):
                        div = div + F.cosine_similarity(z[:, i], z[:, j], dim=-1).pow(2).mean()
                        cnt += 1
                loss = loss + lam_div * div / max(cnt, 1)
            if g is not None and r is not None:
                loss = loss + lam_cross * F.cosine_similarity(g, r, dim=-1).pow(2).mean()
                loss = loss + lam_res * r.pow(2).mean()
            return loss

        return device_loss
