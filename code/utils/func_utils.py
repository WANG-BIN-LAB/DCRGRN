"""Structural labeling and enclosing-subgraph extraction."""

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from tqdm import tqdm


def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    """
    Double-Radius Node Labeling (DRNL).
    Encodes the structural role of each node relative to the target link (src, dst)
    by hashing their distances into a unique integer label.
    """
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(
        adj_wo_src, directed=False, unweighted=True, indices=dst - 1
    )
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.0
    z[dst] = 1.0
    z[torch.isnan(z)] = 0.0
    return z.to(torch.long)


def extract_enclosing_subgraphs(data, num_hops, link_index, edge_index, y):
    """
    Extracts enclosing k-hop subgraphs for a given list of links (edges).
    Used to transform link prediction into a graph classification task.
    """
    data_list = []
    for src, dst in tqdm(link_index.t().tolist(), desc="Extracting subgraphs"):
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst],
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )
        target_nodes = torch.tensor([src, dst])
        src, dst = mapping.tolist()

        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        z = drnl_node_labeling(
            sub_edge_index, src, dst, num_nodes=sub_nodes.size(0)
        )

        sub_data = Data(
            x=data.x[sub_nodes],
            z=z,
            edge_index=sub_edge_index,
            y=y,
            sub_nodes=sub_nodes,
            target_nodes=target_nodes,
        )
        data_list.append(sub_data)
    return data_list
