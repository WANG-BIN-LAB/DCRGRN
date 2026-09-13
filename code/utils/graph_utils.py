"""Expression-similarity graph construction and Node2Vec initialization."""

import random

import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import KNNGraph


def seed_everything(seed):
    """Set deterministic random states used by preprocessing and training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def construct_knn_graph(data, k=1, device=None):
    """Construct an undirected cosine KNN graph from expression profiles."""

    if data.x is None:
        raise ValueError("Gene-expression features are required to build the KNN graph.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = data.clone()
    graph.pos = graph.x
    transform = KNNGraph(
        int(k),
        loop=False,
        force_undirected=True,
        cosine=True,
    )
    graph = transform(graph.to(device)).cpu()
    graph.pos = None
    graph.x = None
    return graph


def train_node2vec_embeddings(graph, seed=None):
    """Learn 32-dimensional Node2Vec features on the expression KNN graph."""

    if seed is not None:
        seed_everything(int(seed))
    model = Node2Vec(
        graph.edge_index,
        embedding_dim=32,
        walk_length=10,
        context_size=5,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=False,
        num_nodes=graph.num_nodes,
    )
    loader = model.loader(batch_size=128, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float("inf")
    patience = 0

    for _ in range(1, 200):
        model.train()
        total_loss = 0.0
        for positive_walks, negative_walks in loader:
            optimizer.zero_grad()
            loss = model.loss(positive_walks, negative_walks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / max(len(loader), 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
        if patience >= 20:
            break
    return model().detach()
