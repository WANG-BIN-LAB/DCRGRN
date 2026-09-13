"""Dataset loading and enclosing-subgraph preprocessing."""

from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, InMemoryDataset

from .func_utils import extract_enclosing_subgraphs


class GeneExpressionDataset(InMemoryDataset):
    """Load a gene-by-cell expression matrix as a PyG node-feature dataset."""

    def __init__(self, root, expression_file, transform=None, pre_transform=None):
        self.expression_file = str(expression_file)
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ["gene_expression.pt"]

    def process(self):
        expression = np.asarray(
            pd.read_csv(self.expression_file, index_col=0, header=0)
        )
        expression = StandardScaler().fit_transform(expression.T).T
        data = Data(x=torch.tensor(expression, dtype=torch.float))
        torch.save(data, self.processed_paths[0])


def load_expression_data(expression_file, cache_dir):
    """Create or load the standardized gene-expression dataset."""

    return GeneExpressionDataset(
        root=str(cache_dir),
        expression_file=str(expression_file),
    )


class DCRGRNDataset(InMemoryDataset):
    """Create fixed-split DRNL enclosing subgraphs for link prediction."""

    SPLITS = ("train", "validation", "test")

    def __init__(
        self,
        expression_dataset,
        split_dir,
        dataset_tag,
        num_hops=2,
        split="train",
    ):
        normalized_split = str(split).lower()
        if normalized_split == "val":
            normalized_split = "validation"
        if normalized_split not in self.SPLITS:
            raise ValueError(f"Unsupported split: {split}")

        self.expression_dataset = expression_dataset
        self.base_data = expression_dataset[0]
        self.split_dir = Path(split_dir)
        self.dataset_tag = str(dataset_tag).replace(" ", "_")
        self.num_hops = int(num_hops)
        super().__init__(expression_dataset.root)

        index = self.SPLITS.index(normalized_split)
        self.data, self.slices = torch.load(
            self.processed_paths[index], weights_only=False
        )

    @property
    def processed_file_names(self):
        return [
            f"{self.dataset_tag}_{split}_subgraphs.pt" for split in self.SPLITS
        ]

    @staticmethod
    def _load_edges(file_path):
        frame = pd.read_csv(file_path, index_col=0, header=0)
        positive = frame.loc[frame["Label"] == 1, ["TF", "Target"]].to_numpy()
        negative = frame.loc[frame["Label"] == 0, ["TF", "Target"]].to_numpy()
        positive = torch.as_tensor(positive, dtype=torch.long).t().contiguous()
        negative = torch.as_tensor(negative, dtype=torch.long).t().contiguous()
        return positive, negative

    def process(self):
        files = {
            "train": self.split_dir / "Train_set.csv",
            "validation": self.split_dir / "Validation_set.csv",
            "test": self.split_dir / "Test_set.csv",
        }
        edges = {name: self._load_edges(path) for name, path in files.items()}
        training_graph = edges["train"][0]

        subgraphs = {}
        for split_name in self.SPLITS:
            positive, negative = edges[split_name]
            positive_subgraphs = extract_enclosing_subgraphs(
                self.base_data,
                self.num_hops,
                positive,
                training_graph,
                1,
            )
            negative_subgraphs = extract_enclosing_subgraphs(
                self.base_data,
                self.num_hops,
                negative,
                training_graph,
                0,
            )
            subgraphs[split_name] = positive_subgraphs + negative_subgraphs

        all_subgraphs = chain.from_iterable(subgraphs.values())
        max_label = max(int(graph.z.max()) for graph in all_subgraphs)
        for graph in chain.from_iterable(subgraphs.values()):
            graph.x = F.one_hot(graph.z, max_label + 1).to(torch.float)
            graph.z = None

        for index, split_name in enumerate(self.SPLITS):
            torch.save(self.collate(subgraphs[split_name]), self.processed_paths[index])
