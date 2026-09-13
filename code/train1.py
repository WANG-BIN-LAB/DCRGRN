"""Training and evaluation entry point for DCR-GRN."""

import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from torch_geometric import seed_everything as pyg_seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import parser
from models import DCRGRN
from utils.data_utils import DCRGRNDataset, load_expression_data
from utils.eval_utils import evaluate_auc_ap
from utils.graph_utils import construct_knn_graph, train_node2vec_embeddings


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
DATASET_NAMES = ("hESC", "hHEP", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L")


def set_random_seed(seed):
    """Set deterministic random states on CPU and CUDA."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_seeds(args):
    """Return the exact random seeds requested for repeated runs."""

    if args.seed_list.strip():
        seeds = [int(value.strip()) for value in args.seed_list.split(",")]
    else:
        seeds = [args.seed + index for index in range(args.runs)]
    if len(seeds) != args.runs:
        raise ValueError(
            f"Expected {args.runs} seeds, but --seed_list contains {len(seeds)}."
        )
    return seeds


def checkpoint_name(variant, args, seed, run_index):
    """Construct a stable public checkpoint file name."""

    model_tag = "DCRGRN" if variant == "dcrgrn" else variant
    return (
        f"{model_tag}_Specific_{args.dataset}{args.num}_"
        f"seed{seed}_run{run_index}_best_model.pt"
    )


def find_backbone_checkpoint(args, seed, run_index):
    """Find the seed-matched retained backbone checkpoint."""

    checkpoint_dir = Path(args.teacher_dir).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (CODE_DIR / checkpoint_dir).resolve()
    candidates = [
        checkpoint_dir / checkpoint_name("backbone", args, seed, run_index),
        checkpoint_dir / checkpoint_name("backbone", args, seed, 1),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_matching_parameters(model, checkpoint):
    """Load common parameters shared by the backbone and DCR-GRN."""

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    return len(compatible), incompatible


def load_training_edges(split_dir):
    """Load positive training edges used as the observed prior GRN."""

    frame = pd.read_csv(split_dir / "Train_set.csv", index_col=0, header=0)
    edges = frame.loc[frame["Label"] == 1, ["TF", "Target"]].to_numpy()
    return torch.as_tensor(edges, dtype=torch.long).t().contiguous()


def train_epoch(
    model,
    loader,
    grn_data,
    knn_graph,
    optimizer,
    device,
    epoch,
    args,
    teacher=None,
):
    """Train one epoch and return loss and training metrics."""

    model.train()
    total_loss = 0.0
    predictions = []
    labels_all = []

    for batch in tqdm(loader, ncols=70, leave=False):
        batch = batch.to(device)
        labels = batch.y.float().view(-1)
        optimizer.zero_grad()
        global_logit, local_logit, knn_logit, final_logit = model(
            batch, grn_data, knn_graph
        )

        if model.variant == "backbone":
            loss = model.evidential_loss(labels, epoch=epoch)
        else:
            loss = (
                BCEWithLogitsLoss()(final_logit.view(-1), labels)
                + BCEWithLogitsLoss()(local_logit.view(-1), labels)
                + BCEWithLogitsLoss()(knn_logit.view(-1), labels)
                + BCEWithLogitsLoss()(global_logit.view(-1), labels)
            )
            loss = loss + model.decomposition_loss(
                labels,
                lambda_reconstruction=args.lambda_reconstruction,
                lambda_orthogonality=args.lambda_orthogonality,
                lambda_shared=args.lambda_shared,
                lambda_contrastive=args.lambda_contrastive,
                lambda_routing=args.lambda_routing,
            )
            loss = loss + model.bottleneck_loss(
                labels,
                lambda_kl=args.cib_lambda_kl,
                lambda_gain=args.cib_lambda_gain,
                lambda_entropy=args.cib_lambda_entropy,
            )

            if teacher is not None and args.anchor_weight > 0:
                with torch.no_grad():
                    teacher_logit = teacher(batch, grn_data, knn_graph)[-1].view(-1)
                    teacher_probability = torch.sigmoid(teacher_logit)
                    confident = (
                        torch.abs(teacher_probability - 0.5)
                        > args.anchor_confidence
                    )
                    correct = (teacher_probability >= 0.5).eq(labels >= 0.5)
                    anchor_mask = confident & correct
                if anchor_mask.any():
                    anchor_loss = (
                        final_logit.view(-1)[anchor_mask]
                        - teacher_logit[anchor_mask]
                    ).pow(2).mean()
                    loss = loss + args.anchor_weight * anchor_loss

        loss = loss + model.auxiliary_loss()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions.append(final_logit.detach().view(-1).cpu())
        labels_all.append(labels.detach().cpu())

    predictions = torch.cat(predictions)
    labels_all = torch.cat(labels_all)
    mean_loss = total_loss / max(len(loader), 1)
    return mean_loss, evaluate_auc_ap(predictions, labels_all)


@torch.no_grad()
def evaluate(model, loader, grn_data, knn_graph, device):
    """Evaluate final edge logits on a fixed data split."""

    model.eval()
    total_loss = 0.0
    predictions = []
    labels_all = []
    for batch in tqdm(loader, ncols=70, leave=False):
        batch = batch.to(device)
        labels = batch.y.float().view(-1)
        final_logit = model(batch, grn_data, knn_graph)[-1].view(-1)
        total_loss += BCEWithLogitsLoss()(final_logit, labels).item()
        predictions.append(final_logit.cpu())
        labels_all.append(labels.cpu())
    predictions = torch.cat(predictions)
    labels_all = torch.cat(labels_all)
    return total_loss / max(len(loader), 1), evaluate_auc_ap(
        predictions, labels_all
    )


def prepare_data(args, seed, device):
    """Load fixed splits and construct all graph views."""

    expression_file = (
        PROJECT_DIR
        / "Benchmark Dataset"
        / "Specific Dataset"
        / args.dataset
        / f"TFs+{args.num}"
        / "BL--ExpressionData.csv"
    )
    split_dir = PROJECT_DIR / "data" / "Specific" / f"{args.dataset} {args.num}"
    cache_dir = PROJECT_DIR / "Data_process" / "Specific" / f"{args.dataset} {args.num}"
    expression_dataset = load_expression_data(expression_file, cache_dir)

    knn_graph = construct_knn_graph(expression_dataset[0], device=device)
    knn_graph.x = train_node2vec_embeddings(knn_graph, seed=seed)
    set_random_seed(seed)

    dataset_tag = f"{args.dataset}_{args.num}"
    train_dataset = DCRGRNDataset(
        expression_dataset, split_dir, dataset_tag, num_hops=2, split="train"
    )
    validation_dataset = DCRGRNDataset(
        expression_dataset,
        split_dir,
        dataset_tag,
        num_hops=2,
        split="validation",
    )
    test_dataset = DCRGRNDataset(
        expression_dataset, split_dir, dataset_tag, num_hops=2, split="test"
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.bs, shuffle=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    features = expression_dataset[0].x.to(device)
    directed_edges = load_training_edges(split_dir).to(device)
    grn_data = Data(x=features, edge_index=directed_edges).to(device)
    grn_data.edge_index_directed = directed_edges
    return (
        train_dataset,
        train_loader,
        validation_loader,
        test_loader,
        grn_data,
        knn_graph.to(device),
    )


def build_model(args, train_dataset, grn_size, device):
    """Instantiate a configured DCR-GRN model."""

    return DCRGRN(
        train_dataset=train_dataset,
        grn_size=grn_size,
        in_channels=train_dataset[0].num_features,
        hidden_channels=32,
        out_channels=32,
        num_layers=args.num_layers,
        variant=args.variant,
    ).to(device)


def run_once(args, seed, run_index, logger):
    """Train and evaluate one seed."""

    set_random_seed(seed)
    pyg_seed_everything(seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    (
        train_dataset,
        train_loader,
        validation_loader,
        test_loader,
        grn_data,
        knn_graph,
    ) = prepare_data(args, seed, device)
    model = build_model(args, train_dataset, grn_data.x.size(1), device)

    teacher = None
    if args.variant != "backbone" and args.load_backbone:
        backbone_path = find_backbone_checkpoint(args, seed, run_index)
        if backbone_path.exists():
            loaded, incompatible = load_matching_parameters(model, backbone_path)
            logger.info(
                "Initialized %d matching parameters from %s; missing=%d, unexpected=%d",
                loaded,
                backbone_path,
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
            if args.anchor_weight > 0:
                teacher_args = argparse_namespace_with_variant(args, "backbone")
                teacher = build_model(
                    teacher_args, train_dataset, grn_data.x.size(1), device
                )
                teacher.load_state_dict(
                    torch.load(
                        backbone_path,
                        map_location=device,
                        weights_only=True,
                    )
                )
                teacher.eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
        else:
            logger.warning(
                "Backbone checkpoint not found at %s; training from random initialization.",
                backbone_path,
            )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    checkpoint_dir = PROJECT_DIR / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name(
        args.variant, args, seed, run_index
    )

    best_validation_auc = float("-inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            grn_data,
            knn_graph,
            optimizer,
            device,
            epoch,
            args,
            teacher=teacher,
        )
        _, validation_metrics = evaluate(
            model, validation_loader, grn_data, knn_graph, device
        )
        validation_auc = validation_metrics["AUC"]
        logger.info(
            "Epoch %03d | loss %.4f | train AUROC %.4f AUPRC %.4f | "
            "validation AUROC %.4f AUPRC %.4f",
            epoch,
            train_loss,
            train_metrics["AUC"],
            train_metrics["AP"],
            validation_auc,
            validation_metrics["AP"],
        )
        if round(validation_auc, 4) > round(best_validation_auc, 4):
            best_validation_auc = validation_auc
            patience = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience += 1
        scheduler.step()
        if patience >= args.patience:
            break

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    _, test_metrics = evaluate(model, test_loader, grn_data, knn_graph, device)
    logger.info(
        "Test | AUROC %.4f AUPRC %.4f EPR@10 %.4f F1@10 %.4f",
        test_metrics["AUC"],
        test_metrics["AP"],
        test_metrics.get("EPR@10", 0.0),
        test_metrics.get("F1@10", 0.0),
    )
    return np.asarray(
        [
            test_metrics["AUC"],
            test_metrics["AP"],
            test_metrics.get("EPR@10", 0.0),
            test_metrics.get("F1@10", 0.0),
        ]
    )


def argparse_namespace_with_variant(args, variant):
    """Copy parsed arguments while replacing only the model variant."""

    import argparse

    values = vars(args).copy()
    values["variant"] = variant
    return argparse.Namespace(**values)


def configure_logger(args):
    """Create console and file logging handlers."""

    log_dir = PROJECT_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / (
        f"{args.variant}_Specific_{args.dataset}{args.num}_{timestamp}.log"
    )
    logger = logging.getLogger("dcrgrn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def main():
    args = parser.parse_args()
    if args.dataset not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    logger = configure_logger(args)
    logger.info("Configuration: %s", args)

    results = []
    for run_index, seed in enumerate(parse_seeds(args), start=1):
        logger.info("Run %d/%d | seed %d", run_index, args.runs, seed)
        results.append(run_once(args, seed, run_index, logger))

    result_matrix = np.stack(results)
    means = result_matrix.mean(axis=0)
    standard_deviations = result_matrix.std(axis=0)
    logger.info(
        "Mean +/- std | AUROC %.4f +/- %.4f | AUPRC %.4f +/- %.4f | "
        "EPR@10 %.4f +/- %.4f | F1@10 %.4f +/- %.4f",
        means[0],
        standard_deviations[0],
        means[1],
        standard_deviations[1],
        means[2],
        standard_deviations[2],
        means[3],
        standard_deviations[3],
    )


if __name__ == "__main__":
    main()
