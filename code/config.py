"""Command-line configuration for DCR-GRN experiments."""

import argparse
import torch


def str_to_bool(value):
    """Parse common command-line boolean spellings."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


parser = argparse.ArgumentParser(
    description=(
        "Train DCR-GRN for cell-type-specific gene regulatory network "
        "inference. Train the backbone before a DCR-GRN variant when matching "
        "backbone checkpoints are not available."
    )
)

# Training
parser.add_argument("--runs", type=int, default=5, help="number of repeated runs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int, default=80, help="maximum training epochs")
parser.add_argument("--cuda", type=str_to_bool, default=torch.cuda.is_available())
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--bs", type=int, default=32, help="batch size")
parser.add_argument("--patience", type=int, default=10, help="early-stopping patience")
parser.add_argument("--num_layers", type=int, default=2, help="number of GNN layers")
parser.add_argument("--seed", type=int, default=2022, help="base random seed")
parser.add_argument(
    "--seed_list",
    type=str,
    default="2022,2023,2024,2025,2026",
    help="comma-separated seeds; its length must equal --runs",
)
parser.add_argument(
    "--teacher_dir",
    type=str,
    default="../results/checkpoints",
    help="directory containing retained backbone checkpoints",
)

# Shared-private decomposition and conditional information bottleneck
parser.add_argument("--lambda_reconstruction", type=float, default=0.05)
parser.add_argument("--lambda_orthogonality", type=float, default=0.01)
parser.add_argument("--lambda_shared", type=float, default=0.01)
parser.add_argument("--lambda_contrastive", type=float, default=0.02)
parser.add_argument("--lambda_routing", type=float, default=0.005)
parser.add_argument("--load_backbone", type=str_to_bool, default=True)
parser.add_argument("--anchor_weight", type=float, default=0.02)
parser.add_argument("--anchor_confidence", type=float, default=0.25)
parser.add_argument("--cib_lambda_kl", type=float, default=5e-4)
parser.add_argument("--cib_lambda_gain", type=float, default=0.01)
parser.add_argument("--cib_lambda_entropy", type=float, default=0.001)

# Data
parser.add_argument("--num", type=str, default="500", choices=["500", "1000"])
parser.add_argument(
    "--dataset",
    type=str,
    default="hESC",
    choices=["hESC", "hHEP", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"],
)
parser.add_argument(
    "--variant",
    type=str,
    default="dcrgrn",
    choices=["backbone", "dcrgrn"],
    help="complete DCR-GRN or its retained first-stage backbone",
)
