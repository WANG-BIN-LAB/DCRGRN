"""Command-line configuration for the released DCR-GRN experiments."""

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
        "inference. Use --variant backbone before --variant dcrgrn when no "
        "retained backbone checkpoints are available."
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
parser.add_argument("--ratio", type=float, default=0.4, help="legacy compatibility option")
parser.add_argument("--metric", type=str, default="auc_ap")
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
parser.add_argument("--spore_lambda_rec", type=float, default=0.05)
parser.add_argument("--spore_lambda_orth", type=float, default=0.01)
parser.add_argument("--spore_lambda_share", type=float, default=0.01)
parser.add_argument("--spore_lambda_syn", type=float, default=0.02)
parser.add_argument("--spore_lambda_degree", type=float, default=0.005)
parser.add_argument("--spore_load_retained", type=str_to_bool, default=True)
parser.add_argument("--spore_nf_anchor_weight", type=float, default=0.02)
parser.add_argument("--spore_nf_anchor_delta", type=float, default=0.25)
parser.add_argument("--cib_lambda_kl", type=float, default=5e-4)
parser.add_argument("--cib_lambda_gain", type=float, default=0.01)
parser.add_argument("--cib_lambda_entropy", type=float, default=0.001)

# Data
parser.add_argument("--netType", type=str, default="Specific", choices=["Specific"])
parser.add_argument("--num", type=str, default="500", choices=["500", "1000"])
parser.add_argument(
    "--dataset",
    type=str,
    default="hESC",
    choices=["hESC", "hHEP", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"],
)
parser.add_argument("--train_percent", type=float, default=1.0)
parser.add_argument("--val_percent", type=float, default=1.0)
parser.add_argument("--test_percent", type=float, default=1.0)
parser.add_argument(
    "--variant",
    type=str,
    default="dcrgrn",
    choices=["dcrgrn", "backbone", "spore_cib", "directed_idpath_evidential_pcconv_lowdeg"],
    help="public model name; internal names remain accepted for old checkpoints",
)
