# DCR-GRN

Official research implementation of **DCR-GRN: Dual-Domain Complementary
Representation Learning for Cell-Type-Specific Gene Regulatory Network
Inference**.

DCR-GRN models each candidate TF–target edge from two complementary domains:

1. the **expression domain**, combining TF–target identity profiles with an
   expression-similarity KNN graph; and
2. the **topology domain**, combining an enclosing-subgraph view with a
   directed global-GRN view.

Within each domain, masked shared–private factorization separates common and
view-specific evidence. A component-wise conditional information bottleneck
compresses these components, and structure-conditioned routing constructs the
final topology representation. The expression and topology representations
are then fused to predict the regulatory edge probability.

## Repository layout

```text
DCRGRN/
├── Benchmark Dataset/       # expression matrices and benchmark networks
├── data/                    # fixed train/validation/test edge splits
├── code/
│   ├── config.py            # centralized experiment configuration
│   ├── train1.py            # training and evaluation entry point
│   ├── models/
│   └── utils/
├── .gitignore
└── requirements.txt
```

`Data_process/` and `results/` are generated automatically during training and
are therefore not included in the release package.

## Installation

The code was verified with Python 3.10, PyTorch 2.5.0 (CUDA 12.4), and PyTorch
Geometric 2.6.1. Install a PyTorch build matching your CUDA driver first, then:

```bash
pip install -r requirements.txt
```

PyTorch Geometric operations used by KNNGraph and Node2Vec require compatible
`pyg-lib` and `torch-cluster` wheels. Install builds matching the selected
PyTorch and CUDA versions by following the PyTorch Geometric installation
instructions.

## Data

The release contains seven cell types (`hESC`, `hHEP`, `mDC`, `mESC`,
`mHSC-E`, `mHSC-GM`, and `mHSC-L`) under the TFs+500 and TFs+1000 settings.
Expression matrices are stored under `Benchmark Dataset/Specific Dataset/`,
while the fixed train, validation, and test edge splits are stored under
`data/Specific/`. The fixed splits should not be regenerated when reproducing
the reported results.

## Training

All default paper settings are defined in `code/config.py`. Command-line
arguments only override those defaults.

DCR-GRN follows a two-stage training protocol. The public variant name
`backbone` denotes the retained dual-domain base model, not an additional
method reported in the comparison table. Its checkpoint is used to initialize
DCR-GRN and to provide seed-matched teacher anchoring. The public variant name
`dcrgrn` denotes the complete proposed model.

For one dataset and one gene setting, run the two stages in order:

```bash
cd code
python train1.py --variant backbone --dataset hESC --num 500
python train1.py --variant dcrgrn --dataset hESC --num 500
```

To reproduce all seven cell types for TFs+500:

```bash
cd code
for dataset in hESC hHEP mDC mESC mHSC-E mHSC-GM mHSC-L; do
    python train1.py --variant backbone --dataset "$dataset" --num 500
    python train1.py --variant dcrgrn --dataset "$dataset" --num 500
done
```

Repeat the same commands with `--num 1000` for the TFs+1000 setting. The
default configuration runs five seeds (2022--2026), uses validation AUROC for
early stopping, and writes checkpoints and logs under `results/`.

## Outputs

- checkpoints: `results/checkpoints/`
- training logs: `results/logs/`
- preprocessing cache: `Data_process/`

Generated caches, checkpoints, and logs are excluded from Git. The raw
benchmark data and fixed splits are retained as ordinary files rather than
links to the original experiment directory.

## Reproducibility

The default evaluation protocol uses five fixed seeds (2022–2026), validation
AUROC for early stopping, and reports AUROC and AUPRC on the held-out test set.
