"""Evaluation metrics for binary regulatory-edge prediction."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_auc_ap(y_pred, y_true, top_k_rates=(0.01, 0.05, 0.1, 0.2)):
    """Compute AUROC, AUPRC, and top-ranked-edge enrichment metrics."""

    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise TypeError("Both y_pred and y_true must be torch.Tensor instances.")
    y_pred = torch.sigmoid(y_pred)
    yt = y_true.cpu().numpy().flatten().astype(int)
    yp = y_pred.cpu().numpy().flatten()

    auc = roc_auc_score(yt, yp)
    ap = average_precision_score(yt, yp)
    result = {"AUC": auc, "AP": ap}

    sorted_indices = np.argsort(yp)[::-1]
    sorted_labels = yt[sorted_indices]
    sample_count = len(sorted_labels)
    positive_count = sorted_labels.sum()
    for rate in top_k_rates:
        k = max(1, int(sample_count * rate))
        selected_positive_count = sorted_labels[:k].sum()
        precision = selected_positive_count / k
        recall = selected_positive_count / max(positive_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        base_rate = positive_count / max(sample_count, 1)
        enrichment = precision / max(base_rate, 1e-8)
        percentage = int(rate * 100)
        result[f"EPR@{percentage:02d}"] = round(enrichment, 4)
        result[f"F1@{percentage:02d}"] = round(f1, 4)
        result[f"Prec@{percentage:02d}"] = round(precision, 4)
        result[f"Rec@{percentage:02d}"] = round(recall, 4)
    return result
