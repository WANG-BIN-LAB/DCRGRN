import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_auc_ap(y_pred, y_true, top_k_rates=[0.01, 0.05, 0.1, 0.2]):
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise ValueError('Both y_pred and y_true need to be torch.Tensor.')
    y_pred = torch.sigmoid(y_pred)
    yt = y_true.cpu().numpy().flatten().astype(int)
    yp = y_pred.cpu().numpy().flatten()

    auc = roc_auc_score(yt, yp)
    ap = average_precision_score(yt, yp)
    result = {'AUC': auc, 'AP': ap}

    si = np.argsort(yp)[::-1]; st = yt[si]; n = len(st); np_pos = st.sum()
    for rate in top_k_rates:
        k = max(1, int(n * rate)); tk = st[:k]; pk = tk.sum()
        prec = pk / k; rec = pk / max(np_pos, 1)
        f1 = 2*prec*rec / max(prec+rec, 1e-8)
        epr = prec / max(np_pos/max(n,1), 1e-8)
        pct = int(rate*100)
        result[f'EPR@{pct:02d}'] = round(epr, 4)
        result[f'F1@{pct:02d}'] = round(f1, 4)
        result[f'Prec@{pct:02d}'] = round(prec, 4)
        result[f'Rec@{pct:02d}'] = round(rec, 4)
    return result
