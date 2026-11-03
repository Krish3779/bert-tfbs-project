import csv
import os

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


class MetricTracker:
    """Collect predictions over one pass, then compute all metrics at once.

    probs  : model output, shape (B, 2), rows are class probabilities
    labels : true labels, shape (B,)
    """

    def __init__(self):
        self.y_true, self.y_pred, self.y_score = [], [], []
        self.loss_sum, self.n = 0.0, 0

    def update(self, probs, labels, loss=None):
        probs = probs.detach().float().cpu()
        labels = labels.detach().cpu()
        self.y_true += labels.tolist()
        self.y_pred += probs.argmax(dim=1).tolist()
        self.y_score += (probs[:, 1] / probs.sum(dim=1)).tolist()
        if loss is not None:
            self.loss_sum += float(loss) * labels.size(0)
        self.n += labels.size(0)

    def compute(self):
        yt, yp, ys = np.array(self.y_true), np.array(self.y_pred), np.array(self.y_score)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        try:
            roc, pr = roc_auc_score(yt, ys), average_precision_score(yt, ys)
        except ValueError:  # only one class present
            roc = pr = float('nan')
        return {
            'loss': self.loss_sum / max(self.n, 1),
            'acc': accuracy_score(yt, yp),
            'precision': precision_score(yt, yp, zero_division=0),
            'recall': recall_score(yt, yp, zero_division=0),   # = sensitivity
            'specificity': float(tn) / (tn + fp) if (tn + fp) else float('nan'),
            'f1': f1_score(yt, yp, zero_division=0),
            'mcc': matthews_corrcoef(yt, yp),
            'roc_auc': roc,
            'pr_auc': pr,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        }


def append_csv(path, row):
    """Append one dict as a row; write the header if the file is new."""
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
