"""Prediction container and metrics (accuracy, F1, AUROC, calibration, selective accuracy).

All metrics operate on numpy arrays so they work on pooled out-of-fold predictions too.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from config import cfg


@dataclass
class Predictions:
    """Per-sample outputs of a model over one loader."""
    labels: np.ndarray        # (N,)
    probs: np.ndarray         # (N, C) decision probabilities
    uncertainty: np.ndarray   # (N,) mean(Pl − Bel)
    K: np.ndarray             # (N,) conflict mass
    gate: np.ndarray          # (N,) soft high-conflict gate
    blend: np.ndarray         # (N,) weight given to the aux head
    conflict_flag: np.ndarray  # (N,) K above the reporting threshold

    @property
    def preds(self) -> np.ndarray:
        return self.probs.argmax(axis=1)


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, k: int = cfg.NUM_CLASSES) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    return np.bincount(np.asarray(labels) * k + np.asarray(preds), minlength=k * k).reshape(k, k)


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(labels: np.ndarray, preds: np.ndarray, k: int = cfg.NUM_CLASSES) -> float:
    return float(per_class_f1(confusion_matrix(labels, preds, k)).mean())


def balanced_accuracy(cm: np.ndarray) -> float:
    return float(np.mean(np.diag(cm) / np.clip(cm.sum(axis=1), 1, None)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = which == b
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def macro_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """One-vs-rest AUROC averaged over classes that have both positives and negatives."""
    scores = [roc_auc_score(labels == c, probs[:, c]) for c in range(probs.shape[1])
              if 0 < (labels == c).sum() < len(labels)]
    return float(np.mean(scores)) if scores else float('nan')


def selective_accuracy(correct: np.ndarray, risk: np.ndarray, coverage: float) -> float:
    """Accuracy on the ``coverage`` fraction of samples with the lowest ``risk``."""
    keep = max(1, int(np.ceil(coverage * len(correct))))
    return float(correct[np.argsort(risk, kind='stable')[:keep]].mean())


def compute_metrics(p: Predictions) -> Dict:
    k = p.probs.shape[1]
    preds = p.preds
    cm = confusion_matrix(p.labels, preds, k)
    correct = preds == p.labels
    return {
        'acc': float(correct.mean()),
        'macro_f1': float(per_class_f1(cm).mean()),
        'per_class_f1': per_class_f1(cm).tolist(),
        'bal_acc': balanced_accuracy(cm),
        'auroc': macro_auroc(p.probs, p.labels),
        'brier': brier_score(p.probs, p.labels),
        'ece': expected_calibration_error(p.probs, p.labels),
        'sel_acc_unc': selective_accuracy(correct, p.uncertainty, cfg.SELECTIVE_COVERAGE),
        'sel_acc_conf': selective_accuracy(correct, -p.probs.max(axis=1), cfg.SELECTIVE_COVERAGE),
        'conflict_rate': float(p.conflict_flag.mean()),
        'mean_K': float(p.K.mean()),
        'mean_gate': float(p.gate.mean()),
        'mean_decision_blend': float(p.blend.mean()),
        'avg_uncertainty': float(p.uncertainty.mean()),
        'confusion': cm,
    }
