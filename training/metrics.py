from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def binary_metrics(y_true_attack, p_attack, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true_attack).astype(bool)
    p = np.asarray(p_attack, dtype=float)
    pred = p >= threshold
    tp = int((y & pred).sum())
    tn = int((~y & ~pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    both_classes = 0 < y.sum() < len(y)
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "accuracy": _div(tp + tn, len(y)),
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "fpr": _div(fp, fp + tn),  # false alarms among normal flows
        "fnr": _div(fn, fn + tp),  # missed attacks among attack flows
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pr_auc": float(average_precision_score(y, p)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "confusion_matrix": [
            [tn, fp],
            [fn, tp],
        ],  # rows: true Normal/Attack; cols: predicted
    }


def multiclass_metrics(y_true, y_pred, labels: list[str]) -> dict[str, Any]:
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "per_class": {
            lab: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for lab, p, r, f, s in zip(labels, prec, rec, f1, sup)
        },
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }


def threshold_sweep(y_true_attack, p_attack, thresholds=None) -> list[dict[str, Any]]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        m = binary_metrics(y_true_attack, p_attack, float(t))
        rows.append(
            {
                k: m[k]
                for k in (
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "fpr",
                    "fnr",
                    "fp",
                    "fn",
                )
            }
        )
    return rows


def evaluate(
    classes, y_true, proba, normal_label: str = "Normal", threshold: float = 0.5
) -> dict[str, Any]:
    classes = [str(c) for c in classes]
    proba = np.asarray(proba)
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(classes, dtype=object)[np.argmax(proba, axis=1)]
    normal_idx = classes.index(normal_label)
    p_attack = np.clip(1.0 - proba[:, normal_idx], 0.0, 1.0)
    labels = sorted(set(classes) | set(y_true), key=lambda c: (c != normal_label, c))
    return {
        "multiclass": multiclass_metrics(y_true, y_pred.astype(str), labels),
        "binary": binary_metrics(y_true != normal_label, p_attack, threshold),
    }
