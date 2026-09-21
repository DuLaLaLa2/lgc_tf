import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_binary_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    if len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
        metrics["aupr"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auc"] = float("nan")
        metrics["aupr"] = float("nan")

    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics["specificity"] = tn / (tn + fp + 1e-8)
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    metrics["tp"] = int(tp)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["threshold"] = float(threshold)

    return metrics


def find_best_threshold(
    y_true,
    y_prob,
    metric_name: str = "mcc",
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
):
    best_threshold = 0.5
    best_score = -1e9
    best_metrics = None

    thresholds = np.arange(start, end + 1e-8, step)

    for threshold in thresholds:
        metrics = compute_binary_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
        )

        score = metrics[metric_name]

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics