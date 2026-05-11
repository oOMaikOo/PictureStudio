"""
Extended metrics: accuracy, precision, recall, F1, confusion matrix, ROC/AUC, top-k.
"""
from typing import List, Dict, Optional


def compute_metrics(
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    pred_probs: Optional[List[List[float]]] = None,
) -> Dict:
    n = len(class_names)
    if not true_labels:
        return {}

    # Confusion matrix
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < n and 0 <= p < n:
            cm[t][p] += 1

    total = len(true_labels)
    correct = sum(cm[i][i] for i in range(n))
    accuracy = correct / total if total else 0.0

    # Per-class P/R/F1
    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n) if j != i)
        fn = sum(cm[i][j] for j in range(n) if j != i)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Macro averages
    macro_prec = sum(v["precision"] for v in per_class.values()) / n if n else 0
    macro_rec = sum(v["recall"] for v in per_class.values()) / n if n else 0
    macro_f1 = sum(v["f1"] for v in per_class.values()) / n if n else 0

    # Weighted averages
    total_samples = sum(v["support"] for v in per_class.values()) or 1
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / total_samples

    result = {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": class_names,
        "total_samples": total,
    }

    # ROC/AUC (binary or multi-class OVR) – optional, needs probabilities
    if pred_probs and n == 2:
        try:
            roc = _compute_binary_roc_auc(true_labels, [p[1] for p in pred_probs])
            result["roc_auc"] = round(roc, 4)
            result["roc_curve"] = _compute_roc_curve(true_labels, [p[1] for p in pred_probs])
        except Exception:
            pass

    # Top-k accuracy (if multi-class, k=3)
    if pred_probs and n > 2:
        result["top3_accuracy"] = round(_top_k_accuracy(true_labels, pred_probs, k=3), 4)

    return result


def _compute_binary_roc_auc(true_labels: List[int], scores: List[float]) -> float:
    """Simple trapezoidal AUC for binary classification."""
    paired = sorted(zip(scores, true_labels), reverse=True)
    pos = sum(true_labels)
    neg = len(true_labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp  # trapezoidal
    return auc / (pos * neg)


def _compute_roc_curve(true_labels: List[int], scores: List[float]) -> Dict:
    paired = sorted(zip(scores, true_labels), reverse=True)
    pos = sum(true_labels)
    neg = len(true_labels) - pos
    tprs, fprs, thresholds = [0.0], [0.0], [1.0]
    tp = fp = 0
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / pos if pos else 0)
        fprs.append(fp / neg if neg else 0)
        thresholds.append(score)
    tprs.append(1.0)
    fprs.append(1.0)
    return {"tpr": tprs, "fpr": fprs, "thresholds": thresholds}


def _top_k_accuracy(true_labels: List[int], probs: List[List[float]], k: int) -> float:
    correct = 0
    for true, prob in zip(true_labels, probs):
        top_k = sorted(range(len(prob)), key=lambda i: prob[i], reverse=True)[:k]
        if true in top_k:
            correct += 1
    return correct / len(true_labels) if true_labels else 0.0


def compute_multilabel_metrics(
    true_matrix: List[List[float]],
    pred_matrix: List[List[float]],
    class_names: List[str],
) -> Dict:
    """Compute metrics for multi-label classification (sigmoid + 0.5 threshold outputs)."""
    n_samples = len(true_matrix)
    n_classes = len(class_names)
    if n_samples == 0 or n_classes == 0:
        return {}

    # Hamming accuracy: fraction of individual label predictions that are correct
    total_bits = n_samples * n_classes
    correct_bits = sum(
        sum(1 for t, p in zip(t_row, p_row) if round(t) == round(p))
        for t_row, p_row in zip(true_matrix, pred_matrix)
    )
    hamming_acc = correct_bits / total_bits if total_bits else 0.0

    # Exact match: every label correct
    exact_correct = sum(
        1 for t_row, p_row in zip(true_matrix, pred_matrix)
        if all(round(t) == round(p) for t, p in zip(t_row, p_row))
    )
    exact_acc = exact_correct / n_samples if n_samples else 0.0

    # Per-class P/R/F1
    per_class = {}
    for i, name in enumerate(class_names):
        tp = sum(1 for t, p in zip(true_matrix, pred_matrix)
                 if round(t[i]) == 1 and round(p[i]) == 1)
        fp = sum(1 for t, p in zip(true_matrix, pred_matrix)
                 if round(t[i]) == 0 and round(p[i]) == 1)
        fn = sum(1 for t, p in zip(true_matrix, pred_matrix)
                 if round(t[i]) == 1 and round(p[i]) == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
        }

    macro_f1 = sum(v["f1"] for v in per_class.values()) / n_classes if n_classes else 0.0
    macro_prec = sum(v["precision"] for v in per_class.values()) / n_classes if n_classes else 0.0
    macro_rec = sum(v["recall"] for v in per_class.values()) / n_classes if n_classes else 0.0

    return {
        "accuracy": round(hamming_acc, 4),          # compat key used by training page
        "hamming_accuracy": round(hamming_acc, 4),
        "exact_match_accuracy": round(exact_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(macro_f1, 4),
        "per_class": per_class,
        "class_names": class_names,
        "total_samples": n_samples,
        "is_multi_label": True,
    }


def format_metrics_text(metrics: Dict) -> str:
    if not metrics:
        return "Keine Metriken verfügbar."
    lines = [
        f"Accuracy:          {metrics.get('accuracy', 0)*100:.2f}%",
        f"Precision (Macro): {metrics.get('macro_precision', 0)*100:.2f}%",
        f"Recall (Macro):    {metrics.get('macro_recall', 0)*100:.2f}%",
        f"F1 (Macro):        {metrics.get('macro_f1', 0)*100:.2f}%",
        f"F1 (Weighted):     {metrics.get('weighted_f1', 0)*100:.2f}%",
    ]
    if "roc_auc" in metrics:
        lines.append(f"ROC-AUC:           {metrics.get('roc_auc', 0):.4f}")
    if "top3_accuracy" in metrics:
        lines.append(f"Top-3 Accuracy:    {metrics.get('top3_accuracy', 0)*100:.2f}%")
    lines += ["", "Klassen-Metriken:"]
    for cls, vals in metrics.get("per_class", {}).items():
        lines.append(
            f"  {cls:<22} P={vals['precision']*100:.1f}%  "
            f"R={vals['recall']*100:.1f}%  F1={vals['f1']*100:.1f}%  (n={vals['support']})"
        )
    return "\n".join(lines)
