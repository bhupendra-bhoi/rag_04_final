"""Score the adherence judge as the binary classifier it actually is.

RAGBench's `adherence_score` is a boolean per row (are all claims in the answer
supported by the retrieved context?), and our judge predicts the same boolean
per row. comparison.csv currently scores it by averaging the booleans into a
fraction per config and taking MAE against the gold fraction — which can hide
whether the judge is getting individual rows right: two configs can land on
the same MAE while one gets every row's call correct and the other is wrong
about a completely different half of the rows.

This scores it as a classification problem instead: precision/recall/F1 for
the "adherent" class, plus accuracy and balanced accuracy, computed per config
from the same per_query records already in every report JSON.

Note on AUC-ROC: our judge outputs a hard 0/1 call, not a probability, so a
real ROC curve only has one interior point. Algebraically, AUC for a
hard-binary predictor reduces exactly to balanced accuracy
((sensitivity + specificity) / 2) — reported as such below rather than
silently calling an ROC routine that can't do more than that with this input.

Usage:
    python3 scripts/adherence_classification_report.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def confusion_counts(rows):
    tp = fp = tn = fn = 0
    for gt, pred in rows:
        if gt == 1.0 and pred == 1.0:
            tp += 1
        elif gt == 0.0 and pred == 1.0:
            fp += 1
        elif gt == 0.0 and pred == 0.0:
            tn += 1
        elif gt == 1.0 and pred == 0.0:
            fn += 1
    return tp, fp, tn, fn


def safe_div(a, b):
    return a / b if b else float("nan")


def is_nan(x):
    return x != x


def classification_metrics(rows):
    tp, fp, tn, fn = confusion_counts(rows)
    n = tp + fp + tn + fn

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)  # sensitivity / TPR
    specificity = safe_div(tn, tn + fp)  # TNR
    f1 = (
        float("nan")
        if is_nan(precision) or is_nan(recall) or (precision + recall == 0)
        else 2 * precision * recall / (precision + recall)
    )
    accuracy = safe_div(tp + tn, n)
    balanced_accuracy = (
        float("nan")
        if is_nan(recall) or is_nan(specificity)
        else (recall + specificity) / 2
    )

    return {
        "n": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "gold_positive_rate": safe_div(tp + fn, n),
        "pred_positive_rate": safe_div(tp + fp, n),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy_auc": balanced_accuracy,
    }


def load_config_rows(report_path):
    data = json.loads(report_path.read_text())
    section = data["sections"][0]
    name = section.get("config_name", report_path.stem)

    rows = []
    for q in section.get("per_query", []):
        gt = q.get("adherence_score__gt")
        pred = q.get("adherence_score__pred")
        if gt is None or pred is None or is_nan(pred):
            continue
        rows.append((gt, pred))

    mae = None
    for m in section.get("summary", []):
        if m.get("metric") == "adherence_score":
            mae = m.get("mean_abs_error")

    return name, rows, mae


def fmt(x, spec):
    return "  n/a" if is_nan(x) else format(x, spec)


def main():
    report_files = sorted(REPO_ROOT.glob("rag-experiments/*/reports/*.json"))
    results = []
    for f in report_files:
        try:
            name, rows, mae = load_config_rows(f)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"[skip] {f}: {e}")
            continue
        if not rows:
            print(f"[skip] {name}: no scored adherence rows")
            continue
        metrics = classification_metrics(rows)
        metrics["config_name"] = name
        metrics["mae"] = mae
        metrics["experiment"] = f.parent.parent.name
        results.append(metrics)

    header = (
        f"{'config':38s} {'experiment':30s} {'n':>4s} {'gold+':>6s} {'pred+':>6s} "
        f"{'prec':>6s} {'recall':>6s} {'F1':>6s} {'acc':>6s} {'bal_acc=AUC':>11s} {'MAE':>6s}"
    )
    print(header)
    print("-" * len(header))

    def sort_key(r):
        return -r["f1"] if not is_nan(r["f1"]) else 999

    for m in sorted(results, key=sort_key):
        print(
            f"{m['config_name']:38s} {m['experiment']:30s} {m['n']:>4d} "
            f"{fmt(m['gold_positive_rate'], '>6.0%')} {fmt(m['pred_positive_rate'], '>6.0%')} "
            f"{fmt(m['precision'], '>6.3f')} {fmt(m['recall'], '>6.3f')} {fmt(m['f1'], '>6.3f')} "
            f"{fmt(m['accuracy'], '>6.3f')} {fmt(m['balanced_accuracy_auc'], '>11.3f')} "
            f"{m['mae']:>6.3f}"
        )

    print(
        "\nbal_acc=AUC: for a hard 0/1 predictor, AUC-ROC reduces algebraically to "
        "balanced accuracy ((sensitivity+specificity)/2) — see module docstring.\n"
        "n/a: undefined (e.g. no positive predictions/no positive gold rows in that config)."
    )


if __name__ == "__main__":
    main()
