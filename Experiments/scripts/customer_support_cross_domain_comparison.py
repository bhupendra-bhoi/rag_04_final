"""Cross-domain comparison for the customer-support sweep (techqa/emanual/delucionqa).

Reads the per-config detailed JSON reports (not just comparison.csv) from all
three -openrouter-experiment/reports/ dirs and produces a comparison that:

  1. Reports refusal rate separately from adherence. Refused queries
     ("The passages do not provide sufficient information...") are scored
     utilization=0, completeness=0, adherence≈1 by the judge, which inflates
     raw adherence for configs that refuse more often. techqa in particular
     showed 35-55% refusal rates that made its raw adherence numbers
     uninterpretable; this recomputes adherence/utilization/completeness over
     answered queries only so configs are compared on answer quality, not on
     how often they gave up.

  2. Ranks configs within each dataset (1=best) on the answered-only metrics,
     then averages that rank across datasets by config "slot" (v1, v2, ...)
     to surface what generalizes across the whole customer-support domain
     rather than what wins on any single subset.

Usage:
    python3 scripts/customer_support_cross_domain_comparison.py

Run this after adding new configs (e.g. v9_bge_sentence, v10_hyde_match) and
re-running the relevant notebook(s) + runner.compare(), so the new reports
exist on disk.
"""
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "techqa": REPO_ROOT / "rag-experiments/techqa-openrouter-experiment/reports",
    "emanual": REPO_ROOT / "rag-experiments/emanual-openrouter-experiment/reports",
    "delucionqa": REPO_ROOT / "rag-experiments/delucionqa-openrouter-experiment/reports",
}

METRICS = ["relevance_score", "utilization_score", "completeness_score", "adherence_score"]
REFUSAL_MARKER = "sufficient information"


def nanmean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else float("nan")


def slot_of(config_name):
    """Extract the 'v1'..'v10' slot from a config name like 'techqa_or_v9_bge_sentence'."""
    m = re.search(r"_(v\d+)_", config_name)
    return m.group(1) if m else config_name


def load_dataset_configs(report_dir):
    configs = []
    for path in sorted(report_dir.glob("*.json")):
        report = json.loads(path.read_text())
        section = report["sections"][0]
        name = section["config_name"]
        per_query = section["per_query"]
        n = len(per_query)
        refused = [q for q in per_query if REFUSAL_MARKER in (q.get("answer") or "")]
        answered = [q for q in per_query if q not in refused]

        raw = {m: x["mean_score"] for x in section["summary"] for m in [x["metric"]] if m in METRICS}
        answered_only = {
            m: nanmean([q[f"{m}__pred"] for q in answered]) for m in METRICS
        }

        configs.append({
            "name": name,
            "slot": slot_of(name),
            "n": n,
            "refusals": len(refused),
            "refusal_rate": len(refused) / n if n else float("nan"),
            "raw": raw,
            "answered_only": answered_only,
        })
    return configs


def rank_within_dataset(configs):
    """Average rank (1=best) across the 4 answered-only metrics, per config."""
    ranks = {c["name"]: [] for c in configs}
    for metric in METRICS:
        ranked = sorted(configs, key=lambda c: -c["answered_only"][metric])
        for i, c in enumerate(ranked):
            ranks[c["name"]].append(i + 1)
    return {name: sum(rs) / len(rs) for name, rs in ranks.items()}


def main():
    per_dataset = {}
    for dataset, report_dir in DATASETS.items():
        if not report_dir.exists() or not any(report_dir.glob("*.json")):
            print(f"[skip] {dataset}: no reports found at {report_dir}")
            continue
        configs = load_dataset_configs(report_dir)
        avg_rank = rank_within_dataset(configs)
        for c in configs:
            c["avg_rank"] = avg_rank[c["name"]]
        per_dataset[dataset] = configs

    if not per_dataset:
        print("No reports found in any dataset — nothing to compare.")
        return

    # --- Per-dataset detail: refusal rate + answered-only metrics ---
    for dataset, configs in per_dataset.items():
        print(f"\n{'=' * 100}")
        print(f"{dataset.upper()} — answered-only metrics (refusals excluded), rank 1=best of {len(configs)}")
        print(f"{'=' * 100}")
        header = f"{'config':38s} {'refuse%':>8s} {'rel':>7s} {'util':>7s} {'comp':>7s} {'adh':>7s} {'avg_rank':>9s}"
        print(header)
        for c in sorted(configs, key=lambda c: c["avg_rank"]):
            ao = c["answered_only"]
            print(
                f"{c['name']:38s} {c['refusal_rate']:>7.0%} "
                f"{ao['relevance_score']:>7.3f} {ao['utilization_score']:>7.3f} "
                f"{ao['completeness_score']:>7.3f} {ao['adherence_score']:>7.3f} {c['avg_rank']:>9.2f}"
            )

    # --- Cross-dataset aggregation by slot (v1, v2, ...) ---
    slot_data = {}
    for dataset, configs in per_dataset.items():
        for c in configs:
            slot_data.setdefault(c["slot"], {})[dataset] = c

    print(f"\n{'=' * 100}")
    print("CROSS-DATASET — mean answered-only rank by config slot (lower=better; '-' = not run in that dataset)")
    print(f"{'=' * 100}")
    dataset_names = list(per_dataset.keys())
    header = f"{'slot':6s} " + " ".join(f"{d:>12s}" for d in dataset_names) + f" {'mean_rank':>10s}  configs"
    print(header)

    rows = []
    for slot, by_dataset in slot_data.items():
        ranks = [by_dataset[d]["avg_rank"] for d in dataset_names if d in by_dataset]
        mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        names = " / ".join(by_dataset[d]["name"] for d in dataset_names if d in by_dataset)
        rows.append((slot, by_dataset, mean_rank, names, len(ranks)))

    for slot, by_dataset, mean_rank, names, coverage in sorted(rows, key=lambda r: r[2]):
        cells = " ".join(
            f"{by_dataset[d]['avg_rank']:>12.2f}" if d in by_dataset else f"{'-':>12s}"
            for d in dataset_names
        )
        coverage_flag = "" if coverage == len(dataset_names) else f"  (only {coverage}/{len(dataset_names)} datasets)"
        print(f"{slot:6s} {cells} {mean_rank:>10.2f}  {names}{coverage_flag}")

    print(
        "\nNote: a slot's mean_rank across datasets is only a fair 'universal config' signal "
        "when the same underlying technique was used in every dataset for that slot — check "
        "the config names in the last column, not just the slot number, before trusting it."
    )


if __name__ == "__main__":
    main()
