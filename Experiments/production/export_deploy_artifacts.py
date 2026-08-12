"""Export the Customer Support domain (techqa/emanual/delucionqa) into the
shared-demo-app deploy format described in final_submission/TEAMMATE_GUIDE.md.

This is the local equivalent of the guide's "Step 2 Colab cell" — same output
schema, same alignment algorithm, just running against our own already-chosen
production pipeline (production/configs/*.yaml) instead of a fresh Colab
notebook. Each RAGBench subset we own (techqa, emanual, delucionqa) becomes
its own deploy_artifacts/<slug>/ folder — the app's schema is one-subset-per-
folder (entry.json has a single "subset" field), and the validator already
supports checking multiple folders in one run.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 — symmetric (encodes
queries and documents identically), satisfying the guide's one hard
compatibility rule. No asymmetric/dual-encoder setup was used, so nothing to
flag to the integration lead.

Usage:
    python3 production/export_deploy_artifacts.py --dataset emanual
    python3 production/export_deploy_artifacts.py --all
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# final_submission/ is a SIBLING of this repo (Capstone Project/final_submission/),
# not inside it — it's the shared folder with TEAMMATE_GUIDE.md + validate_my_folder.py.
SUBMISSION_ROOT = REPO_ROOT.parent / "final_submission"
OUT_ROOT = SUBMISSION_ROOT / "deploy_artifacts"
ZIP_ROOT = SUBMISSION_ROOT

DATASETS = {
    "techqa": {"label": "TechQA", "limit": 314},
    "emanual": {"label": "eManual", "limit": 132},
    "delucionqa": {"label": "DelucionQA", "limit": 184},
}

LEADING_INT_RE = re.compile(r"^(\d+)")


def load_production_config(dataset: str) -> dict:
    import yaml
    path = REPO_ROOT / "production" / "configs" / f"{dataset}.yaml"
    return yaml.safe_load(path.read_text())


def chunk_fixed_word(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Matches rag/modules/chunking/strategies/fixed_word/strategy.py exactly."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + max_words]))
        i += max_words - overlap_words
    return chunks


def build_corpus(rows, chunk_cfg: dict):
    """Pool + dedupe all documents (preserving order), then chunk each once.

    Matches TEAMMATE_GUIDE.md's alignment algorithm exactly:
        unique_docs = list(dict.fromkeys(all_docs))
        doc_to_gid = {doc: i for i, doc in enumerate(unique_docs)}
    """
    all_docs = [doc for row in rows for doc in row["documents"]]
    unique_docs = list(dict.fromkeys(all_docs))
    doc_to_gid = {doc: i for i, doc in enumerate(unique_docs)}

    chunks, chunk_parent = [], []
    for gid, doc in enumerate(unique_docs):
        for chunk_text in chunk_fixed_word(doc, chunk_cfg["max_words"], chunk_cfg["overlap_words"]):
            chunks.append(chunk_text)
            chunk_parent.append(gid)

    return chunks, chunk_parent, doc_to_gid


def gold_parent_gids_for_row(row, doc_to_gid) -> list[int]:
    """'all_relevant_sentence_keys' entries look like '2af' — the leading int
    is the document index WITHIN this row's own documents list. Map that
    row-local document through doc_to_gid to its index in unique_docs.
    """
    gids = set()
    for key in row.get("all_relevant_sentence_keys") or []:
        m = LEADING_INT_RE.match(key)
        if not m:
            continue
        local_doc_idx = int(m.group(1))
        if local_doc_idx >= len(row["documents"]):
            continue
        doc_text = row["documents"][local_doc_idx]
        gid = doc_to_gid.get(doc_text)
        if gid is not None:
            gids.add(gid)
    return sorted(gids)


def pick_demo_questions(rows, doc_to_gid, n=5) -> list[dict]:
    picked = []
    for row in rows:
        gids = gold_parent_gids_for_row(row, doc_to_gid)
        if not gids:
            continue
        picked.append({
            "question": row["question"],
            "gold_parent_gids": gids,
            "gold_relevance": row.get("relevance_score"),
            "gold_utilization": row.get("utilization_score"),
            "gold_completeness": row.get("completeness_score"),
            "gold_adherence": bool(row.get("adherence_score")),
        })
        if len(picked) == n:
            break
    return picked


def build_prompts(gen_cfg: dict) -> dict:
    system = gen_cfg["system_prompt"].strip()
    # The user_prompt template is boilerplate ("Passages:\n{context}\n\nQuestion:
    # {query}\n\n") plus one dataset-specific trailing directive — that
    # directive is what maps to "instructions"; the app inserts context/
    # question itself.
    instructions = gen_cfg["user_prompt"].strip().splitlines()[-1].strip()
    return {
        "default": "default",
        "prompts": {"default": {"system": system, "instructions": instructions}},
    }


def export_dataset(dataset: str):
    # Imported lazily so --zip-only can run with zero ML dependencies installed.
    import faiss
    import yaml
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    meta = DATASETS[dataset]
    slug = dataset
    print(f"\n=== {meta['label']} ({dataset}) ===")

    config = load_production_config(dataset)
    embed_model_name = config["embedding"]["config"]["model_name"]
    chunk_cfg = config["chunking"]["config"]

    print(f"Loading {dataset} test split...")
    ds = load_dataset("galileo-ai/ragbench", dataset, split="test")
    rows = [dict(r) for r in ds]
    print(f"  {len(rows)} rows")

    chunks, chunk_parent, doc_to_gid = build_corpus(rows, chunk_cfg)
    print(f"  {len(doc_to_gid)} unique documents -> {len(chunks)} chunks")

    questions = pick_demo_questions(rows, doc_to_gid, n=5)
    if len(questions) < 5:
        print(f"  WARNING: only found {len(questions)} questions with non-empty gold_parent_gids")

    print(f"  embedding {len(chunks)} chunks with {embed_model_name}...")
    embedder = SentenceTransformer(embed_model_name)
    vectors = embedder.encode(
        chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    assert index.ntotal == len(chunks)

    out_dir = OUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "corpus.json").write_text(json.dumps({
        "chunks": chunks,
        "chunk_parent": chunk_parent,
        "embed_model": embed_model_name,
    }))

    faiss.write_index(index, str(out_dir / "index.faiss"))

    (out_dir / "questions.json").write_text(json.dumps(questions, indent=2))

    (out_dir / "prompts.json").write_text(json.dumps(build_prompts(config["generation"]["config"]), indent=2))

    (out_dir / "entry.json").write_text(json.dumps({
        "label": meta["label"],
        "slug": slug,
        "subset": dataset,
        "embed_model": embed_model_name,
        "chunks": len(chunks),
        "questions": len(questions),
    }, indent=2))

    print(f"  wrote {out_dir}")
    print(f"  summary: {len(chunks)} chunks, {len(questions)} questions, 1 prompt, embed={embed_model_name}")


def zip_all_domains(combined_name: str = "customer_support"):
    """One zip covering every domain folder currently under OUT_ROOT (deploy_artifacts/).

    Zips OUT_ROOT's contents directly (no extra nesting), so
    `unzip customer_support.zip -d deploy_artifacts` reproduces exactly this
    repo's deploy_artifacts/ layout — same workflow the guide describes for a
    single-subset zip, just covering all three folders in one file.
    """
    for stale in ZIP_ROOT.glob("*.zip"):
        stale.unlink()

    zip_path = ZIP_ROOT / combined_name
    shutil.make_archive(str(zip_path), "zip", str(OUT_ROOT))
    print(f"\nZipped all domains under {OUT_ROOT} -> {zip_path}.zip")
    return zip_path.with_suffix(".zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--zip-only", action="store_true",
        help="Skip regenerating any domain folders; just re-zip what's already under deploy_artifacts/.",
    )
    args = parser.parse_args()

    if not args.zip_only:
        if not args.dataset and not args.all:
            parser.error("pass --dataset <name>, --all, or --zip-only")
        targets = list(DATASETS) if args.all else [args.dataset]
        for dataset in targets:
            export_dataset(dataset)

    zip_all_domains()


if __name__ == "__main__":
    main()
