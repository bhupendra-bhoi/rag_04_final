"""Production RAG pipeline runner.

Loads one of the chosen production configs (production/configs/*.yaml — see
each file's header for why it was picked), builds the retrieval index once
from the corresponding RAGBench subset, and answers questions against it:
either a single --question, or an interactive loop for several.

To point this at your own documents instead of RAGBench, replace
load_documents() with a loader over your own corpus (data_sources/loaders/
has the existing HuggingFace loader as a reference implementation) —
everything downstream (chunking, embedding, retrieval, generation) is
unchanged, since it's driven entirely by the same production config.

Usage:
    python3 production/run_pipeline.py --dataset emanual --question "How do I enable Ambient mode?"
    python3 production/run_pipeline.py --dataset techqa --interactive
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rag.config.loader import ConfigLoader
from rag.config.enums import Mode
from rag.pipeline.rag_pipeline import RAGPipeline
from data_sources.processors import DataProcessor
from data_sources.loaders.base import DatasetLoadingConfig
from data_sources.loaders.registry import loader_registry
from data_sources.loaders.huggingface_loader import HuggingFaceLoader  # noqa: F401 (registers loader)
from parsers import parser_registry

# Full RAGBench test-split sizes for each subset.
DATASET_SPLIT_SIZES = {
    "techqa": 314,
    "emanual": 132,
    "delucionqa": 184,
}


def load_documents(dataset: str, limit: int):
    loader = loader_registry.create(
        "huggingface",
        dataset_name="galileo-ai/ragbench",
        subset=dataset,
        split="test",
        config=DatasetLoadingConfig(limit=limit, use_cache=True),
    )
    raw_data = loader.load()
    parser = parser_registry.create("noop")
    return DataProcessor(parser_strategy=parser).process_dataset(raw_data)


def build_pipeline(dataset: str, limit: int) -> RAGPipeline:
    config_path = REPO_ROOT / "production" / "configs" / f"{dataset}.yaml"
    config = ConfigLoader.load(config_path)
    config.mode = Mode.PROD  # quiet logging (warnings/errors only) for real use

    print(f"Loading {dataset} corpus (up to {limit} rows)...")
    documents = load_documents(dataset, limit)

    pipeline = RAGPipeline(config)
    print(f"Building index ({len(documents)} documents)...")
    pipeline.build_index(documents)
    print("Index ready.\n")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the production RAG pipeline.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPLIT_SIZES))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Rows to index (default: the full test split for --dataset).",
    )
    parser.add_argument("--question", help="Ask a single question and exit.")
    parser.add_argument("--interactive", action="store_true", help="Drop into a REPL loop.")
    args = parser.parse_args()

    limit = args.limit or DATASET_SPLIT_SIZES[args.dataset]
    pipeline = build_pipeline(args.dataset, limit)

    if args.question:
        result = pipeline.query(args.question)
        print(f"Q: {args.question}\nA: {result.answer}")
        return

    print("Interactive mode — type a question, or 'quit'/'exit' to stop.")
    idx = 0
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit"):
            break
        result = pipeline.query(question, query_index=idx)
        idx += 1
        print(f"\n{result.answer}")


if __name__ == "__main__":
    main()
