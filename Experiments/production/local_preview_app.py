"""Local preview app for the exported Customer Support deploy_artifacts/ folders.

Pick a domain (techqa/emanual/delucionqa) and one of its 5 exported demo
questions, and see retrieval + generation run end-to-end locally — a stand-in
for what the shared Gradio app will do with your folder, so you can sanity-
check your export before sending the zip to the integration lead.

Retrieval mirrors validate_my_folder.py exactly (dense + BM25, fused with
RRF), then adds a cross-encoder rerank step before generation. Note: the
*real* shared app's reranker/judge model choices are the integration lead's
(per TEAMMATE_GUIDE.md, "reranking and the judge are shared... you do not
ship them") — this uses BAAI/bge-reranker-v2-m3 and Groq's
llama-3.3-70b-versatile as reasonable stand-ins (what we used throughout our
own project), not necessarily identical to the real shared app.

Usage:
    python3 production/local_preview_app.py
Then open the printed local URL (usually http://127.0.0.1:7860).
"""
import json
import os
import re
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
# final_submission/ is a sibling of this repo, not inside it.
DEPLOY_ROOT = REPO_ROOT.parent / "final_submission" / "deploy_artifacts"

TOP_K_WIDE = 30
TOP_K_CONTEXT = 5
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
GEN_MODEL = "llama-3.3-70b-versatile"

load_dotenv(REPO_ROOT / ".env", override=True)
_groq_key = (os.getenv("GROQ_API_KEY") or "").split(",")[0].strip()
groq_client = Groq(api_key=_groq_key) if _groq_key else None

_cross_encoder = None  # lazy-loaded on first retrieval call


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def rrf(rank_lists, k=60) -> list[int]:
    scores = {}
    for ranked in rank_lists:
        for r, i in enumerate(ranked):
            scores[i] = scores.get(i, 0) + 1 / (k + r + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


class Domain:
    def __init__(self, folder: Path):
        self.entry = json.loads((folder / "entry.json").read_text())
        corpus = json.loads((folder / "corpus.json").read_text())
        self.chunks = corpus["chunks"]
        self.chunk_parent = corpus["chunk_parent"]
        self.embed_model_name = corpus["embed_model"]
        self.index = faiss.read_index(str(folder / "index.faiss"))
        self.questions = json.loads((folder / "questions.json").read_text())
        self.prompts = json.loads((folder / "prompts.json").read_text())
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.bm25 = BM25Okapi([tokenize(c) for c in self.chunks])

    def retrieve(self, question: str, top_k: int = TOP_K_CONTEXT) -> list[int]:
        global _cross_encoder
        qe = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        _, dense_idx = self.index.search(qe, min(TOP_K_WIDE, len(self.chunks)))
        dense = [int(x) for x in dense_idx[0]]

        bm25_scores = self.bm25.get_scores(tokenize(question))
        sparse = [int(x) for x in np.argsort(bm25_scores)[::-1][:TOP_K_WIDE]]

        candidates = rrf([dense, sparse])[:TOP_K_WIDE]

        if _cross_encoder is None:
            _cross_encoder = CrossEncoder(RERANK_MODEL)
        pairs = [[question, self.chunks[i]] for i in candidates]
        rerank_scores = _cross_encoder.predict(pairs)
        reranked = [c for c, _ in sorted(zip(candidates, rerank_scores), key=lambda x: -x[1])]
        return reranked[:top_k]

    def generate(self, question: str, chunk_indices: list[int]) -> str:
        if groq_client is None:
            return "(No GROQ_API_KEY found in .env -- can't generate. Retrieval above still ran.)"
        prompt_name = self.prompts["default"]
        p = self.prompts["prompts"][prompt_name]
        context = "\n\n".join(
            f"[Document {i + 1}]\n{self.chunks[ci]}" for i, ci in enumerate(chunk_indices)
        )
        user_msg = f"Passages:\n{context}\n\nQuestion: {question}\n\n{p['instructions']}"
        try:
            resp = groq_client.chat.completions.create(
                model=GEN_MODEL,
                messages=[
                    {"role": "system", "content": p["system"]},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=750,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"(Generation failed: {e})"


def load_domains() -> dict[str, Domain]:
    print("Loading domains from", DEPLOY_ROOT)
    found = {}
    if DEPLOY_ROOT.exists():
        for folder in sorted(DEPLOY_ROOT.iterdir()):
            if (folder / "entry.json").exists():
                d = Domain(folder)
                found[d.entry["label"]] = d
                print(f"  loaded {d.entry['label']} ({len(d.chunks)} chunks, {len(d.questions)} questions)")
    if not found:
        raise SystemExit(f"No domain folders found under {DEPLOY_ROOT} -- run export_deploy_artifacts.py first.")
    return found


domains = load_domains()


def on_domain_change(label: str):
    d = domains[label]
    choices = [q["question"] for q in d.questions]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def on_run(label: str, question: str):
    d = domains[label]
    q_obj = next((q for q in d.questions if q["question"] == question), None)
    chunk_indices = d.retrieve(question)

    context_display = "\n\n---\n\n".join(
        f"(doc {d.chunk_parent[ci]}) {d.chunks[ci]}" for ci in chunk_indices
    )
    answer = d.generate(question, chunk_indices)

    gold = "(no gold data for this question)"
    if q_obj:
        gold = (
            f"gold_relevance={q_obj['gold_relevance']}  "
            f"gold_utilization={q_obj['gold_utilization']}  "
            f"gold_completeness={q_obj['gold_completeness']}  "
            f"gold_adherence={q_obj['gold_adherence']}\n"
            f"gold_parent_gids={q_obj['gold_parent_gids']}"
        )
    return answer, context_display, gold


with gr.Blocks(title="Customer Support -- local preview") as demo:
    gr.Markdown(
        "# Customer Support domain -- local preview\n"
        "Pick a domain and one of its 5 exported demo questions, and see retrieval + generation "
        "run end-to-end before sending your zip. Reranker/generator here are local stand-ins -- "
        "the real shared app's choices are the integration lead's."
    )
    domain_labels = list(domains.keys())
    first_domain = domains[domain_labels[0]]

    domain_dd = gr.Dropdown(choices=domain_labels, value=domain_labels[0], label="Domain")
    question_dd = gr.Dropdown(
        choices=[q["question"] for q in first_domain.questions],
        value=first_domain.questions[0]["question"],
        label="Demo question",
    )
    run_btn = gr.Button("Run", variant="primary")

    answer_out = gr.Textbox(label="Generated answer", lines=6)
    context_out = gr.Textbox(label="Retrieved context (reranked)", lines=12)
    gold_out = gr.Textbox(label="Gold reference (for your own comparison)", lines=3)

    domain_dd.change(on_domain_change, inputs=domain_dd, outputs=question_dd)
    run_btn.click(on_run, inputs=[domain_dd, question_dd], outputs=[answer_out, context_out, gold_out])


if __name__ == "__main__":
    demo.launch()
