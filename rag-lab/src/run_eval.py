"""Labs 5 + 7 — one run = one config = one row in runs/results.csv
Retrieval only:   uv run python src/run_eval.py config.yaml --retrieval-only
Full (w/ judge):  uv run python src/run_eval.py config.yaml
"""
import argparse
import json
import re
from datetime import datetime

import ollama
import pandas as pd
import yaml
from answer import answer
from common import EVAL, ROOT, RUNS, Timer, read_jsonl, write_jsonl
from retrieve import Retriever


# ---------- retrieval metrics ----------
def is_relevant(chunk, gold):
    """A chunk is relevant if source matches and (no pages given, or page overlaps)."""
    # TODO: compare chunk["source"] with gold["source"]; check chunk["page"] in gold["pages"]
    # Q: a chunk that straddles pages 12-13 but stores page=12 — does this miss it?
    ...


def first_hit_rank(chunks, gold):
    # TODO: return 1-based rank of first relevant chunk, or None
    ...


def retrieval_metrics(ranks, ks=(1, 5, 10)):
    m = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks) for k in ks}
    # TODO: m["mrr"] = mean of 1/rank (0 when None)
    return m


# ---------- answer metrics (LLM judge) ----------
JUDGE_PROMPT = """TODO: write a judge prompt that returns ONLY JSON:
{"correctness": 0|1|2, "faithful": 0|1, "citations_valid": 0|1, "reason": "..."}
Give it: question, reference answer, model answer, the retrieved context.
Define each score precisely — vague rubrics = noisy judge.
"""


def judge(q, gold, ans, chunks, model="qwen3:14b"):
    # TODO: call ollama.chat with JUDGE_PROMPT + inputs, temperature 0, format="json"
    #       parse with json.loads; on failure return None and count failures
    ...


# ---------- runner ----------
def main(cfg_path, retrieval_only):
    cfg = yaml.safe_load(open(cfg_path))
    run_id = datetime.now().strftime("%m%d_%H%M%S")
    golden = read_jsonl(EVAL / "golden.jsonl")
    r = Retriever(cfg["chunks"], cfg["embedder"], cfg["use_prefix"], cfg["add_heading"])

    per_q, ranks = [], []
    for g in golden:
        with Timer() as t:
            hits = r.search(g["question"], cfg["mode"], cfg["k"], cfg["rerank"])
        rank = first_hit_rank(hits, g) if g["type"] != "none" else None
        if g["type"] != "none":
            ranks.append(rank)
        row = {"id": g["id"], "type": g["type"], "rank": rank, "retrieve_ms": t.ms,
               "chunks": [h["chunk_id"] for h in hits]}
        if not retrieval_only:
            a = answer(g["question"], hits, cfg["model"], cfg["prompt"],
                       cfg["temperature"], cfg["num_ctx"], cfg["think"])
            row |= a
            row |= judge(g["question"], g, a["answer"], hits) or {}
        per_q.append(row)

    write_jsonl(RUNS / run_id / "per_question.jsonl", per_q)
    df = pd.DataFrame(per_q)
    summary = {"run_id": run_id, **cfg, **retrieval_metrics(ranks),
               "retrieve_ms": round(df["retrieve_ms"].mean(), 1)}
    if not retrieval_only:
        for col in ["correctness", "faithful", "citations_valid", "seconds"]:
            if col in df:
                summary[col] = round(df[col].mean(), 3)

    print(json.dumps(summary, indent=2, default=str))
    print("\nhit@5 by type:")
    print(df[df.type != "none"].groupby("type")["rank"]
          .apply(lambda s: (s.notna() & (s <= 5)).mean()).round(2))

    out = RUNS / "results.csv"
    pd.DataFrame([summary]).to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"\nlogged -> {out}   details -> runs/{run_id}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--retrieval-only", action="store_true")
    a = ap.parse_args()
    main(ROOT / a.config, a.retrieval_only)
