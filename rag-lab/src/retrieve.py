"""Lab 5 — Index + retrieval. Brute-force numpy (fine for <50 files).
Build: uv run python src/retrieve.py A_fixed_512_64 nomic-embed-text
"""
import json
import re
import sys
from functools import lru_cache

import numpy as np
import ollama
from rank_bm25 import BM25Okapi
from common import DATA, read_jsonl

# Prefix conventions per model. Verify these against each model's docs.
PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "bge-m3": ("", ""),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}


def index_name(chunks, embedder, use_prefix, add_heading):
    return f"{chunks}__{embedder}__p{int(use_prefix)}__h{int(add_heading)}"


def embed(texts, model, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        # TODO: ollama.embed on texts[i:i+batch], extend out
        ...
    v = np.array(out, dtype=np.float32)
    # TODO: L2-normalize rows so dot product == cosine
    return v


def build(chunks, embedder, use_prefix=True, add_heading=False):
    rows = read_jsonl(DATA / "chunks" / f"{chunks}.jsonl")
    d_prefix = PREFIXES[embedder][1] if use_prefix else ""
    # TODO (B+ experiment): if add_heading, prepend f"{source} > {heading}\n" to the text
    texts = [d_prefix + r["text"] for r in rows]
    vecs = embed(texts, embedder)
    d = DATA / "index" / index_name(chunks, embedder, use_prefix, add_heading)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "vecs.npy", vecs)
    (d / "info.json").write_text(json.dumps(
        {"chunks": chunks, "embedder": embedder, "use_prefix": use_prefix, "add_heading": add_heading}))
    print(f"indexed {len(rows)} chunks -> {d.name}")


class Retriever:
    def __init__(self, chunks, embedder, use_prefix=True, add_heading=False):
        d = DATA / "index" / index_name(chunks, embedder, use_prefix, add_heading)
        self.vecs = np.load(d / "vecs.npy")
        self.rows = read_jsonl(DATA / "chunks" / f"{chunks}.jsonl")
        self.embedder = embedder
        self.q_prefix = PREFIXES[embedder][0] if use_prefix else ""
        self.bm25 = BM25Okapi([bm25_tokenize(r["text"]) for r in self.rows])

    def dense(self, query, k=50):
        # TODO: embed query (with q_prefix), scores = self.vecs @ q, return top-k indices (best first)
        ...

    def sparse(self, query, k=50):
        # TODO: self.bm25.get_scores(bm25_tokenize(query)) -> top-k indices
        ...

    def hybrid(self, query, k=50, rrf_k=60):
        # TODO: Reciprocal Rank Fusion of dense() and sparse():
        #       score[i] += 1 / (rrf_k + rank)   (rank starts at 1)
        ...

    def search(self, query, mode="dense", k=5, rerank=False, pool=50):
        idx = {"dense": self.dense, "sparse": self.sparse, "hybrid": self.hybrid}[mode](query, pool)
        if rerank:
            idx = rerank_idx(query, idx, self.rows)
        return [self.rows[i] for i in idx[:k]]


def bm25_tokenize(text):
    # TODO: this is a Lab 1 decision. Naive version below SPLITS '10.0.0.5', 'dns-forwarder',
    #       'corp.example.com' into pieces. Is that what you want? Try alternatives, measure.
    return re.findall(r"\w+", text.lower())


@lru_cache
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device="mps")


def rerank_idx(query, idx, rows):
    # TODO: scores = _reranker().predict([(query, rows[i]["text"]) for i in idx])
    #       return idx sorted by score desc
    ...


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], use_prefix=True, add_heading=False)
