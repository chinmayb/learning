"""Lab 2 — Embeddings.  Run: uv run python src/lab2_embed.py"""
import random

import matplotlib.pyplot as plt
import numpy as np
import ollama
from common import Timer

MODELS = ["nomic-embed-text", "bge-m3", "mxbai-embed-large"]

# (label, text_a, text_b). Pre-filled with DNS cases. ADD 5+ from your docs.
PAIRS = [
    ("paraphrase", "An A record maps a hostname to an IPv4 address.",
                   "A hostname resolves to an IPv4 address through an A record."),
    ("negation",   "DNSSEC validation is enabled by default.",
                   "DNSSEC validation is not enabled by default."),
    ("allow/deny", "Allow zone transfers to 10.0.0.5.",
                   "Deny zone transfers to 10.0.0.5."),
    ("keyword_trap", "Forward zones send queries to upstream servers.",
                     "Forward the email to the zone manager for review."),
    ("q_vs_answer", "What port does DNS use?",
                    "DNS servers listen on UDP and TCP port 53."),
    ("acronym",    "Configure the RPZ feed.",
                   "Configure the response policy zone feed."),
    ("unrelated",  "Set the TTL of the SOA record to 3600.",
                   "The cafeteria opens at 8am on weekdays."),
]


def embed(texts, model, prefix=""):
    # TODO: call ollama.embed(model=model, input=[prefix + t for t in texts])
    #       return np.array(resp["embeddings"], dtype=np.float32)
    ...


def cosine(a, b):
    # TODO: normalize rows of a and b, return a @ b.T
    ...


def pair_scores(model, q_prefix="", d_prefix=""):
    a = embed([p[1] for p in PAIRS], model, q_prefix)
    b = embed([p[2] for p in PAIRS], model, d_prefix)
    sims = np.diag(cosine(a, b))
    for (label, *_), s in zip(PAIRS, sims):
        print(f"  {label:14s} {s:.3f}")


def random_baseline(model, sentences, n=50):
    """Experiment 2: what does 'unrelated' score for this model?"""
    pairs = [random.sample(sentences, 2) for _ in range(n)]
    # TODO: embed both sides, collect diagonal cosine scores, return np.array
    ...


def speed(model, n=100):
    texts = ["DNS resolver forwards queries upstream."] * n
    with Timer() as t:
        v = embed(texts, model)
    print(f"  dim={v.shape[1]}  {t.ms / n:.1f} ms/text")


if __name__ == "__main__":
    for m in MODELS:
        print(f"\n== {m}")
        speed(m)
        pair_scores(m)

    # Experiment 3 — nomic prefixes
    print("\n== nomic WITH prefixes")
    pair_scores("nomic-embed-text", "search_query: ", "search_document: ")

    # Experiment 2 — histograms. TODO: load ~200 sentences from your parsed docs
    # sentences = [...]
    # for m in MODELS:
    #     plt.hist(random_baseline(m, sentences), bins=30, alpha=.5, label=m)
    # plt.legend(); plt.title("cosine of random pairs"); plt.show()
