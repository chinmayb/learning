"""Lab 3b — Chunk parsed docs into strategy files.
Run: uv run python src/lab3_chunk.py
Output: data/chunks/<strategy>.jsonl
Every chunk: {chunk_id, doc_id, source, page, heading, text, n_tokens, parent_id?}
"""
import matplotlib.pyplot as plt
from common import DATA, count_tokens, read_jsonl, tokenizer, write_jsonl


def make_chunk(rec, text, idx, **extra):
    return {"chunk_id": f"{rec['doc_id']}:{rec['page']}:{idx}",
            "doc_id": rec["doc_id"], "source": rec["source"], "page": rec["page"],
            "heading": rec["heading"], "domain": rec.get("domain", "misc"), "text": text,
            "n_tokens": count_tokens(text), **extra}


def fixed(recs, size=512, overlap=64):
    """A: sliding window of `size` tokens, stepping size-overlap."""
    tok, out = tokenizer(), []
    for rec in recs:
        ids = tok.encode(rec["text"], add_special_tokens=False)
        # TODO: slice ids into windows, tok.decode(window) -> text, append make_chunk(...)
        # Q: chunks don't cross page boundaries here. Good or bad?
    return out


def structural(recs, max_tokens=512, min_tokens=50):
    """B: split on blank lines (paragraphs), merge small paragraphs up to max_tokens."""
    out = []
    for rec in recs:
        paras = [p for p in rec["text"].split("\n\n") if p.strip()]
        # TODO: greedy merge paras until adding the next would exceed max_tokens
        #       a single para > max_tokens: split it with fixed() logic or keep it?
    return out


def small_to_big(recs, small=128):
    """C: small chunks for search; each points to its parent (the whole record)."""
    out = []
    for rec in recs:
        parent_id = f"{rec['doc_id']}:{rec['page']}"
        # TODO: create small chunks (reuse fixed logic, overlap 0?) with parent_id=parent_id
        #       Lab 6 will swap small chunk -> parent text before calling the LLM
    return out


STRATEGIES = {
    "A_fixed_512_64": lambda r: fixed(r, 512, 64),
    "A_fixed_256_32": lambda r: fixed(r, 256, 32),
    "B_struct_512":   lambda r: structural(r, 512),
    "C_small_128":    lambda r: small_to_big(r, 128),
}

if __name__ == "__main__":
    recs = read_jsonl(DATA / "parsed" / "docs.jsonl")
    for name, fn in STRATEGIES.items():
        chunks = fn(recs)
        write_jsonl(DATA / "chunks" / f"{name}.jsonl", chunks)
        sizes = [c["n_tokens"] for c in chunks]
        print(f"{name:16s} {len(chunks):5d} chunks  min={min(sizes, default=0)} max={max(sizes, default=0)}")
        plt.hist(sizes, bins=40, alpha=.4, label=name)
    plt.legend(); plt.xlabel("tokens/chunk"); plt.show()
