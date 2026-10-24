"""Lab 6 — Generation. Answer only from retrieved chunks, with citations."""
import ollama
from common import DATA, read_jsonl

SYSTEM_V1 = """TODO: write your rules. Consider:
- answer ONLY from the numbered context
- cite like [1][3] after each claim
- exact phrase to use when the answer isn't in the context
- keep CLI commands / IPs / values verbatim
"""

PROMPTS = {"v1": SYSTEM_V1}  # add v2, v3 as you iterate — log which one each run used


def format_context(chunks):
    # TODO: return a string like:
    # [1] (dns_admin_guide.pdf p.12 | Forwarding)
    # <text>
    #
    # [2] ...
    ...


def expand_parents(chunks):
    """Strategy C only: swap each small chunk for its parent record text (dedupe parents)."""
    # TODO: load data/parsed/docs.jsonl, map f"{doc_id}:{page}" -> text, replace, dedupe
    ...


def answer(question, chunks, model="qwen3:8b", prompt="v1",
           temperature=0.0, num_ctx=8192, think=False):
    messages = [
        {"role": "system", "content": PROMPTS[prompt]},
        {"role": "user", "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"},
    ]
    # TODO: resp = ollama.chat(model=model, messages=messages, think=think,
    #                          options={"temperature": temperature, "num_ctx": num_ctx})
    # return {"answer": ..., "prompt_tokens": resp["prompt_eval_count"],
    #         "gen_tokens": resp["eval_count"], "seconds": resp["total_duration"] / 1e9}
    # Q: if prompt_tokens is suspiciously close to num_ctx — what happened?
    ...


if __name__ == "__main__":
    from retrieve import Retriever
    r = Retriever("A_fixed_512_64", "nomic-embed-text")
    q = "TODO: a question from your golden set"
    hits = r.search(q, mode="dense", k=5)
    print(answer(q, hits))
