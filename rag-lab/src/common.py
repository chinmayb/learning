"""Shared plumbing. DONE for you — read it, don't edit unless you want to."""
import json
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
EVAL = ROOT / "eval"


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


@lru_cache
def tokenizer():
    """Qwen3 tokenizer = same family as your generator. Downloads ~10MB once."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


class Timer:
    """with Timer() as t: ...;  t.ms"""
    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self._t) * 1000
