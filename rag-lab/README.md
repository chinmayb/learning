# rag-lab starter

```bash
uv init --no-workspace . 2>/dev/null || true
uv add ollama tiktoken transformers pymupdf python-docx numpy pandas \
       rank-bm25 sentence-transformers matplotlib pyyaml
# run everything from the project root, e.g.:
uv run python src/lab1_tokens.py
```

| Lab | File | Status |
|---|---|---|
| 1 | src/lab1_tokens.py | TODOs |
| 2 | src/lab2_embed.py | TODOs |
| 3 | src/lab3_parse.py, src/lab3_chunk.py | TODOs |
| 4 | eval/golden.jsonl | template — write 40+ |
| 5 | src/retrieve.py + src/run_eval.py --retrieval-only | TODOs |
| 6 | src/answer.py | TODOs |
| 7 | src/run_eval.py (judge) + config.yaml | TODOs |

Plumbing (I/O, loops, logging, tables) is done. Core logic is yours.
