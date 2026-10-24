"""Lab 1 — Tokenization.  Run: uv run python src/lab1_tokens.py"""
import ollama
import pandas as pd
import tiktoken
from common import tokenizer

GEN_MODEL = "qwen3:8b"

# Pre-filled with DNS/networking traps. ADD 3 samples copied from YOUR docs.
SAMPLES = {
    "english": "The resolver forwards queries it cannot answer to the configured upstream servers.",
    "ipv4_cidr": "Allow recursion only for 10.120.0.0/16 and 192.168.1.0/24.",
    "ipv6": "Listen on 2001:db8:85a3::8a2e:370:7334 port 53.",
    "fqdn": "_ldap._tcp.dc._msdcs.corp.example.com. 3600 IN SRV 0 100 389 dc01.corp.example.com.",
    "record_types": "A AAAA CNAME MX NS PTR SOA SRV TXT DS DNSKEY RRSIG NSEC3",
    "cli": "dig @8.8.8.8 example.com +dnssec +short -t AAAA",
    "mac_hex": "MAC 00:1A:2B:3C:4D:5E, serial 0x7F3A9C",
    # "my_paragraph": "...",   # TODO
    # "my_table_row": "...",   # TODO (as extracted from a PDF — ugly is fine)
    # "my_jargon": "...",      # TODO (product names, acronyms)
}


def gpt_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    # TODO: return number of tokens
    ...


def qwen_tokens(text):
    tok = tokenizer()
    # TODO: return (count, pieces)
    # hint: tok.encode(text, add_special_tokens=False) -> ids
    #       tok.convert_ids_to_tokens(ids) -> pieces
    ...


def ollama_tokens(text):
    # TODO: call ollama.generate(model=GEN_MODEL, prompt=text, options={"num_predict": 1})
    #       return resp["prompt_eval_count"]
    # Q: does this match qwen_tokens? If not — what's being added?
    ...


def table():
    rows = []
    for name, text in SAMPLES.items():
        q_count, pieces = qwen_tokens(text)
        rows.append({
            "sample": name,
            "chars": len(text),
            "gpt": gpt_tokens(text),
            "qwen": q_count,
            "ollama": ollama_tokens(text),
            "chars/tok": round(len(text) / q_count, 2),
        })
        print(f"\n{name}: {pieces}")
    print("\n", pd.DataFrame(rows).to_string(index=False))


def truncation_trap(long_text: str):
    """Experiment 3. long_text should be ~12k tokens (concatenate a few docs)."""
    question = "\n\nQuestion: What was the very first sentence of this text?"
    prompt = long_text + question
    for opts in [{}, {"num_ctx": 16384}]:
        # TODO: call ollama.generate with options=opts
        #       print opts, prompt_eval_count, and the answer
        # Q: which one actually "saw" the question?
        ...


if __name__ == "__main__":
    table()
    # truncation_trap(open("data/parsed/some_long_doc.txt").read())  # after Lab 3
