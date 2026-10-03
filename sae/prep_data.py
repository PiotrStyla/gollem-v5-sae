"""Parquet (wikitext) -> tokens.bin uint16 przez tokenizer.json GoLLeM-v5.

Padding-vocab: token zwraca id < 12288; uint16 wystarcza (vocab 12288).
"""
import argparse
import os

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = bez limitu")
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    chunks = []
    total = 0
    for path in a.parquet:
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        for enc in tok.encode_batch(texts, add_special_tokens=False):
            ids = np.asarray(enc.ids, dtype=np.uint16)
            chunks.append(ids)
            total += len(ids)
        print(f"{os.path.basename(path)}: {total:,} tokenow", flush=True)
        if a.max_tokens and total >= a.max_tokens:
            break

    out = np.concatenate(chunks)
    if a.max_tokens:
        out = out[: a.max_tokens]
    assert out.max() < 65536
    out.tofile(a.out)
    print(f"ZAPISANO {a.out}: {len(out):,} tokenow, max_id={int(out.max())}")


if __name__ == "__main__":
    main()
