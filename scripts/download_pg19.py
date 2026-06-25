#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
"""Download PG19 (emozilla/pg19) into a single concatenated .txt for the Pinball
text pipeline (src/pinball/data/karpathy_loader.py reads one file and caches a .pt).

PG19 is long-form public-domain books (Project Gutenberg pre-1919) — a standard
long-range LM benchmark. Splits: train (~28,602 books), validation (50), test (100).

Streaming avoids materialising the whole Arrow dataset; --max-books caps it for
quick experiments. Books are separated by an <|endoftext|> marker so the model
sees document boundaries (the loader otherwise concatenates them seamlessly).

Examples:
  # small, fast: the 100-book test split (~good for first long-range runs)
  python scripts/download_pg19.py --split test --out data/pg19_test.txt

  # a 500-book slice of train for training experiments
  python scripts/download_pg19.py --split train --max-books 500 --out data/pg19_train500.txt

  # the full train split (large: ~11 GB text — be sure you want this)
  python scripts/download_pg19.py --split train --out data/pg19_train.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

# Matches gpt2/tiktoken end-of-text; karpathy_loader splits on lines, so a marker
# on its own line keeps book boundaries explicit in the token stream.
BOOK_SEP = "\n<|endoftext|>\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"],
                    help="PG19 split (default: test — small, good for a first run).")
    ap.add_argument("--out", default=None, help="Output .txt path (default: data/pg19_<split>.txt).")
    ap.add_argument("--max-books", type=int, default=None,
                    help="Stop after this many books (default: all of the split).")
    ap.add_argument("--no-streaming", action="store_true",
                    help="Download the full Arrow dataset instead of streaming.")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path("data") / f"pg19_{args.split}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("emozilla/pg19", split=args.split, streaming=not args.no_streaming)

    n_books = 0
    n_chars = 0
    with open(out, "w", encoding="utf-8") as fh:
        for ex in ds:
            text = ex.get("text") or ""
            if not text:
                continue
            if n_books > 0:
                fh.write(BOOK_SEP)
            fh.write(text)
            n_books += 1
            n_chars += len(text)
            if n_books % 50 == 0:
                print(f"  {n_books} books, {n_chars/1e6:.1f}M chars …", flush=True)
            if args.max_books is not None and n_books >= args.max_books:
                break

    print(f"Wrote {n_books} books, {n_chars/1e6:.1f}M chars -> {out}")
    print(f"Point your config at it:  text_file: ./{out.as_posix()}")


if __name__ == "__main__":
    main()
