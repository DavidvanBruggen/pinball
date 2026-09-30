# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import torch, random, logging, itertools
from typing import Tuple, Optional
from datasets import load_dataset
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _ensure_eot_token(tokenizer, eot_token="<|eot|>") -> int:
    if tokenizer.eos_token is None:
        # add a fresh special token if the tokenizer has no eos
        tokenizer.add_special_tokens({"eos_token": eot_token})
    return tokenizer.eos_token_id

def _chunk_stream(iterator, chunk_len):
    """
    Yield fixed‑length chunks from an iterator of token lists.
    Remainder is carried to the next chunk (no data loss).
    """
    buf = []
    for item in iterator:
        buf.extend(item)
        while len(buf) >= chunk_len + 1:   # need +1 for y‑shift
            yield torch.tensor(buf[:chunk_len + 1])  # x‖y target built later
            buf = buf[chunk_len:]


# ---------------------------------------------------------------------------
# main factory
# ---------------------------------------------------------------------------

def create_karpathy_dataloaders(
    text_path: str,
    tokenizer,
    block_size: int = 8192,
    batch_size: int = 1,
    val_split: float = 0.01,
    seed: int = 42,
    stream_name: Optional[str] = None,
    tokens_per_epoch: int = 10_000_000_000   # 10 B default for FineWeb‑Edu
) -> Tuple[object, object]:
    """
    If `stream_name` is None  -> original behaviour (load whole text file).
    If `stream_name` is set   -> use HF streaming dataset with that subset name.
    """

    torch.manual_seed(seed)
    random.seed(seed)

    # -----------------------------------------------------------------------
    # 1. ensure <eot> token exists
    # -----------------------------------------------------------------------
    eot_id = _ensure_eot_token(tokenizer)

    # -----------------------------------------------------------------------
    # 2a. NON‑STREAMING mode  (original behaviour)
    # -----------------------------------------------------------------------
    if stream_name is None:
        text_path = Path(text_path)
        token_path = text_path.with_suffix(".pt")

        # load‑or‑tokenize whole file
        if token_path.exists():
            logger.info(f"Loading tokenised data from {token_path}")
            # mmap: map the cached tensor instead of copying it into RAM. PG19's cache is
            # 24.5 GB int64; with a plain load every process holds its own copy, so two runs
            # side by side need ~49 GB. Mapped, they share one page-cache copy and only each
            # batch's slices are materialised. Same tokens either way.
            tokens = torch.load(token_path, mmap=True)
        else:
            logger.info(f"Reading raw text from {text_path}")
            logger.info("Tokenising …")
            # Tokenise in bounded character chunks and concatenate tensors, instead of
            # reading the whole file into one string and tokenising in a single call.
            # The single-call path builds a ~100M+ element Python int list (several GB)
            # before the tensor, which OOMs low-RAM machines (e.g. Colab ~12 GB).
            chunk_chars = 4_000_000  # ~4 MB of text per tokenizer call
            parts = []
            buf, buf_len = [], 0

            def _flush(buf):
                if not buf:
                    return
                ids = tokenizer("".join(buf), add_special_tokens=False).input_ids
                if ids:
                    parts.append(torch.tensor(ids, dtype=torch.long))

            with open(text_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    buf.append(line)
                    buf_len += len(line)
                    if buf_len >= chunk_chars:
                        _flush(buf)
                        buf, buf_len = [], 0
            _flush(buf)
            parts.append(torch.tensor([tokenizer.eos_token_id], dtype=torch.long))
            tokens = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)
            del parts
            torch.save(tokens, token_path)
            logger.info(f"Saved {len(tokens):,} tokens → {token_path}")

        split = int(len(tokens) * (1 - val_split))
        train_tokens, val_tokens = tokens[:split], tokens[split:]

        logger.info(f"Train tokens = {len(train_tokens):,}")
        logger.info(f"Val tokens   = {len(val_tokens):,}")

        def _sample_batch(tok_tensor, device):
            ix = torch.randint(0, len(tok_tensor) - block_size - 1, (batch_size,))
            x = torch.stack([tok_tensor[i:i+block_size] for i in ix]).to(device)
            y = torch.stack([tok_tensor[i+1:i+block_size+1] for i in ix]).to(device)
            return {"input_ids": x,
                    "labels": y,
                    "attention_mask": torch.ones_like(x)}

        return (lambda device: _sample_batch(train_tokens, device),
                lambda device: _sample_batch(val_tokens,   device))

    # -----------------------------------------------------------------------
    # 2b. STREAMING mode (e.g. FineWeb‑Edu)
    # -----------------------------------------------------------------------
    logger.info(f"Opening streaming dataset = {stream_name}")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", stream_name, streaming=True)["train"]

    safe_val_split = max(float(val_split), 1e-12)
    val_take = int(max(10_000, 1.0 / safe_val_split))

    # Prefer deterministic non-overlapping train/val partitioning on stream.
    if hasattr(ds, "take") and hasattr(ds, "skip"):
        val_docs = ds.take(val_take)
        train_source = ds.skip(val_take)
    else:
        logger.warning(
            "Streaming dataset backend has no take/skip; using two independent streams with islice split."
        )
        ds_val = load_dataset("HuggingFaceFW/fineweb-edu", stream_name, streaming=True)["train"]
        ds_train = load_dataset("HuggingFaceFW/fineweb-edu", stream_name, streaming=True)["train"]
        val_docs = itertools.islice(ds_val, val_take)
        train_source = itertools.islice(ds_train, val_take, None)

    def _tokenise(doc):
        return tokenizer(doc["text"] + tokenizer.eos_token,
                        add_special_tokens=False).input_ids

    # training iterator (infinite / shuffled later)
    if hasattr(train_source, "shuffle"):
        train_iter = train_source.shuffle(buffer_size=10_000, seed=seed)
    else:
        logger.warning("Training stream does not support shuffle(); using sequential order.")
        train_iter = train_source
    train_chunks = _chunk_stream((_tokenise(d) for d in train_iter), block_size)

    # build finite list of validation chunks (continuous stream chunking, same semantics as train)
    raw_chunks = []
    val_doc_counter = {"count": 0}

    def _tokenise_val(doc):
        val_doc_counter["count"] += 1
        return _tokenise(doc)

    val_chunk_iter = _chunk_stream((_tokenise_val(doc) for doc in val_docs), block_size)
    for seq in itertools.islice(val_chunk_iter, 10_000):
        raw_chunks.append(seq)

    assert len(raw_chunks) >= batch_size, "Validation set too small"
    logger.info(
        f"Validation set = {len(raw_chunks)} chunks from {val_doc_counter['count']} docs "
        f"(chunk_len={block_size + 1}, eos_appended_per_doc=True)"
    )

    val_chunks = itertools.cycle(raw_chunks)   # never exhausts

    # running token counters so you can decide when "epoch" passed
    train_tok_counter = {"count": 0}
    val_tok_counter   = {"count": 0}

    def _next_batch(chunk_iter, counter_dict, device):
        xs, ys = [], []
        for _ in range(batch_size):
            seq = next(chunk_iter)  # length block_size+1
            xs.append(seq[:-1])
            ys.append(seq[1:])
            counter_dict["count"] += block_size
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": torch.ones_like(x),
            "tokens_seen": counter_dict["count"]   # optional logging
        }

    get_train_batch = lambda device: _next_batch(train_chunks, train_tok_counter, device)
    get_val_batch   = lambda device: _next_batch(val_chunks,   val_tok_counter,   device)

    # Convenience: expose a helper to check if an epoch-equivalent has passed
    def tokens_processed(split="train"):
        return (train_tok_counter if split=="train" else val_tok_counter)["count"]

    # Return both getters plus the counter helper
    get_train_batch.tokens_processed = tokens_processed
    get_val_batch.tokens_processed   = tokens_processed
    get_train_batch.TOKENS_PER_EPOCH = tokens_per_epoch  # constant stored

    # >>> attach attributes so callers can access them later
    get_train_batch.chunks  = train_chunks
    get_train_batch.counter = train_tok_counter
    # <<<

    return get_train_batch, get_val_batch
