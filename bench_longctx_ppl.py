# SPDX-License-Identifier: GPL-3.0-or-later
"""Long-context diagnostic perplexity: does Pinball's hierarchy actually use evidence
*outside the local window*? Average PPL hides this; bucketed PPL exposes it.

Per-token NLL is stratified by:
  (1) in-context recurrence distance   -- distance to the same token's previous occurrence
                                          within the model's visible left context
  (2) token rarity                     -- corpus frequency of the target token
  (3) rare x far cross-bucket          -- the headline: rare token last seen >window back
  (4) boundary proximity               -- first tokens after a newline / paragraph break

With --compare, the SAME weights are evaluated full vs hierarchy-ablated (ablate_levels=
[1,2,3] => windowed-only: every cross-level/hierarchy edge cut, only L0 local attention
remains) and reports ΔNLL per bucket. A hierarchy that helps global coherence should lower
NLL most on the rare+far and boundary buckets, even when total PPL barely moves.

Run on the free 4090 (torch ordering):
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python \
        bench_longctx_ppl.py --checkpoint pinball/sepqkv_gpt2_128bin/checkpoints/pinball_best.pt \
        --data data/pg19_test.txt --compare
"""
import sys, math, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

# In-context recurrence-distance buckets (upper-exclusive edges). The local window is 128,
# so the diagnostic split is "<128 (in window)" vs ">=128 (needs hierarchy)".
DIST_EDGES = [1, 16, 64, 128, 512, 2048, 8192, 1 << 30]
DIST_LABELS = ["never", "1-15", "16-63", "64-127", "128-511", "512-2k", "2k-8k", ">8k"]
# Rarity by corpus token count (upper-exclusive). Tunable via --rare-edges.
RARE_EDGES = [3, 11, 101, 1001, 1 << 30]
RARE_LABELS = ["freq1-2", "freq3-10", "freq11-100", "freq101-1k", "freq>1k"]


def build(cfg_path, ckpt, ablate, device, seq_len):
    cfg = PinballConfig.from_yaml(cfg_path)
    cfg.block_size = seq_len
    cfg.batch_size = 1
    if ablate:
        cfg.ablate_levels = [1, 2, 3]
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                    tie_weights=True, max_seq_len=seq_len).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = sd.get("model_state_dict", sd)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)} (non-strict)")
    m.eval()
    return m, tok


@torch.no_grad()
def logits_for(m, ids, device):
    # Mirror the trainer's AR next-token eval contract (trainer.validate, objective_mode=="ar"):
    # input is the unmasked tokens; reveal_target_ids is the left-shifted sequence and
    # reveal_mask marks every position but the last as a predict/query position. The model
    # needs these to emit next-token logits; a bare m(ids) yields garbage.
    reveal_target = ids.clone()
    reveal_mask = torch.zeros_like(ids, dtype=torch.bool)
    if ids.size(1) > 1:
        reveal_target[:, :-1] = ids[:, 1:]
        reveal_mask[:, :-1] = True
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else __import__("contextlib").nullcontext()
    with ctx:
        out = m(ids, attention_mask=None, reveal_target_ids=reveal_target, reveal_mask=reveal_mask)
    if isinstance(out, dict):
        out = out.get("logits", out)
    f = out[0] if isinstance(out, (tuple, list)) else out
    if f.dim() == 3 and f.size(-1) == int(getattr(m, "hidden_dim", -1)):
        proj = getattr(m, "output_projection", None) or getattr(getattr(m, "module", m), "output_projection", None)
        f = proj(f)
    return f.float()


def bucket_idx(edges, x):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges) - 1


class Acc:
    """Per-bucket NLL accumulators across several stratifications."""
    def __init__(self):
        self.dist = [[0.0, 0] for _ in DIST_LABELS]
        self.rare = [[0.0, 0] for _ in RARE_LABELS]
        # rare x far: rows=rarity, cols=(in-window <128, far >=128)
        self.cross = [[[0.0, 0], [0.0, 0]] for _ in RARE_LABELS]
        self.boundary = [0.0, 0]      # first K after a newline
        self.nonbound = [0.0, 0]
        self.total = [0.0, 0]

    def add(self, nll, dist, rarebkt, is_boundary):
        self.total[0] += nll; self.total[1] += 1
        di = bucket_idx(DIST_EDGES, dist) if dist is not None else 0
        self.dist[di][0] += nll; self.dist[di][1] += 1
        self.rare[rarebkt][0] += nll; self.rare[rarebkt][1] += 1
        far = 1 if (dist is not None and dist >= 128) else 0
        self.cross[rarebkt][far][0] += nll; self.cross[rarebkt][far][1] += 1
        if is_boundary:
            self.boundary[0] += nll; self.boundary[1] += 1
        else:
            self.nonbound[0] += nll; self.nonbound[1] += 1


def ppl(pair):
    s, n = pair
    return math.exp(s / n) if n > 0 else float("nan")


def run_model(m, tok, token_ids, freq, device, seq_len, stride, boundary_k, max_tokens, nl_ids):
    acc = Acc()
    N = min(len(token_ids), max_tokens)
    ids_all = token_ids[:N]
    pos = 0
    while pos < N - 1:
        win = ids_all[pos: pos + seq_len]
        if len(win) < 2:
            break
        ids = torch.tensor(win, device=device, dtype=torch.long).unsqueeze(0)
        logits = logits_for(m, ids, device)          # [1, T, V]
        T = ids.size(1)
        lp = F.log_softmax(logits[0, :-1], dim=-1)    # predict token t+1 from <=t
        tgt = ids[0, 1:]
        nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)   # [T-1]

        # only score tokens whose full left context is inside this window: the new `stride`
        # tail (except the first window, which scores everything it can).
        score_start = 0 if pos == 0 else max(0, T - 1 - stride)
        # in-context recurrence distance: last occurrence within THIS window before the target
        last_seen = {}
        last_para = None   # most recent paragraph-break (blank line) position in this window
        for j in range(T):
            tid = win[j]
            if tid in nl_ids:
                last_para = j
            tpos = j - 1   # this is the position predicting token at index j (target = win[j])
            if j >= 1 and j - 1 >= score_start:
                prev = last_seen.get(tid, None)
                dist = (j - prev) if prev is not None else None
                rb = bucket_idx(RARE_EDGES, freq.get(tid, 0))
                is_b = False
                if boundary_k > 0 and last_para is not None:
                    # boundary = within boundary_k tokens AFTER a paragraph break (blank line)
                    is_b = 0 < (j - last_para) <= boundary_k
                acc.add(float(nll[j - 1]), dist, rb, is_b)
            last_seen[tid] = j
        if pos == 0:
            pos += seq_len
        else:
            pos += stride
    return acc


def print_table(name, acc):
    print(f"\n===== {name} =====")
    print(f"  TOTAL ppl = {ppl(acc.total):8.3f}   (tokens scored = {acc.total[1]:,})")
    print("  -- by in-context recurrence distance --")
    for lbl, pr in zip(DIST_LABELS, acc.dist):
        if pr[1]:
            print(f"    {lbl:>8}: ppl {ppl(pr):9.3f}  n={pr[1]:>9,}")
    print("  -- by target rarity --")
    for lbl, pr in zip(RARE_LABELS, acc.rare):
        if pr[1]:
            print(f"    {lbl:>11}: ppl {ppl(pr):9.3f}  n={pr[1]:>9,}")
    print("  -- rare x far (prev occ >=128 back) --")
    for lbl, row in zip(RARE_LABELS, acc.cross):
        inw, far = row
        sw = f"in<128 ppl {ppl(inw):8.2f} (n={inw[1]:,})" if inw[1] else "in<128 -"
        sf = f"far>=128 ppl {ppl(far):8.2f} (n={far[1]:,})" if far[1] else "far>=128 -"
        print(f"    {lbl:>11}: {sw:>34}   {sf}")
    print(f"  -- boundary (first {acc.boundary[1] and 'K'} after newline) --")
    if acc.boundary[1]:
        print(f"    after-boundary ppl {ppl(acc.boundary):8.3f}  n={acc.boundary[1]:,}")
    if acc.nonbound[1]:
        print(f"    non-boundary   ppl {ppl(acc.nonbound):8.3f}  n={acc.nonbound[1]:,}")


def delta_table(full, abl):
    """Headline: ΔNLL = full - ablated (negative => hierarchy LOWERS loss / helps)."""
    def dnll(a, b):
        if a[1] == 0 or b[1] == 0:
            return None
        return a[0] / a[1] - b[0] / b[1]
    print("\n===== Δ mean-NLL (FULL - WINDOWED-ONLY); negative => hierarchy HELPS =====")
    print("  -- by in-context recurrence distance --")
    for lbl, fa, ab in zip(DIST_LABELS, full.dist, abl.dist):
        d = dnll(fa, ab)
        if d is not None:
            flag = "  <-- hierarchy helps" if d < -0.005 else ("  (worse)" if d > 0.005 else "")
            print(f"    {lbl:>8}: ΔNLL {d:+.4f}   n={fa[1]:>9,}{flag}")
    print("  -- rare x far --")
    for lbl, frow, arow in zip(RARE_LABELS, full.cross, abl.cross):
        d = dnll(frow[1], arow[1])
        if d is not None:
            flag = "  <-- hierarchy helps" if d < -0.005 else ""
            print(f"    {lbl:>11} far>=128: ΔNLL {d:+.4f}   n={frow[1][1]:>8,}{flag}")
    db = dnll(full.boundary, abl.boundary)
    if db is not None:
        print(f"  -- boundary: ΔNLL {db:+.4f}   n={full.boundary[1]:,}")
    dt = dnll(full.total, abl.total)
    if dt is not None:
        print(f"  -- TOTAL: ΔNLL {dt:+.4f}  (full ppl {ppl(full.total):.3f} vs windowed {ppl(abl.total):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="pinball/sepqkv_gpt2_128bin/checkpoints/pinball_best.pt")
    ap.add_argument("--config", default="configs/pinball_wikitext.yaml")
    ap.add_argument("--data", default="data/pg19_test.txt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=None, help="default seq_len//2")
    ap.add_argument("--max-tokens", type=int, default=200_000)
    ap.add_argument("--boundary-k", type=int, default=32)
    ap.add_argument("--compare", action="store_true", help="also run hierarchy-ablated (windowed-only) and show ΔNLL")
    args = ap.parse_args()
    stride = args.stride or (args.seq_len // 2)
    dev = args.device

    tok0 = AutoTokenizer.from_pretrained("gpt2")
    # Read only enough characters to cover max_tokens (~8 chars/token, generous slack) so we
    # don't tokenize a multi-hundred-MB corpus in full.
    char_budget = max(args.max_tokens * 8, 100_000) + 200_000
    with open(args.data, errors="ignore") as fh:
        text = fh.read(char_budget)
    print(f"Tokenizing {args.data} (read {len(text):,} chars) ...")
    token_ids = tok0.encode(text)
    N = min(len(token_ids), args.max_tokens)
    print(f"  {len(token_ids):,} tokens; scoring first {N:,}")
    # corpus frequency over the scored stream
    freq = {}
    for t in token_ids[:N]:
        freq[t] = freq.get(t, 0) + 1
    # paragraph-break tokens (blank line) for boundary detection -- a single "\n" is too
    # frequent in prose to be a boundary signal, so require a double newline.
    nl_ids = {i for i in set(token_ids[:N]) if "\n\n" in tok0.decode([i])}

    print(f"\nLoading FULL model from {args.checkpoint}")
    m_full, _ = build(args.config, args.checkpoint, ablate=False, device=dev, seq_len=args.seq_len)
    full = run_model(m_full, tok0, token_ids, freq, dev, args.seq_len, stride, args.boundary_k, N, nl_ids)
    print_table("FULL Pinball (hierarchy intact)", full)
    del m_full
    torch.cuda.empty_cache()

    if args.compare:
        print(f"\nLoading WINDOWED-ONLY model (ablate_levels=[1,2,3]) from {args.checkpoint}")
        m_abl, _ = build(args.config, args.checkpoint, ablate=True, device=dev, seq_len=args.seq_len)
        abl = run_model(m_abl, tok0, token_ids, freq, dev, args.seq_len, stride, args.boundary_k, N, nl_ids)
        print_table("WINDOWED-ONLY (hierarchy ablated)", abl)
        delta_table(full, abl)


if __name__ == "__main__":
    main()
