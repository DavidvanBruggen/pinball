"""SLOT NOMINATOR ablation (no "off" mode: a zero gate is ill-defined with the key boost, which adds z*u to the slot keys); random / recent = replacement picks from each chunk's allowed set, gate recomputed from w. Adapted from the xq descent ablation on a trained checkpoint, bucketed by recurrence distance.

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/ablate_slots.py CONFIG CKPT [NSEQ] [B] [OUT]
Conditions (same weights, same PG19 val sequences, compiled eval path):
  full    descent as trained
  off     no sparse read at all (zero-ablation)
  random  same per-level candidate counts, uniformly random CLOSED nodes (causal, L0 >= 128 back)
  recent  same counts, the most recent closed nodes per level (query-position only, no content)
random/recent are the controls: if full beats them, the per-query CONTENT selection matters,
not merely having an extra read of the hierarchy."""
import sys, math, json, logging, random
import torch, torch.nn.functional as F
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs

CFG, CKPT = sys.argv[1], sys.argv[2]
NSEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 64
B = int(sys.argv[4]) if len(sys.argv) > 4 else 4
OUT = sys.argv[5] if len(sys.argv) > 5 else "xq_ablate.json"
LABELS = ["never", "<128", "128-511", "512-2k", "2k-8k", ">8k"]
EDGES = [128, 512, 2048, 8192]

cfg = PinballConfig.from_yaml(CFG)
L = int(cfg.block_size)
inp = resolve_model_inputs(cfg, block_size=L)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=L).cuda()
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
mi, un = m.load_state_dict(sd.get("model_state_dict", sd), strict=False)
print(f"[load] missing {len(mi)} unexpected {len(un)}", mi[:5], un[:5], flush=True)
del sd
m.eval()

# PG19 val = last 1% of the token cache (the trainer's split); NSEQ evenly spaced windows
tok = torch.load("data/pg19_train.pt", mmap=True, weights_only=False)
split = int(len(tok) * (1 - float(cfg.val_split)))
val = tok[split:]
starts = np.linspace(0, len(val) - L - 1, NSEQ).astype(np.int64)
seqs = torch.stack([val[s:s + L].clone() for s in starts.tolist()])

def buckets(ids):
    """bucket index per predicted target position j-1 (target ids[j]), as the trainer does"""
    out = np.zeros(ids.size(0) - 1, dtype=np.int64)
    last = {}
    for j, t in enumerate(ids.tolist()):
        if j >= 1:
            p = last.get(t)
            if p is None:
                out[j - 1] = 0
            else:
                d = j - p
                out[j - 1] = 1 + sum(d >= e for e in EDGES)
        last[t] = j
    return out
bk = np.stack([buckets(s) for s in seqs])                       # [NSEQ, L-1]

MODE = {"v": "full"}
STATS = {}
import pinball.model.layers.hierarchical_message_passing as H
_orig_sel = H._nom_stream_select_fn
def _sel(w, u, lvl_packed, a, cnt, ok, order, a_sorted, groups, nch, K, tau, prefix, gate_bias,
         want_w=False):
    rows, g, stats = _orig_sel(w, u, lvl_packed, a, cnt, ok, order, a_sorted, groups, nch, K, tau,
                               prefix, gate_bias, want_w)
    mode = MODE["v"]
    if mode == "full":
        return rows, g, stats
    B = int(w.size(0)); dev = w.device
    okf = ok.reshape(-1)
    if mode == "gate0":
        return rows, torch.full_like(g, torch.finfo(g.dtype).min), stats
    c3 = cnt.view(1, nch, 1).expand(B, nch, K)
    if mode == "random":
        idx = (torch.rand(B, nch, K, device=dev) * c3.float()).long()
    else:   # recent: the K rows that became available most recently
        idx = c3 - 1 - torch.arange(K, device=dev).view(1, 1, K)
    idx = idx.clamp(min=0, max=int(order.numel()) - 1)
    new = order.index_select(0, idx.reshape(-1)).view(B, nch * K)
    new = torch.where(okf.view(1, -1), new, torch.zeros_like(new))
    same = (new.view(B, nch, K, 1) == rows.view(B, nch, 1, K)).any(-1) & ok.view(1, nch, K)
    STATS.setdefault(mode, []).append(float(same.sum()) / max(1.0, float(ok.sum()) * B))
    g2 = w.gather(1, new)
    if gate_bias is not None:
        mu, var = H._nom_prefix_stats_fn(w, a, cnt, nch)
        g2 = ((g2.view(B, nch, K) - mu.unsqueeze(-1)) / (var + 1e-6).sqrt().unsqueeze(-1)).reshape(B, nch * K)
        g2 = g2 + gate_bias.to(g2.dtype)
    g2 = torch.where(okf.view(1, -1), g2, torch.full_like(g2, torch.finfo(g2.dtype).min))
    return new, g2, stats
H._nom_stream_select_fn = _sel

@torch.no_grad()
def nll_all():
    out = []
    for i in range(0, NSEQ, B):
        ids = seqs[i:i + B].cuda()
        rt = ids.clone(); rt[:, :-1] = ids[:, 1:]
        rm = torch.zeros_like(ids, dtype=torch.bool); rm[:, :-1] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(ids, attention_mask=None, reveal_target_ids=rt, reveal_mask=rm)
        o = o.get("logits", o) if isinstance(o, dict) else o
        f = o[0] if isinstance(o, (tuple, list)) else o
        f = f.float()[:, :-1]
        out.append(F.cross_entropy(f.reshape(-1, f.size(-1)), ids[:, 1:].reshape(-1),
                                   reduction="none").view(ids.size(0), -1).cpu())
    return torch.cat(out).numpy()

torch.manual_seed(0)
MODE["v"] = "full"; nll_all()                                   # warm-up (compile)
res = {}
for mode in ["full", "random", "recent", "full"]:
    MODE["v"] = mode
    torch.manual_seed(0)
    n = nll_all()
    key = mode if mode not in res else mode + "_rep"
    res[key] = n
    row = [math.exp(n[bk == b].mean()) for b in range(len(LABELS))]
    print(f"{key:10s} ppl {math.exp(n.mean()):7.3f} | " +
          " ".join(f"{l}={v:.2f}" for l, v in zip(LABELS, row)), flush=True)

print("\nreproducibility full vs full_rep: max|dNLL|", float(np.abs(res["full"] - res["full_rep"]).max()))
print("\npaired dNLL (cond - full), nats per token, 95% CI by bootstrap over sequences:")
rng = np.random.default_rng(0)
summary = {}
for mode in ["random", "recent"]:
    d = res[mode] - res["full"]
    line = []
    for b in [None] + list(range(len(LABELS))):
        msk = np.ones_like(bk, dtype=bool) if b is None else (bk == b)
        per = np.array([d[s][msk[s]].sum() for s in range(NSEQ)]); cnt = np.array([msk[s].sum() for s in range(NSEQ)])
        est = per.sum() / cnt.sum()
        bs = []
        for _ in range(2000):
            ix = rng.integers(0, NSEQ, NSEQ); bs.append(per[ix].sum() / max(1, cnt[ix].sum()))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        name = "all" if b is None else LABELS[b]
        line.append(f"{name}={est:+.4f}[{lo:+.4f},{hi:+.4f}]")
        summary.setdefault(mode, {})[name] = (est, lo, hi)
    print(f"  {mode:7s} " + "  ".join(line), flush=True)
print("control overlap with the selector picks:", {k: round(float(np.mean(v)), 3) for k, v in STATS.items()})
print("bucket counts:", {LABELS[b]: int((bk == b).sum()) for b in range(len(LABELS))})
json.dump({"summary": summary, "ppl": {k: float(math.exp(v.mean())) for k, v in res.items()}}, open(OUT, "w"), indent=1)
