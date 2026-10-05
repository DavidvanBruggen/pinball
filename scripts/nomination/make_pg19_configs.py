"""Derive the PG19 16k arm configs from the WikiText 4k arms (one rule set, applied to all)."""
import re, pathlib

C = pathlib.Path("/home/david/Projects/pinball/configs")
SRC = {
    "nope": "pinball_wikitext_pack_glob400_l0coarse_l0sel_q_nope_d384_4k.yaml",
    "q": "pinball_wikitext_pack_glob400_l0coarse_l0sel_q_d384_4k.yaml",
    "nope_noslots": "pinball_wikitext_pack_glob400_l0coarse_l0sel_q_nope_d384_4k.yaml",
    "transformer": "transformer_wikitext_d384_4k.yaml",
}
OUT = {
    "nope": "pinball_pg19_16k_l0coarse_nope.yaml",
    "q": "pinball_pg19_16k_l0coarse_q.yaml",
    "nope_noslots": "pinball_pg19_16k_l0coarse_nope_noslots.yaml",
    "transformer": "transformer_pg19_16k_d384.yaml",
}
WHAT = {
    "nope": "MAIN ARM: far NoPE + far-key bias + hierarchy-nominated sparse slots",
    "q": "NoPE CONTROL: identical but every key RoPE-rotated (the WikiText q arm)",
    "nope_noslots": "NOMINATION CONTROL: far NoPE + bias, static block + window only (no sparse slots)",
    "transformer": "DENSE BASELINE: matched GPT-style transformer, d384, 12 layers, RoPE",
}
VER = {
    "nope": "71.6M params; static block 384 rows (L4+L5); 159 chunks x 256 slots; peak 19.9 GiB\n"
            "#   (b1, eager fwd+bwd); flex 0 failed; causal p=12000 exact 0; stream selector == dense picks\n"
            "#   (verify_stream.py @16k)",
    "q": "71.6M params; static block 384 rows; peak 20.3 GiB (b1, eager); flex 0 failed",
    "nope_noslots": "68.6M params; static block 384 rows, 0 slots; peak 17.9 GiB (b1, eager); flex 0\n"
                    "#   failed; causal p=12000 exact 0; levels-prefix far NoPE == dense reference (<= 4.8e-7,\n"
                    "#   scripts/nomination/verify_levels_nope.py)",
    "transformer": "46.9M params; peak 14.7 GiB (b1, eager fwd+bwd)",
}
WIKI = {
    "nope": "0.947 overall / 0.910 512-2k / 0.860 >2k vs the transformer (ep54-63)",
    "q": "0.959 / 0.940 / 0.950 vs the transformer (ep42-51)",
    "nope_noslots": "(new arm)",
    "transformer": "best 24.53 @ ep109",
}

def sub(s, key, val, count=1):
    pat = rf"^{re.escape(key)}:.*$"
    n = len(re.findall(pat, s, flags=re.M))
    assert n == count, (key, n)
    return re.sub(pat, f"{key}: {val}", s, flags=re.M)

COMMON_HDR = """# WIDTH d384 (as the WikiText arms). Measured 16k x6 on the Blackwell, compiled, steady state:
#   transformer 185k tok/s (84.9 GB); pinball NoPE + slots (stream selector) 160k (0.87x,
#   70.7 GB); pinball no-slots 184k (0.99x, 62.3 GB). d768 gave the same ratio (0.82-0.87x at
#   16k x4), so width does not buy speed here; d384 trains ~2x faster per token.
# PG19 SETTINGS (all four arms identical):
#   text_file ./data/pg19_train.txt -- the loader reads the token cache next to it
#     (./data/pg19_train.pt, 24.5 GB int64, mmap). Copy BOTH, or it re-tokenises 11 GB of text.
#   block_size 16384; val_split 0.01 (last 1% of train, ~30M tokens; 50 eval batches/epoch).
#   samples_per_epoch 1000 = 16.4M tokens/epoch (same tokens/epoch as the WikiText arms);
#   num_epochs 200 = 3.3B tokens ~ one PG19 pass. batch_size: SET PER GPU (see below).
#   longctx_diag_long_edges [8192] -> buckets never, <128, 128-511, 512-2k, 2k-8k, >8k.
#   Report data/pg19_test.txt at the end.
# BATCH: at d384, 16k, compiled: ~11.6 GB per sequence for pinball, ~13.9 GB for the
#   transformer (batch 6 = 71 / 85 GB). Batch 1 was LAUNCH-BOUND for pinball (105k vs 146k tok/s
#   at batch 4): use batch >= 4. Measure on the cluster GPU and set batch_size (and/or
#   gradient_accumulation_steps) so all arms see the SAME tokens per optimizer step.
# CHECK on every run: grep -ci "flex.*fail" <log> must be 0 (pinball arms). The far-NoPE
#   arms raise instead of degrading; the q control could degrade silently to additive.
"""
PINBALL_HDR = """# SELECTOR (slot arms): local_pack_global_select_impl stream + gumbel_norm prefix (2026-10-05):
#   same picks as the dense selector (bit-equal in eval/raw/prefix, verify_stream.py), linear;
#   prefix = noise scaled by the spread at each row's activation chunk (was chunk: per chunk).
#   16k x6 Blackwell: 146k -> 160k tok/s. The WikiText arms these derive from used chunk noise.
# local_pack_global_nom_compile: true (slot arms): the selector runs as ONE compiled graph shared
#   by all layers (training only). fp32 == eager ~1e-7; bf16 picks 98.6-99.9% equal (fused
#   rounding; verify_nom_compile.py). 16k x6: l0_coarse 163.2k, shared 168.1k, transformer
#   185.8k tok/s. NO eager fallback: a compile failure on a new GPU crashes the run (intended).
# HIERARCHY AT 16k (pinball arms): compression_ratios [16,4,4,4,4] (was [16,4,4]) -> coarse
#   levels 2048/1024/512/256/128 rows. local_pack_global_block 400 then holds L5+L4 = 384
#   rows: the SAME static block size as L3+L2 at 4096, so the far cost per query stays
#   constant (keeping 3 levels would need 1536 static rows per query -- quadratic in N).
#   Level lists extended to 6 levels: num_layers, internal_cycles, local_pack_query_levels,
#   local_attn_levels/windows/causal_levels. Slot budget (256 per 128-row chunk) and the
#   ~128-token window unchanged (HSA, 2510.17196: fixed K, small window).
"""

for arm, src in SRC.items():
    s = (C / src).read_text()
    pin = arm != "transformer"
    s = sub(s, "text_file", "./data/pg19_train.txt   # + ./data/pg19_train.pt (token cache) alongside")
    s = sub(s, "block_size", "16384")
    s = sub(s, "batch_size", "4                      # SET PER GPU: ~21 GiB per 16k sequence (pinball, eager)")
    s = sub(s, "samples_per_epoch", "1000              # 16.4M tokens/epoch")
    s = sub(s, "num_epochs", "200                       # ~3.3B tokens ~ one PG19 pass")
    s = sub(s, "longctx_diag_every", "1\nlongctx_diag_long_edges: [8192]   # adds 2k-8k and >8k buckets")
    s = sub(s, "gen_prompt_tokens", "1024")
    s = re.sub(r"^checkpoint_dir:.*$", f"checkpoint_dir: ./{'transformer' if not pin else 'pinball'}/pg19_16k_{arm}/checkpoints",
               s, count=1, flags=re.M)
    if pin:
        s = sub(s, "compression_ratios", "[16, 4, 4, 4, 4]   # 16k: +2 coarse levels (see header)")
        s = sub(s, "overlap_ratios", "[0.5, 0.5, 0.5, 0.5, 0.5]")
        s = sub(s, "num_layers", "[0, 0, 0, 0, 0, 0]")
        s = sub(s, "internal_cycles", "[0, 0, 0, 0, 0, 0]")
        s = sub(s, "local_pack_query_levels", "[0, 1, 2, 3, 4, 5]   # MUST cover every level")
        s = sub(s, "local_attn_levels", "[0, 1, 2, 3, 4, 5]")
        s = sub(s, "local_attn_windows", "[16, 16, 16, 16, 16, 16]")
        s = sub(s, "local_attn_causal_levels", "[0, 1, 2, 3, 4, 5]")
    if arm in ("nope", "q"):
        # linear selector (2026-10-05): stream needs a per-row noise scale -> prefix
        s = sub(s, "local_pack_global_gumbel_norm", "prefix   # noise scale at each row's activation chunk (stream-exact)")
        s = s.replace("local_pack_global_gumbel_norm: prefix   # noise scale at each row's activation chunk (stream-exact)",
                      "local_pack_global_gumbel_norm: prefix   # noise scale at each row's activation chunk (stream-exact)\n"
                      "local_pack_global_select_impl: stream   # linear selector + fused nomination gather (+10% @16k)\n"
                      "local_pack_global_nom_compile: true     # selector as one compiled graph (training); no eager fallback")
    if arm == "nope_noslots":
        # drop every slot/nomination knob (their validations require content selection);
        # the static block (local_pack_global_block) and far NoPE + bias stay
        for k in ("coarse", "chunk", "l0_budget", "logit", "score_bias", "nominator", "candidates",
                  "gumbel", "nom_gate", "boost", "gumbel_norm"):
            s = re.sub(rf"^local_pack_global_{k}:.*$", f"# local_pack_global_{k}: (removed: no slots in this arm)",
                       s, count=1, flags=re.M)
        s = sub(s, "local_pack_global_select", "levels   # NO sparse slots: static block + window (control)")
    hdr = (f"# ============================================================================\n"
           f"# PG19 @16k, d384 -- {WHAT[arm]} (2026-10-05).\n"
           f"# Derived from {src} (WikiText 4k: {WIKI[arm]}) by the PG19 rules below; nothing\n"
           f"# else differs.\n" + COMMON_HDR + (PINBALL_HDR if pin else "") +
           f"# VERIFIED @16k on the Blackwell (scripts/nomination/smoke_pg19.py): {VER[arm]}.\n"
           "# SCORES: (none yet)\n"
           "# ============================================================================\n"
           "# ---- parent header (history) ----\n")
    (C / OUT[arm]).write_text(hdr + s)
    print("wrote", OUT[arm])
