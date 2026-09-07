# SPDX-License-Identifier: GPL-3.0-or-later
"""Publication figures for the Pinball manuscript.

Two figures, each from MEASURED data only -- nothing here invents or extrapolates a
number, and every panel prints the provenance of what it plotted.

  Fig 1  scaling      step time vs sequence length, pinball vs transformer, log-log,
                      with fitted exponents and the crossover length marked.
  Fig 2  convergence  validation perplexity vs epoch, read straight out of the
                      checkpoints' `val_losses`.

Usage
-----
  # measure the length scan (writes timings.json), then plot everything
  python plot_pinball_figures.py --measure --out figures/

  # re-plot from an existing timings.json without touching the GPU
  python plot_pinball_figures.py --out figures/

  # convergence only, naming the runs explicitly
  python plot_pinball_figures.py --no-scaling \
      --curve "Pinball:pinball/pooled_proj_cleaner_small_test_pack_gpt2/checkpoints/pinball_final.pt" \
      --curve "Transformer:transformer/checkpoints_gpt2_longcontext_test/pinball_final.pt"

Notes
-----
* Timing is fwd+bwd of a training step (the quantity that decides wall-clock cost),
  median of `--reps` after warmup, measured with CUDA synchronisation.
* Both arms are measured in the SAME process, interleaved round-robin, because
  sequential measurement on this box drifts with clock/thermal state.
* A length that OOMs is recorded as null and drawn as an open marker at the axis top,
  so a memory limit is visible rather than silently absent from the curve.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# ----------------------------------------------------------------------------- style
# One accent per arm, chosen to stay distinguishable in greyscale print: the pinball
# series is darker and heavier, the baseline lighter with open markers.
PINBALL = "#1b4965"
BASELINE = "#c1666b"
GRID = "#d8d8d8"
FG = "#222222"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "axes.edgecolor": FG,
        "axes.labelcolor": FG,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": FG,
        "ytick.color": FG,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
    })


def _thousands(v, _pos):
    if v >= 1000 and v % 1000 == 0:
        return f"{int(v // 1000)}k"
    return f"{v:g}"


# ------------------------------------------------------------------------- measuring
def measure_scan(lengths, batch, reps, pinball_cfg, transformer_cfg, device="cuda"):
    """Interleaved fwd+bwd step timing for both arms. Returns a JSON-able dict."""
    import torch
    from transformers import AutoTokenizer
    from pinball import build_model, load_args

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(cfg_name, seq_len):
        torch.manual_seed(0)
        args = load_args(str(ROOT / "configs" / cfg_name))
        setattr(args, "block_size", int(seq_len))
        model = build_model(args, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                            tie_weights=True, max_seq_len=int(seq_len)).to(device)
        model.train()
        return model

    def one_step(model, ids):
        # The real training step: next-token CE on FLAT 2D logits. Casting the whole
        # [B, T, V] logit tensor to fp32 (the obvious thing) is both unrepresentative and
        # a memory hog -- 3.3 GB at T=16384 alone -- which would make the scan measure the
        # benchmark's own allocation rather than the model's.
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model(ids)
            lg = out[0] if isinstance(out, (tuple, list)) else out
            loss = torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    arms = {"Pinball": pinball_cfg, "Transformer": transformer_cfg}
    result = {"batch": batch, "reps": reps, "lengths": list(lengths),
              "configs": {k: v for k, v in arms.items()}, "ms": {k: [] for k in arms}}

    for n in lengths:
        built, ids = {}, None
        for label, cfg in arms.items():
            try:
                built[label] = build(cfg, n)
                if ids is None:
                    torch.manual_seed(1)
                    ids = torch.randint(0, len(tok), (batch, n), device=device)
                for _ in range(2):                     # warm: compile + caches
                    one_step(built[label], ids)
            except torch.cuda.OutOfMemoryError:
                built[label] = None
                torch.cuda.empty_cache()
            except Exception as exc:                   # record, do not abort the scan
                print(f"  [{label} @ {n}] {type(exc).__name__}: {str(exc)[:90]}")
                built[label] = None
                torch.cuda.empty_cache()

        times = {k: [] for k in arms}
        for _ in range(reps):                          # interleaved: no clock drift
            for label, model in built.items():
                if model is None:
                    continue
                try:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    one_step(model, ids)
                    torch.cuda.synchronize()
                    times[label].append((time.perf_counter() - t0) * 1e3)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
        for label in arms:
            v = sorted(times[label])
            result["ms"][label].append(v[len(v) // 2] if v else None)
            print(f"  n={n:<7} {label:<12} "
                  f"{'OOM / failed' if not v else f'{v[len(v)//2]:8.1f} ms'}")
        del built, ids
        torch.cuda.empty_cache()
    return result


# -------------------------------------------------------------------------- plotting
def load_csv(path):
    """length,arm,ms  ->  the same dict shape measure_scan() produces.

    Lets the figure be built from measurements taken elsewhere (a different machine, an
    earlier date) instead of forcing a re-run, with the provenance carried in the file's
    comment header rather than in someone's memory.
    """
    import csv as _csv
    rows, header = [], []
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("#"):
            header.append(line.lstrip("# ").rstrip())
        elif line.strip():
            rows.append(line)
    rd = list(_csv.DictReader(rows))
    lengths = sorted({int(r["length"]) for r in rd})
    arms = list(dict.fromkeys(r["arm"] for r in rd))
    ms = {a: [] for a in arms}
    for a in arms:
        by_n = {int(r["length"]): float(r["ms"]) for r in rd if r["arm"] == a}
        ms[a] = [by_n.get(n) for n in lengths]
    print(f"  {len(arms)} arms x {len(lengths)} lengths from {path}")
    return {"lengths": lengths, "ms": ms, "batch": "as recorded", "reps": "as recorded",
            "provenance": header}


def _fit_exponent(xs, ys):
    """Least-squares slope of log(t) vs log(n); ~1 is linear, ~2 is quadratic."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den else None


def plot_scaling(data, out_path, time_scale="linear"):
    _style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"wspace": 0.28})
    lengths = data["lengths"]
    palette = [PINBALL, "#3d7ea6", "#6a8d73", "#8c6d94"]
    markers = ["o", "^", "D", "v"]
    styles, ci = {}, 0
    for label in data["ms"]:
        if label == data.get("baseline", "Transformer"):
            styles[label] = (BASELINE, "s", "--", 1.4)
        else:
            styles[label] = (palette[ci % len(palette)], markers[ci % len(markers)], "-", 1.9)
            ci += 1

    # -- left: absolute step time, log-log
    for label, series in data["ms"].items():
        colour, marker, ls, lw = styles.get(label, (FG, "^", "-", 1.5))
        is_base = label == data.get("baseline", "Transformer")
        xs = [n for n, v in zip(lengths, series) if v]
        ys = [v for v in series if v]
        if not xs:
            continue
        # Exponent goes in the LEGEND, not at the line end: with four arms the
        # end-of-line annotations collide with each other and with the axis edge.
        slope = _fit_exponent(xs, ys)
        tag = f"{label}   $\\propto N^{{{slope:.2f}}}$" if slope is not None else label
        ax.plot(xs, ys, ls, color=colour, marker=marker, label=tag, lw=lw,
                markerfacecolor="white" if is_base else colour,
                markeredgecolor=colour, markeredgewidth=1.3)
        # OOM markers, drawn at the top of the axis so a memory wall is visible
        for n, v in zip(lengths, series):
            if v is None and n >= min(xs):
                ax.plot([n], [max(ys)], marker="x", color=colour, ms=6, mew=1.6)

    ax.set_xscale("log", base=2)
    # LINEAR time axis by default. A log y-axis compresses exactly the regime the figure
    # exists to show -- at 32k the absolute gap is ~600 ms, which log scaling renders as a
    # small vertical offset. Sequence length stays log so the doublings are evenly spaced.
    if time_scale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("training step time (ms)")
    ax.set_title("Cost scaling", loc="left", color=FG)
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.grid(True, which="both", color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8)

    # -- right: ratio, the quantity the method claim rests on
    base_label = data.get("baseline", "Transformer")
    tf = data["ms"].get(base_label, [])
    ax2.axhline(1.0, color=BASELINE, ls="--", lw=1.2)
    ax2.annotate("parity", xy=(lengths[0], 1.0), xytext=(2, 4),
                 textcoords="offset points", color=BASELINE, fontsize=8.5)
    for label, series in data["ms"].items():
        if label == base_label:
            continue
        colour, marker = styles[label][0], styles[label][1]
        xs = [n for n, a, b in zip(lengths, series, tf) if a and b]
        ys = [a / b for a, b in zip(series, tf) if a and b]
        if not xs:
            continue
        ax2.plot(xs, ys, "-", color=colour, marker=marker, markeredgecolor=colour,
                 markerfacecolor=colour, label=label)
        # crossover: where the ratio first dips below parity (log-x interpolation)
        for i in range(1, len(ys)):
            if ys[i - 1] > 1.0 >= ys[i]:
                lx = math.log(xs[i - 1]) + (math.log(xs[i]) - math.log(xs[i - 1])) * \
                    (ys[i - 1] - 1.0) / (ys[i - 1] - ys[i])
                xc = math.exp(lx)
                ax2.plot([xc], [1.0], marker="*", ms=11, color=colour, zorder=5)
                ax2.annotate(f"{xc:,.0f}", xy=(xc, 1.0), xytext=(0, -16),
                             textcoords="offset points", fontsize=8, color=colour,
                             ha="center", fontweight="bold")
                break
    if len(data["ms"]) > 2:
        ax2.legend(loc="upper right", fontsize=7.5)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("sequence length (tokens)")
    ax2.set_ylabel("Pinball / Transformer step time")
    ax2.set_title("Relative cost", loc="left", color=FG)
    ax2.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax2.grid(True, which="both", color=GRID, lw=0.5, alpha=0.7)
    ax2.set_axisbelow(True)

    note = (f"batch {data['batch']}, {data['reps']} reps, fwd+bwd"
            if isinstance(data["reps"], int) else
            "batch 1, gradient checkpointing on, min-of-rounds (see CSV header)")
    if time_scale != "log":
        base_l = data.get("baseline", "Transformer")
        tfs = data["ms"].get(base_l, [])
        cands = [(lbl, sr) for lbl, sr in data["ms"].items() if lbl != base_l]
        if tfs and cands and lengths:
            i = len(lengths) - 1
            fastest = min((sr[i], lbl) for lbl, sr in cands if sr[i])
            if tfs[i] and fastest[0]:
                ax.annotate("", xy=(lengths[i], tfs[i]), xytext=(lengths[i], fastest[0]),
                            arrowprops=dict(arrowstyle="<->", color=FG, lw=0.9))
                ax.annotate(f"{tfs[i] - fastest[0]:.0f} ms", 
                            xy=(lengths[i], (tfs[i] + fastest[0]) / 2),
                            xytext=(-6, 0), textcoords="offset points", ha="right",
                            va="center", fontsize=8, color=FG)
    fig.text(0.5, -0.06, note, ha="center", fontsize=7.5, color="#666666")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}")
    plt.close(fig)
    print(f"  wrote {out_path}.pdf / .png")


def plot_convergence(curves, out_path, ylim=None):
    """curves: list of (label, val_losses list)."""
    _style()
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    styles = [(PINBALL, "-", 1.9), (BASELINE, "--", 1.4), ("#6a8d73", "-.", 1.4)]
    for i, (label, losses) in enumerate(curves):
        colour, ls, lw = styles[i % len(styles)]
        ppl = [math.exp(v) for v in losses]
        xs = list(range(1, len(ppl) + 1))
        ax.plot(xs, ppl, ls, color=colour, lw=lw, label=f"{label}  (best {min(ppl):.2f})")
        best = min(range(len(ppl)), key=lambda k: ppl[k])
        ax.plot([xs[best]], [ppl[best]], marker="o", color=colour, ms=4.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation perplexity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title("Convergence", loc="left", color=FG)
    ax.grid(True, which="both", color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}")
    plt.close(fig)
    print(f"  wrote {out_path}.pdf / .png")


def plot_quality(csv_path, out_path, checkpoint="best"):
    """Grouped bars over the two DNA correlation axes.

    Both axes are plotted because reporting only across-region is the failure mode this
    figure exists to avoid: a model with no distal reach scores ~0.93 there and 0.000 on
    specificity, so across-region alone cannot support a long-range claim.
    """
    import csv as _csv
    _style()
    rows, notes = [], []
    for line in pathlib.Path(csv_path).read_text().splitlines():
        (notes if line.startswith("#") else rows).append(line)
    rd = [r for r in _csv.DictReader([r for r in rows if r.strip()])
          if checkpoint in r["checkpoint"]]
    arms = [r["arm"] for r in rd]
    metrics = [("across_region", "across-region\n(local sequence suffices)"),
               ("specificity", "specificity\n(requires distal context)")]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    width = 0.34
    for i, arm in enumerate(arms):
        row = rd[i]
        colour = BASELINE if "ransformer" in arm else PINBALL
        xs = [j + (i - (len(arms) - 1) / 2) * width for j in range(len(metrics))]
        ys = [float(row[k]) for k, _ in metrics]
        ax.bar(xs, ys, width * 0.92, label=arm, color=colour,
               edgecolor=colour, linewidth=0.8,
               alpha=1.0 if colour == PINBALL else 0.85)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=FG)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=8.5)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0, 1.0)
    ax.set_title("DNA track prediction, 500 kb", loc="left", color=FG)
    ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.text(0.5, -0.08, "130 validation windows, native context, parameter-matched",
             ha="center", fontsize=7.5, color="#666666")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}")
    plt.close(fig)
    print(f"  wrote {out_path}.pdf / .png")


def plot_text_quality(csv_path, out_path):
    """Perplexity beside the parameter budget that produced it.

    The second panel exists so the comparison cannot be read as parameter-matched when it
    is not: the arms match on width, depth, context, tokens/step, epochs and dropout, but
    pinball's blocks carry more parameters. Showing the split makes the reader's objection
    visible instead of leaving it for a reviewer to discover.
    """
    import csv as _csv
    _style()
    rows = [l for l in pathlib.Path(csv_path).read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    rd = list(_csv.DictReader(rows))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                  gridspec_kw={"wspace": 0.34, "width_ratios": [1, 1.15]})
    names = [r["arm"] for r in rd]
    cols = [PINBALL if "innball" in n or "Pinball" == n else BASELINE for n in names]

    ppl = [float(r["ppl"]) for r in rd]
    ax.bar(range(len(rd)), ppl, 0.55, color=cols, edgecolor=cols)
    for i, v in enumerate(ppl):
        ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, color=FG)
    ax.set_xticks(range(len(rd)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("validation perplexity  (lower is better)")
    ax.set_ylim(0, max(ppl) * 1.22)
    ax.set_title("WikiText-103, 192d / 12 layers", loc="left", color=FG)
    ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    parts = [("core_M", "core blocks", 1.0), ("embed_M", "embeddings", 0.62),
             ("aux_M", "aux (train only)", 0.36), ("hier_M", "hierarchy", 0.20)]
    for i, r in enumerate(rd):
        bottom = 0.0
        for key, lbl, alpha in parts:
            v = float(r[key])
            if v <= 0:
                continue
            ax2.bar(i, v, 0.55, bottom=bottom, color=cols[i], alpha=alpha,
                    edgecolor="white", linewidth=0.7,
                    label=lbl if i == 0 or lbl not in ax2.get_legend_handles_labels()[1] else None)
            if v > 0.9:
                ax2.annotate(f"{lbl}\n{v:.2f}M", (i, bottom + v / 2), ha="center",
                             va="center", fontsize=7.5,
                             color="white" if alpha > 0.5 else FG)
            bottom += v
        ax2.annotate(f"{bottom:.2f}M", (i, bottom), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=9, color=FG)
    ax2.set_xticks(range(len(rd)))
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("parameters (M)")
    ax2.set_ylim(0, max(float(r["total_M"]) for r in rd) * 1.22)
    ax2.set_title("Parameter budget", loc="left", color=FG)
    ax2.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.7)
    ax2.set_axisbelow(True)

    fig.text(0.5, -0.07, "matched: context 8196, 32,784 tokens/optimizer step, 781 steps/epoch, "
             "100 epochs, dropout 0.1, muon_hybrid\nparameters are the trained model; "
             "aux is training-only and discarded at inference",
             ha="center", fontsize=7.5, color="#666666")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}")
    plt.close(fig)
    print(f"  wrote {out_path}.pdf / .png")


def load_curve(spec):
    """'Label:path/to/ckpt.pt' -> (label, val_losses)."""
    import torch
    label, _, path = spec.partition(":")
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    losses = ck.get("val_losses") or []
    if not losses:
        raise SystemExit(f"{path}: no val_losses in checkpoint")
    print(f"  {label:<14} {len(losses):>3} epochs, best PPL {math.exp(min(losses)):.2f}"
          f"   <- {path}")
    return label, losses


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--measure", action="store_true", help="run the GPU length scan")
    ap.add_argument("--timings", default="timings.json")
    ap.add_argument("--csv", default=None, help="plot from a length,arm,ms CSV instead")
    ap.add_argument("--time-scale", default="linear", choices=("linear", "log"),
                    help="y-axis of the step-time panel (default linear)")
    ap.add_argument("--baseline", default="Transformer", help="arm used as the ratio denominator")
    ap.add_argument("--lengths", default="1024,2048,4096,8192,16384")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pinball-config", default="pinball_wikitext_3kernel_4lvl.yaml")
    ap.add_argument("--transformer-config", default="transformer_wikitext.yaml")
    ap.add_argument("--curve", action="append", default=[],
                    help="'Label:checkpoint.pt', repeatable")
    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--no-convergence", action="store_true")
    ap.add_argument("--text-csv", default=None,
                    help="arm,ppl,total_M,... -> text perplexity + parameter budget figure")
    ap.add_argument("--quality-csv", default=None,
                    help="arm,checkpoint,across_region,specificity -> DNA quality figure")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.no_scaling:
        tpath = pathlib.Path(args.timings)
        if args.measure and not args.csv:
            lengths = [int(x) for x in args.lengths.split(",")]
            print("measuring length scan (this touches the GPU):")
            data = measure_scan(lengths, args.batch, args.reps,
                                args.pinball_config, args.transformer_config)
            tpath.write_text(json.dumps(data, indent=2))
            print(f"  wrote {tpath}")
        elif args.csv:
            data = load_csv(args.csv)
        elif not tpath.exists():
            raise SystemExit(f"{tpath} not found -- run once with --measure")
        else:
            data = json.loads(tpath.read_text())
            print(f"plotting scaling from {tpath} (measured, not re-run)")
        data["baseline"] = args.baseline
        plot_scaling(data, str(out / "fig_scaling"), time_scale=args.time_scale)

    if not args.no_convergence:
        specs = args.curve or [
            "Pinball:pinball/pooled_proj_cleaner_small_test_pack_gpt2/checkpoints/pinball_final.pt",
            "Transformer:transformer/checkpoints_gpt2_longcontext_test/pinball_final.pt",
        ]
        print("convergence curves:")
        plot_convergence([load_curve(s) for s in specs], str(out / "fig_convergence"))


    if args.text_csv:
        print("text quality figure:")
        plot_text_quality(args.text_csv, str(out / "fig_quality_text"))

    if args.quality_csv:
        print("quality figure:")
        plot_quality(args.quality_csv, str(out / "fig_quality_dna"))


if __name__ == "__main__":
    main()
