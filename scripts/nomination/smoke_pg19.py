"""16k check of a PG19 config: build, geometry, one fwd+bwd (peak mem, time), flex health.
usage: python smoke_pg19.py CONFIG [causal]"""
import sys, time, logging, torch
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs

cfgp = sys.argv[1]; do_causal = len(sys.argv) > 2
cfg = PinballConfig.from_yaml(cfgp)
L = int(cfg.block_size)
cfg.hier_layer_compile = False; cfg.transformer_compile = False
inp = resolve_model_inputs(cfg, block_size=L)
torch.manual_seed(0)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=inp.block_size).cuda()
npar = sum(p.numel() for p in m.parameters())
print(f"{cfgp.split('/')[-1]}: L={L} params {npar/1e6:.1f}M")
pin = hasattr(m, "_predict_level_sizes")
if pin:
    print("  level sizes", m._predict_level_sizes(L), "| far_nope", getattr(m, "local_pack_far_nope_dims", 0),
          "bias", getattr(m, "local_pack_far_bias", "-"), "| select", getattr(m, "local_pack_global_select", "-"))
    seen = {}
    mp0 = m.refinement_transformers[0].message_passing
    o_ = mp0._flex_union_attn
    def spy(qp, kp, vp, spec, causal=True, gsel=None):
        gb = spec.get("global_block")
        seen["static"] = (int(gb["rows"].numel()), gb["levels"]) if gb else None
        seen["slots"] = int(gsel[0].size(1)) if (gsel is not None and gsel[0].dim() == 2) else 0
        seen["packed_rows"] = int(spec["num_nodes"])
        return o_(qp, kp, vp, spec, causal, gsel=gsel)
    mp0._flex_union_attn = spy
x = torch.randint(0, inp.vocab_size, (1, L), generator=torch.Generator().manual_seed(1)).cuda()
m.train()
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
with torch.autocast("cuda", torch.bfloat16):
    o = m(x); o = o[0] if isinstance(o, tuple) else o
    loss = torch.nn.functional.cross_entropy(o[0, :-1].float(), x[0, 1:])
loss.backward()
torch.cuda.synchronize()
print(f"  fwd+bwd {time.time() - t:.1f}s (eager, first call) | peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB | loss {float(loss):.3f}")
if pin:
    print(f"  static block {seen.get('static')} | slots/seq (layer 0) {seen.get('slots')} | packed rows {seen.get('packed_rows')}")
    print(f"  flex failed modules: {m.flex_union_status()['failed_modules']}")
if do_causal:
    m.eval(); m.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        y1 = m(x); y1 = (y1[0] if isinstance(y1, tuple) else y1).float()
        x2 = x.clone(); x2[:, 12000:] = torch.randint(0, inp.vocab_size, (1, L - 12000), generator=torch.Generator().manual_seed(2)).cuda()
        y2 = m(x2); y2 = (y2[0] if isinstance(y2, tuple) else y2).float()
    print(f"  causal p=12000: max|diff| before p {(y1[:, :12000] - y2[:, :12000]).abs().max().item():.3e}")
