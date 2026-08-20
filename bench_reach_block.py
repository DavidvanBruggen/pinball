"""Reach for the global-block flexhier arm: A[i,j] = |i-j| <= W  OR  j in block."""
import sys, logging, warnings, torch
warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/david/Projects/pinball/src")
logging.disable(logging.WARNING)
from pinball.instantiate_PINBALL_model import build_pinball
DEV="cuda:1"; CFG=sys.argv[1]
BUD=[int(b) for b in sys.argv[2].split(",")] if len(sys.argv)>2 else [256]
for SEQ in ([int(x) for x in (sys.argv[3].split(',') if len(sys.argv)>3 else ['4096','16384','32768'])]):
    for bud in BUD:
        torch.manual_seed(0)
        m=build_pinball(cfg_path=f"/home/david/Projects/pinball/configs/{CFG}.yaml",
            num_tracks=768,tie_weights=False,device=DEV,set_global_seed=False,
            override=dict(hier_layer_compile=False,hier_refresh_compile=False,
                          use_gradient_checkpointing=False,block_size=SEQ,
                          local_pack_global_block=bud))[0]
        m.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
            m(torch.randn(1,SEQ,768,device=DEV))
        sp=m.refinement_transformers[0].message_passing._local_pack_spec
        n,W=int(sp["num_nodes"]),int(sp["window"]); lvl=sp["levels"]
        g=sp["global_block"]; gm=sp["global_block_mask"]
        r=torch.arange(n,device=DEV)
        A=((r.unsqueeze(1)-r.unsqueeze(0)).abs()<=W) | gm.unsqueeze(0)
        l0=(lvl==0).nonzero().view(-1)
        tok=torch.full((n,),-1,device=DEV,dtype=torch.long); tok[l0]=torch.arange(l0.numel(),device=DEV)
        mid=l0[l0.numel()//2]; F=A.clone()
        # worst-case gap from a token to the nearest block row, in packed slots
        gr=g["rows"].sort().values.float()
        gap=int((torch.cdist(l0.float().unsqueeze(1), gr.unsqueeze(1)).min(1).values.max()))
        print(f"N={SEQ} budget={bud}: pack={n} levels={[int(x.numel()) for x in sp['level_rows']]} "
              f"block={int(g['rows'].numel())} rows L{g['levels']}  worst token->block gap {gap} slots (W={W})")
        for hop in range(1,8):
            vis=tok[F[mid]]; vis=vis[vis>=0]
            pct=100.0*vis.numel()/l0.numel()
            print(f"   after {hop} layer(s): {pct:6.2f}% of tokens")
            if pct>=99.99: print(f"   -> FULL in {hop} layers"); break
            F=(F.float()@A.float())>0
        del m; torch.cuda.empty_cache()
