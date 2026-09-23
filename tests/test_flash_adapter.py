"""CPU tests for the FA3/FA4 -> FA2-dialect adapter and the flash preference plumbing.

FA3 and FA4 cannot run on either local card (sm_89, sm_120), so the real kernels are
stood in for by fakes with the same call signature and return convention. What is under
test is the translation layer, which is where every behavioural difference lives:
dropout, the unbounded-window spelling, and the (out, lse[, S]) return shape.
"""
import pytest
import torch

import pinball.model.layers.hierarchical_message_passing as hmp


def _fa4_fake(record, lse_when_nograd=False, has_return_lse=False):
    """flash_attn.cute.flash_attn_func: always (out, lse); lse only if an input needs grad."""
    if has_return_lse:
        def fa4(q, k, v, softmax_scale=None, causal=False, window_size=(None, None),
                learnable_sink=None, softcap=0.0, pack_gqa=None, return_lse=False):
            record.update(window_size=window_size, causal=causal, v=v, return_lse=return_lse)
            lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1]) if return_lse else None
            return q.clone(), lse
    else:
        def fa4(q, k, v, softmax_scale=None, causal=False, window_size=(None, None),
                learnable_sink=None, softcap=0.0, pack_gqa=None):
            record.update(window_size=window_size, causal=causal, v=v)
            needs = q.requires_grad or k.requires_grad or v.requires_grad or lse_when_nograd
            lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1]) if needs else None
            return q.clone(), lse
    return fa4


def _fa3_fake(record):
    """flash_attn_interface.flash_attn_func: out, or (out, lse) with return_attn_probs."""
    def fa3(q, k, v, softmax_scale=None, causal=False, qv=None, q_descale=None,
            k_descale=None, v_descale=None, window_size=(-1, -1), attention_chunk=0,
            softcap=0.0, num_splits=1, pack_gqa=None, deterministic=False, sm_margin=0,
            return_attn_probs=False):
        record.update(window_size=window_size, causal=causal, v=v)
        out = q.clone()
        return (out, torch.zeros(q.shape[0], q.shape[2], q.shape[1])) if return_attn_probs else out
    return fa3


def _qkv(requires_grad=False):
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 16, 4, 8, generator=g) for _ in range(3))
    if requires_grad:
        q.requires_grad_(True)
    return q, k, v


def test_adapter_exposes_fa2_signature():
    # attention_forward introspects the signature to decide which kwargs to pass.
    fn = hmp._flash_adapter(_fa4_fake({}), "fa4", "error")
    assert hmp._flash_attn_supports_window_size(fn)
    assert hmp._flash_attn_supports_dropout(fn)


@pytest.mark.parametrize("impl,fake", [("fa4", _fa4_fake), ("fa3", _fa3_fake)])
def test_dropout_errors_by_default(impl, fake):
    fn = hmp._flash_adapter(fake({}), impl, "error")
    q, k, v = _qkv()
    with pytest.raises(RuntimeError, match="no attention dropout"):
        fn(q, k, v, dropout_p=0.1, causal=True, window_size=(4, 0))
    fn(q, k, v, dropout_p=0.0, causal=True, window_size=(4, 0))  # zero is always fine


@pytest.mark.parametrize("impl,fake", [("fa4", _fa4_fake), ("fa3", _fa3_fake)])
def test_token_v_dropout_masks_whole_key_tokens(impl, fake):
    rec = {}
    fn = hmp._flash_adapter(fake(rec), impl, "token_v")
    q, k, v = _qkv()
    torch.manual_seed(0)
    fn(q, k, v, dropout_p=0.5, causal=False)
    v_seen = rec["v"]
    ratio = v_seen / v
    # one decision per (batch, key token, head), shared across the head dim
    assert torch.allclose(ratio, ratio[..., :1].expand_as(ratio))
    assert set(torch.unique(ratio.round(decimals=5)).tolist()) <= {0.0, 2.0}
    assert (ratio == 0).any() and (ratio == 2).any()


def test_fa4_window_minus_one_becomes_none():
    rec = {}
    fn = hmp._flash_adapter(_fa4_fake(rec), "fa4", "error")
    q, k, v = _qkv()
    fn(q, k, v, window_size=(-1, -1))
    assert rec["window_size"] == (None, None)
    fn(q, k, v, window_size=(16, 0), causal=True)
    assert rec["window_size"] == (16, 0)


def test_fa3_window_passthrough():
    rec = {}
    fn = hmp._flash_adapter(_fa3_fake(rec), "fa3", "error")
    q, k, v = _qkv()
    fn(q, k, v, window_size=(-1, -1))
    assert rec["window_size"] == (-1, -1)


@pytest.mark.parametrize("impl,fake", [("fa4", _fa4_fake), ("fa3", _fa3_fake)])
def test_return_attn_probs_gives_fa2_triple(impl, fake):
    fn = hmp._flash_adapter(fake({}), impl, "error")
    q, k, v = _qkv(requires_grad=True)
    res = fn(q, k, v, causal=True, window_size=(4, 0), return_attn_probs=True)
    assert isinstance(res, tuple) and len(res) == 3
    out, lse, s = res
    assert out.shape == q.shape and lse.shape == (2, 4, 16) and s is None
    # and a plain call returns the bare tensor, as fa2 does
    assert torch.is_tensor(fn(q, k, v))


def test_fa4_missing_lse_under_nograd_is_loud():
    fn = hmp._flash_adapter(_fa4_fake({}), "fa4", "error")
    q, k, v = _qkv()
    with torch.no_grad(), pytest.raises(RuntimeError, match="no LSE"):
        fn(q, k, v, return_attn_probs=True)


def test_fa4_return_lse_used_when_available():
    rec = {}
    fn = hmp._flash_adapter(_fa4_fake(rec, has_return_lse=True), "fa4", "error")
    q, k, v = _qkv()
    with torch.no_grad():
        out, lse, _ = fn(q, k, v, return_attn_probs=True)
    assert rec["return_lse"] is True and lse is not None


def test_flash_win_lse_runs_through_adapter():
    # _flash_win_lse unpacks three values and passes dropout_p + return_attn_probs directly;
    # before the adapter both FA3 (2-tuple) and FA4 (no dropout_p kwarg) broke it.
    for impl, fake in (("fa4", _fa4_fake), ("fa3", _fa3_fake)):
        fn = hmp._flash_adapter(fake({}), impl, "error")
        q, k, v = _qkv(requires_grad=True)
        out, lse = hmp._flash_win_lse(fn, q, k, v, window=4, causal=True)
        assert out.shape == q.shape and lse.shape == (2, 4, 16)


def test_preference_validation_and_env(monkeypatch):
    with pytest.raises(ValueError):
        hmp.set_flash_preference("fa5")
    with pytest.raises(ValueError):
        hmp.set_flash_preference(nodropout_mode="drop")
    monkeypatch.setenv("PINBALL_FLASH_IMPL", "fa4")
    assert hmp._flash_impl_pref() == "fa4"
    monkeypatch.setenv("PINBALL_FLASH_IMPL", "bogus")
    with pytest.raises(ValueError):
        hmp._flash_impl_pref()
    monkeypatch.delenv("PINBALL_FLASH_IMPL")
    hmp._FLASH_PREF.update(impl=None, nodropout=None)
    assert hmp._flash_impl_pref() == "auto" and hmp._flash_nodropout_pref() == "error"


def test_fa4_refused_off_hopper_and_blackwell_dc():
    fn, exc = hmp._try_fa4(0, (12, 0), "error")      # workstation Blackwell
    assert fn is None and "sm_90/sm_100" in str(exc)
    fn, exc = hmp._try_fa4(0, (8, 9), "error")       # 4090
    assert fn is None
