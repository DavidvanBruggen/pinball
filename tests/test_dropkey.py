"""DropKey score_mod for the flex path (CPU; the score_mod is called directly, no kernel)."""
import torch

from pinball.model.layers.hierarchical_message_passing import (
    _DROPKEY_BITS, _dropkey_hash, _dropkey_score_mod,
)


def _grid(n=512, b=2, h=4):
    return (torch.arange(b, dtype=torch.int32).view(b, 1, 1, 1),
            torch.arange(h, dtype=torch.int32).view(1, h, 1, 1),
            torch.arange(n, dtype=torch.int32).view(1, 1, n, 1),
            torch.arange(n, dtype=torch.int32).view(1, 1, 1, n))


def test_hash_is_uniform_and_seed_dependent():
    b, h, qi, ki = _grid()
    r1 = _dropkey_hash(b, h, qi, ki, torch.tensor(1, dtype=torch.int32))
    r2 = _dropkey_hash(b, h, qi, ki, torch.tensor(2, dtype=torch.int32))
    assert r1.dtype == torch.int32 and int(r1.min()) >= 0 and int(r1.max()) < (1 << _DROPKEY_BITS)
    assert abs(r1.double().mean().item() / (1 << _DROPKEY_BITS) - 0.5) < 5e-3
    # a new seed is a new mask, not a permutation of the old one
    assert (r1 != r2).float().mean().item() > 0.99


def test_drop_fraction_matches_p_and_self_is_kept():
    b, h, qi, ki = _grid()
    sc = torch.zeros(2, 4, 512, 512)
    for p in (0.05, 0.1, 0.3):
        out = _dropkey_score_mod(p, torch.tensor(7, dtype=torch.int32), kv_off=0)(sc, b, h, qi, ki)
        dropped = torch.isinf(out)
        assert abs(dropped.float().mean().item() - p) < 0.01, p
        assert not dropped.diagonal(dim1=-2, dim2=-1).any()


def test_kv_offset_protects_the_shifted_self_key():
    # K = [G prefix rows, then the N query-ordered rows]: query i's own key is i + G.
    G, n = 16, 256
    b, h = torch.zeros(1, 1, 1, 1, dtype=torch.int32), torch.zeros(1, 1, 1, 1, dtype=torch.int32)
    qi = torch.arange(n, dtype=torch.int32).view(1, 1, n, 1)
    ki = torch.arange(n + G, dtype=torch.int32).view(1, 1, 1, n + G)
    out = _dropkey_score_mod(0.9, torch.tensor(3, dtype=torch.int32), kv_off=G)(
        torch.zeros(1, 1, n, n + G), b, h, qi, ki)
    self_scores = out[0, 0, torch.arange(n), torch.arange(n) + G]
    assert torch.isfinite(self_scores).all()


def test_composes_after_base_and_p0_is_identity():
    b, h, qi, ki = _grid(n=64)
    sc = torch.randn(2, 4, 64, 64)
    base = lambda s, *_: s + 1.0
    out = _dropkey_score_mod(0.2, torch.tensor(5, dtype=torch.int32), base=base)(sc, b, h, qi, ki)
    kept = torch.isfinite(out)
    assert torch.equal(out[kept], (sc + 1.0)[kept])
    same = _dropkey_score_mod(0.0, torch.tensor(5, dtype=torch.int32))(sc, b, h, qi, ki)
    assert torch.equal(same, sc)
