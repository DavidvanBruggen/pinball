"""BatchedMuon == pytorch_optimizer.Muon (up to GEMM accumulation order), CPU."""
import copy

import pytest
import torch

po = pytest.importorskip("pytorch_optimizer")
from pinball.train.muon_batched import BatchedMuon  # noqa: E402


def _model():
    torch.manual_seed(0)
    return torch.nn.ModuleDict({
        "a": torch.nn.Linear(48, 48), "b": torch.nn.Linear(48, 48), "c": torch.nn.Linear(48, 96),
        "d": torch.nn.Linear(96, 48), "e": torch.nn.Linear(96, 48), "n": torch.nn.LayerNorm(48),
        "t": torch.nn.Linear(48, 48),
    })


def _groups(m, wd):
    mat = [p for n, p in m.named_parameters() if p.ndim >= 2]
    vec = [p for n, p in m.named_parameters() if p.ndim < 2]
    return [dict(params=mat, lr=2e-2, weight_decay=wd, use_muon=True),
            dict(params=vec, lr=1e-3, weight_decay=0.0, use_muon=False)]


def _grads(m, step):
    g = torch.Generator().manual_seed(step)
    for p in m.parameters():
        p.grad = torch.randn(p.shape, generator=g) * 0.1


@pytest.mark.parametrize("wd", [0.0, 0.1])
def test_matches_muon(wd):
    m1 = _model(); m2 = copy.deepcopy(m1)
    o1, o2 = po.Muon(_groups(m1, wd)), BatchedMuon(_groups(m2, wd))
    for s in range(5):
        _grads(m1, s); _grads(m2, s)
        o1.step(); o2.step()
    for (n, a), b in zip(m1.named_parameters(), m2.parameters()):
        step_size = 2e-2 * 0.2 * (96 ** 0.5)
        assert (a - b).abs().max().item() < 0.05 * step_size, n
        assert not torch.equal(a, _model()[n.split(".")[0]].get_parameter(n.split(".")[1])), n


def test_state_roundtrip():
    """A Muon checkpoint resumes in BatchedMuon exactly as in Muon (same state layout).
    Each optimizer gets its OWN deep copy: torch's load_state_dict keeps references when
    dtype/device already match, so two optimizers loaded from one in-memory state_dict share
    momentum buffers and corrupt each other (a from-disk resume never does)."""
    m1 = _model()
    o1 = po.Muon(_groups(m1, 0.1))
    _grads(m1, 0); o1.step()
    m2, m3 = copy.deepcopy(m1), copy.deepcopy(m1)
    o2 = BatchedMuon(_groups(m2, 0.1)); o2.load_state_dict(copy.deepcopy(o1.state_dict()))
    o3 = po.Muon(_groups(m3, 0.1)); o3.load_state_dict(copy.deepcopy(o1.state_dict()))
    _grads(m2, 1); _grads(m3, 1); o2.step(); o3.step()
    for a, b in zip(m2.parameters(), m3.parameters()):
        assert torch.equal(a, b)
