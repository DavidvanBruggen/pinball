"""Space-filling-curve orderings for the spatial hierarchy ("curve mode").

The model reorders L0 tokens ONCE by a space-filling curve right after the token
stem (and inverse-permutes outputs back to raster). The existing sequence-mode
1D contiguous-window machinery (pooled seed, upward/downward refresh, packed
cross-level local attention, level sizes) then pools contiguous CURVE ranges,
which are spatially compact blocks — exactly quadtree blocks at compression 4,
overlap 0, power-of-2 grids. See configs/pinball_image_maskgit_*.yaml.

Conventions
-----------
- Raster index is row-major over ``dims``: for ``dims=(H, W)`` the pixel at
  (y, x) has raster index ``y*W + x``; generally ``idx = sum(c[d]*stride[d])``
  with ``stride[-1] == 1``.
- ``perm[i]``   = raster index of the i-th CURVE position
  (``x_curve = x.index_select(1, perm)``).
- ``inv_perm[r]`` = curve position of raster index r
  (``x_raster = y.index_select(1, inv_perm)``; ``inv_perm[perm] == arange``).
- ``coords[i]`` = ND coordinate of curve position i (CURVE order), rows ordered
  like ``dims`` (i.e. (y, x) for 2D — note _gilbert2d itself yields (x, y)).
"""

from __future__ import annotations

import functools
import logging
from typing import Iterator, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = ["build_curve"]


def _sgn(v: int) -> int:
    return (v > 0) - (v < 0)


def _gilbert2d(width: int, height: int) -> Iterator[Tuple[int, int]]:
    """Generalized Hilbert ("gilbert") curve for arbitrary W x H rectangles.

    Port of the public-domain algorithm by Jakub Červený
    (github.com/jakubcerveny/gilbert). Yields (x, y) in curve order; every
    consecutive pair is 4-adjacent (|dx|+|dy| == 1), which is the property the
    verification probe asserts.
    """

    def generate(x: int, y: int, ax: int, ay: int, bx: int, by: int):
        w = abs(ax + ay)
        h = abs(bx + by)
        dax, day = _sgn(ax), _sgn(ay)  # unit major direction
        dbx, dby = _sgn(bx), _sgn(by)  # unit orthogonal direction

        if h == 1:
            for _ in range(w):
                yield (x, y)
                x, y = x + dax, y + day
            return
        if w == 1:
            for _ in range(h):
                yield (x, y)
                x, y = x + dbx, y + dby
            return

        ax2, ay2 = ax // 2, ay // 2
        bx2, by2 = bx // 2, by // 2
        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)

        if 2 * w > 3 * h:
            if (w2 % 2) and (w > 2):
                ax2, ay2 = ax2 + dax, ay2 + day  # prefer even steps
            yield from generate(x, y, ax2, ay2, bx, by)
            yield from generate(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
        else:
            if (h2 % 2) and (h > 2):
                bx2, by2 = bx2 + dbx, by2 + dby
            yield from generate(x, y, bx2, by2, ax2, ay2)
            yield from generate(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
            yield from generate(
                x + (ax - dax) + (bx2 - dbx),
                y + (ay - day) + (by2 - dby),
                -bx2,
                -by2,
                -(ax - ax2),
                -(ay - ay2),
            )

    if width >= height:
        yield from generate(0, 0, width, 0, 0, height)
    else:
        yield from generate(0, 0, 0, height, width, 0)


def _morton_perm(dims: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    """(perm, coords) by stable-sorting all raster coords on the interleaved-bit
    Morton (Z-order) key. Key-sorting the ACTUAL coords handles non-power-of-2
    dims without padding: skipped cells simply never appear."""
    nd = len(dims)
    grids = torch.meshgrid(*[torch.arange(int(d)) for d in dims], indexing="ij")
    coords_raster = torch.stack([g.reshape(-1) for g in grids], dim=-1)  # [T, nd] raster order
    nbits = max(int(d - 1).bit_length() for d in dims) if max(dims) > 1 else 1
    key = torch.zeros(coords_raster.size(0), dtype=torch.long)
    for bit in range(nbits):
        for axis in range(nd):
            key |= ((coords_raster[:, axis] >> bit) & 1) << (bit * nd + axis)
    perm = torch.argsort(key, stable=True)
    return perm, coords_raster.index_select(0, perm)


@functools.lru_cache(maxsize=64)
def build_curve(
    dims: Tuple[int, ...], curve: str = "hilbert"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(perm [T], inv_perm [T], coords [T, ndim])`` CPU long tensors.

    curve:
      - "none"/"raster": identity perm, raster coords.
      - "hilbert": ndim==1 identity; ndim==2 gilbert (any rectangle);
        ndim>=3 falls back to Morton (logged once per shape via this cache).
      - "morton": Z-order for any ndim.
    """
    dims = tuple(int(d) for d in dims)
    if len(dims) == 0 or any(d <= 0 for d in dims):
        raise ValueError(f"build_curve: invalid dims {dims}")
    curve = str(curve).lower()
    total = 1
    for d in dims:
        total *= d
    nd = len(dims)

    if curve in ("none", "raster") or nd == 1:
        perm = torch.arange(total, dtype=torch.long)
        grids = torch.meshgrid(*[torch.arange(d) for d in dims], indexing="ij")
        coords = torch.stack([g.reshape(-1) for g in grids], dim=-1)
    elif curve in ("hilbert", "gilbert"):
        if nd == 2:
            h, w = dims
            xy = list(_gilbert2d(w, h))  # yields (x, y)
            coords = torch.tensor([[y, x] for (x, y) in xy], dtype=torch.long)  # rows (y, x)
            perm = coords[:, 0] * w + coords[:, 1]
        else:
            logger.info(
                "build_curve: hilbert requested for %dD dims %s; using Morton fallback.",
                nd, dims,
            )
            perm, coords = _morton_perm(dims)
    elif curve == "morton":
        perm, coords = _morton_perm(dims)
    else:
        raise ValueError(f"build_curve: unknown curve '{curve}'")

    if int(perm.numel()) != total:
        raise RuntimeError(
            f"build_curve: curve '{curve}' covered {int(perm.numel())} of {total} cells for dims {dims}"
        )
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(total, dtype=torch.long)
    return perm.contiguous(), inv_perm.contiguous(), coords.contiguous()
