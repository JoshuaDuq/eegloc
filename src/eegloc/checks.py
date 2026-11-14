"""Checks on one participant's cap fit.

1. Twisted-cap null: the same cap fitted at twists where it cannot be. (A standard layout is nearly
   its own mirror front to back, so twists near 180 degrees are left out.)
2. Split halves: two disjoint random halves of the sites, fitted separately, must place the cap
   alike.
3. Test-retest: the first and second halves of the runs, fitted separately, must too.
4. Independent image: the T1 gave the fit only the scalp's shape. The fitted sites must fall on the
   T1's own bumps (gel standing in a holder's hole) and dents (a holder the head lies on) more
   than the same cap twisted.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from eegloc import geometry
from eegloc.cap import Bounds, Cap, fit
from eegloc.config import (
    MM_PER_PIXEL,
    NULL_HALF_WIDTH_DEG,
    NULL_TWISTS_DEG,
    SPLIT_HALVES,
    T1_BAND_MM,
    T1_DRAWS,
    T1_HEIGHT_RANGE_MM,
    T1_JITTER_DEG,
    T1_MINIMUM_SITES,
    T1_REGION_EXTENT,
    T1_TALL_MARGIN_MM,
    T1_TALL_MM,
    T1_TWIST_RANGE_DEG,
    T1_WINDOW_MM,
)
from eegloc.geometry import Array


def twisted_null(cap: Cap, bounds: Bounds, seeds: Sequence[int]) -> Array:
    """The best score the cap reaches at each of the twists where it cannot be."""
    scores = []
    for twist in NULL_TWISTS_DEG:
        twisted = [*bounds]
        twisted[2] = (twist - NULL_HALF_WIDTH_DEG, twist + NULL_HALF_WIDTH_DEG)
        scores.append(fit(cap, twisted, seeds)[1])
    return np.array(scores)


def split_halves(cap: Cap, bounds: Bounds, seeds: Sequence[int]) -> Array:
    """Median distance (mm) between the placements from two halves of the sites, per split."""
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(SPLIT_HALVES):
        order = rng.permutation(len(cap.names))
        half = len(order) // 2
        first = cap.placed(fit(cap, bounds, seeds, order[:half])[0])
        second = cap.placed(fit(cap, bounds, seeds, order[half:])[0])
        gaps.append(np.median(np.linalg.norm(first - second, axis=1)))
    return np.array(gaps)


def retest(early: Cap, late: Cap, bounds: Bounds, seeds: Sequence[int]) -> Array:
    """Distance (mm) between each site as placed from the early runs and from the late runs."""
    first = early.placed(fit(early, bounds, seeds)[0])
    second = late.placed(fit(late, bounds, seeds)[0])
    return np.linalg.norm(first - second, axis=1)


def band(bump: Array) -> Array:
    """The outer boundary's height, band-passed to the size of a bump or a dent."""
    height = np.clip(np.nan_to_num(bump), *T1_HEIGHT_RANGE_MM)
    fine, coarse = (ndimage.gaussian_filter(height, size / MM_PER_PIXEL) for size in T1_BAND_MM)
    return fine - coarse


def cap_region(bump: Array) -> NDArray[np.bool_]:
    """The scalp a cap can cover: ears and face, far above the floor, masked with a margin."""
    tall = ndimage.binary_opening(np.nan_to_num(bump) > T1_TALL_MM, iterations=3)
    tall = ndimage.binary_dilation(tall, iterations=int(T1_TALL_MARGIN_MM / MM_PER_PIXEL))
    return (geometry.map_polar()[0] < T1_REGION_EXTENT) & ~tall & np.isfinite(bump)


def t1_evidence(cap: Cap, pose: Sequence[float], start: Array) -> dict[str, float]:
    """Bumps and dents of the T1 under the fitted sites, against the same cap twisted.

    `start` holds the starting positions about the head's centre, scored for comparison.
    """
    response = ndimage.maximum_filter(
        np.abs(band(cap.head.bump)), size=int(T1_WINDOW_MM / MM_PER_PIXEL)
    )
    region = cap_region(cap.head.bump)

    def evidence(points: Array) -> float:
        rows, cols = geometry.nearest_pixels(points)
        inside = region[rows, cols]
        if inside.sum() <= T1_MINIMUM_SITES:
            return np.nan
        return float(response[rows[inside], cols[inside]].mean())

    rng = np.random.default_rng(0)
    twisted = np.array(
        [
            evidence(
                cap.placed(
                    [
                        pose[0] + rng.uniform(-T1_JITTER_DEG, T1_JITTER_DEG),
                        pose[1] + rng.uniform(-T1_JITTER_DEG, T1_JITTER_DEG),
                        rng.uniform(*T1_TWIST_RANGE_DEG),
                        pose[3],
                        pose[4],
                    ]
                )
            )
            for _ in range(T1_DRAWS)
        ]
    )
    fitted = evidence(cap.placed(pose))
    return dict(
        fitted=fitted,
        start=evidence(start),
        null_mean=float(np.nanmean(twisted)),
        null_sd=float(np.nanstd(twisted)),
        p=float(((twisted >= fitted).sum() + 1) / (T1_DRAWS + 1)),
    )
