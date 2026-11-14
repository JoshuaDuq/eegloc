"""The cap's real layout, learned from the participants whose fit passed its checks.

A standard montage gives each site an angle from Cz and a bearing on an ideal sphere. The same
physical cap was worn by everyone, so where a site's hole falls relative to the fitted model, in
the cap's own coordinates, repeats across participants where the ideal layout is off. The
correction of a site is the median of that offset over participants, made left-right symmetric
(the cap is), and applied where enough participants contribute.
"""

from collections.abc import Sequence

import numpy as np

from eegloc.cap import Cap, Layout
from eegloc.config import (
    LAYOUT_MATCH_MM,
    LAYOUT_MINIMUM_PARTICIPANTS,
    NOMINAL_RADIUS_MM,
    SECTION_ANGLES,
)
from eegloc.geometry import Array


def mirror_name(name: str) -> str:
    """The site across the midline: Fp1 and Fp2, FT9 and FT10; a midline site is its own."""
    if name.lower().endswith("z"):
        return name
    stem = name.rstrip("0123456789")
    number = int(name[len(stem) :])
    return f"{stem}{number + 1 if number % 2 else number - 1}"


def cap_coordinates(cap: Cap, pose: Sequence[float], points: Array) -> tuple[Array, Array]:
    """Arc length from Cz (mm) and bearing (radians, in the cap's frame) of points on the scalp."""
    pitch, roll, twist = pose[:3]
    cz, right, front = cap.frame(pitch, roll)
    directions = points / np.linalg.norm(points, axis=1, keepdims=True)
    cosine = np.clip(directions @ cz, -1, 1)
    tangents = directions - cosine[:, None] * cz
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    bearing = np.arctan2(tangents @ front, tangents @ right) - np.deg2rad(twist)
    arcs = cap.arc_lengths(cz, tangents)
    arc = np.array(
        [np.interp(a, SECTION_ANGLES, row) for a, row in zip(np.arccos(cosine), arcs, strict=True)]
    )
    return arc, bearing


def offsets(cap: Cap, pose: Sequence[float]) -> tuple[Array, Array]:
    """How far each site's hole lies from the fitted site, on the cap's sphere.

    Radians toward the right ear and toward the nose; NaN for a site without a hole of its own.
    """
    has_hole, nearest = cap.matched(cap.placed(pose), LAYOUT_MATCH_MM)
    arc, bearing = cap_coordinates(cap, pose, cap.hole_points[nearest])
    stretch = pose[3] * np.cos(cap.bearing) ** 2 + pose[4] * np.sin(cap.bearing) ** 2
    d_angle = arc / (NOMINAL_RADIUS_MM * stretch) - cap.angle
    d_bearing = np.angle(np.exp(1j * (bearing - cap.bearing)))
    d_angle[~has_hole], d_bearing[~has_hole] = np.nan, np.nan
    # At Cz itself a bearing means nothing: each offset is a displacement in the cap's plane.
    east = d_angle * np.cos(cap.bearing) - cap.angle * d_bearing * np.sin(cap.bearing)
    north = d_angle * np.sin(cap.bearing) + cap.angle * d_bearing * np.cos(cap.bearing)
    return east, north


def learn(layout: Layout, east: Array, north: Array) -> Layout:
    """The layout corrected by the participants' offsets (participants by sites)."""
    count = np.isfinite(east).sum(axis=0)
    enough = count >= LAYOUT_MINIMUM_PARTICIPANTS
    e, n = np.zeros(len(layout.names)), np.zeros(len(layout.names))
    e[enough] = np.nanmedian(east[:, enough], axis=0)
    n[enough] = np.nanmedian(north[:, enough], axis=0)
    index = {name: i for i, name in enumerate(layout.names)}
    e_symmetric, n_symmetric = e.copy(), n.copy()
    for name, i in index.items():
        j = index[mirror_name(name)]
        if enough[i] and enough[j]:
            e_symmetric[i], n_symmetric[i] = (e[i] - e[j]) / 2, (n[i] + n[j]) / 2
        if i == j:
            e_symmetric[i] = 0.0
    x = layout.angle * np.cos(layout.bearing) + e_symmetric
    y = layout.angle * np.sin(layout.bearing) + n_symmetric
    return Layout(layout.names, np.hypot(x, y), np.arctan2(y, x))
