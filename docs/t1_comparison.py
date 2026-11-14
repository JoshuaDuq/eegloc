"""The electrodes in the T1 against the electrodes in the EPI, on the participants of a study.

    python docs/t1_comparison.py STUDY [JOBS]

For every participant whose second-pass fit met all criteria:
- candidates in the T1: extrema of the band-passed height of the outer boundary (gel bumps, holder
  dents) above a level, within the region a cap can cover;
- how many of the fitted sites have a T1 candidate, or an EPI hole, of their own, and how many
  would for the same sites twisted about Cz to where the cap cannot be;
- the second-pass cap fit repeated with the T1 candidates in place of the EPI holes: its z value
  against twisted caps, and its distance from the fit to the EPI holes. Both fits search the same
  bounds from the same starting positions, so for reference the same distance is given for poses
  drawn at random within those bounds.
Writes <output>/t1_comparison.tsv and prints each quantity's median and range over participants.
"""

import csv
import dataclasses
import json
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from eegloc import checks, geometry, pipeline
from eegloc.cap import fit
from eegloc.config import MM_PER_PIXEL, POSE_NAMES, SECOND_BOUNDS, SECOND_SEEDS, SNAP_MM, Study
from eegloc.epi import Holes

T1_LEVEL_MM = 0.45  # least band-passed height of a candidate
T1_SPACING_MM = 10.0  # a candidate is the strongest extremum within this window
TWISTS = 50
TWIST_RANGE_DEG = (20.0, 340.0)
RANDOM_POSES = 200


def t1_candidates(head):
    """Bumps and dents of the T1 outer boundary, as the holes of a scalp map."""
    strength = np.abs(checks.band(head.bump))
    region = checks.cap_region(head.bump)
    strongest = strength == ndimage.maximum_filter(strength, size=int(T1_SPACING_MM / MM_PER_PIXEL))
    rows, cols = np.nonzero(strongest & (strength > T1_LEVEL_MM) & region)
    return Holes(cols.astype(float), rows.astype(float), strength, region)


def with_candidate(head, sites, candidates):
    """How many sites have a candidate of their own (within the snapping distance, in view)."""
    points = geometry.to_directions(candidates.x, candidates.y)
    points = points * head.floor_radius(points)[:, None]
    distance, nearest = cKDTree(points).query(sites)
    back = cKDTree(sites).query(points)[1]
    in_view = candidates.covered[geometry.nearest_pixels(sites)]
    return int(((distance < SNAP_MM) & (back[nearest] == np.arange(len(sites))) & in_view).sum())


def on_floor(head, points):
    directions = points / np.linalg.norm(points, axis=1, keepdims=True)
    return directions * head.floor_radius(directions)[:, None]


def compare(study, learned, subject):
    """One participant: the counts of sites with a candidate, and the fit to the T1 candidates."""
    record = json.loads((study.output / "second" / f"{subject}.json").read_text())
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    head, epi = prepared.head, prepared.holes["all"]
    t1 = t1_candidates(head)
    cap_layout = pipeline.learned_layout(learned, subject)
    epi_cap = pipeline.build_caps(prepared, cap_layout)["all"]
    t1_cap = pipeline.build_caps(dataclasses.replace(prepared, holes={"all": t1}), cap_layout)[
        "all"
    ]

    # The fitted sites, and the same sites twisted about Cz.
    sites = epi_cap.placed([record["pose"][name] for name in POSE_NAMES])
    cz = sites[[name.lower() for name in epi_cap.names].index("cz")]
    cz = cz / np.linalg.norm(cz)
    rng = np.random.default_rng(0)
    twisted = [
        on_floor(head, Rotation.from_rotvec(np.deg2rad(twist) * cz).apply(sites))
        for twist in rng.uniform(*TWIST_RANGE_DEG, TWISTS)
    ]

    pose, peak = fit(t1_cap, SECOND_BOUNDS, SECOND_SEEDS)
    null = checks.twisted_null(t1_cap, SECOND_BOUNDS, SECOND_SEEDS[:1])
    apart = np.linalg.norm(t1_cap.placed(pose) - sites, axis=1)
    random = [
        np.median(np.linalg.norm(epi_cap.placed(drawn) - sites, axis=1))
        for drawn in rng.uniform(*np.transpose(SECOND_BOUNDS), (RANDOM_POSES, len(SECOND_BOUNDS)))
    ]
    return dict(
        subject=subject,
        t1_candidates=len(t1.x),
        t1_sites=with_candidate(head, sites, t1),
        t1_sites_twisted=float(np.mean([with_candidate(head, turned, t1) for turned in twisted])),
        epi_holes=len(epi.x),
        epi_sites=with_candidate(head, sites, epi),
        epi_sites_twisted=float(np.mean([with_candidate(head, turned, epi) for turned in twisted])),
        t1_fit_z=float((peak - null.mean()) / null.std()),
        epi_fit_z=record["z"],
        t1_fit_to_epi_fit_median_mm=float(np.median(apart)),
        t1_fit_to_epi_fit_max_mm=float(apart.max()),
        random_pose_to_epi_fit_median_mm=float(np.median(random)),
    )


if __name__ == "__main__":
    study = Study.read(sys.argv[1])
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    learned = pipeline.lessons(study)
    passing = [
        subject
        for subject in study.subjects
        if json.loads((study.output / "second" / f"{subject}.json").read_text())["passed"]
    ]
    with Pool(jobs) as pool:
        rows = sorted(
            pool.imap_unordered(partial(compare, study, learned), passing),
            key=lambda row: row["subject"],
        )
    with open(study.output / "t1_comparison.tsv", "w", newline="") as file:
        writer = csv.DictWriter(file, rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} participants; median (range) over participants:")
    for key in rows[0]:
        if key != "subject":
            values = [row[key] for row in rows]
            print(f"  {key}: {np.median(values):.1f} ({min(values):.1f} to {max(values):.1f})")
    print(f"  T1 fits with z > 3: {sum(row['t1_fit_z'] > 3 for row in rows)}")
