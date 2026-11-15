"""Repeatability of the positions across the runs of a session, on the participants of a study.

    python docs/run_retest.py STUDY [JOBS]

The test-retest check of a fit compares the first half of the runs with the second half, and
compares the sites as the cap model places them. Here the second-pass fit is repeated on the holes
of each run alone, and the positions as reported (a hole's position where the site has one) are
compared as well. For every participant:
- between two runs: the distance between the positions of a site, as placed by the cap model and
  as reported; the sites that have a hole in both runs, and the distance between those two holes;
  the holes of one run that are also a hole of the other, whatever the cap (mutual nearest
  neighbours within the snapping distance), and their distance;
- between one run and all runs: the distance between the reported positions;
- for reference, the distance of poses drawn at random within the bounds of the search from the
  fit to all runs.
Across participants: the fraction of the positions within 3 and 5 mm of the other run's; the
standard deviation across runs of a site according to whether it has a hole in every run, in none
or in some; the distances between consecutive runs and between runs further apart; and the
intraclass correlation of single runs, ICC(1,1), of each parameter of the pose and of each
channel's displacement from its starting position.

The runs share the T1 (the scalp and the centre of the head), the starting positions, the bounds
and the cap layout, which was learned from the other participants. They do not share the EPI
reference image or its transform to the T1. The cap stays on the head between the runs: this
measures how well the positions repeat, not how close they are to the electrodes.

Writes <output>/run_retest.tsv (one row per participant) and <output>/run_retest.npz (the poses
and positions of every run), and prints each quantity's median and range over the participants
whose second-pass fit met all criteria.
"""

import csv
import dataclasses
import json
import sys
from functools import partial
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

from eegloc import pipeline
from eegloc.cap import fit
from eegloc.config import POSE_NAMES, SECOND_BOUNDS, SECOND_SEEDS, SNAP_MM, Study
from eegloc.epi import find_holes, run_shell, scalp_layer

RANDOM_POSES = 200
WITHIN_MM = (3.0, 5.0)
FAR_APART = 3  # runs with at least two others between them


def positions(cap, pose):
    """The sites as the cap model places them, as reported, and which of them have a hole."""
    sites = cap.placed(pose)
    has_hole, nearest = cap.matched(sites, SNAP_MM)
    return sites, np.where(has_hole[:, None], cap.hole_points[nearest], sites), has_hole


def between_runs(positions):
    """Distance between the positions of each site in every pair of runs (pairs by sites)."""
    first, second = np.triu_indices(len(positions), 1)
    return np.linalg.norm(positions[first] - positions[second], axis=2)


def repeated(first, second):
    """Holes of two runs that are the same hole, and how far apart the two detections are."""
    distance, nearest = cKDTree(second).query(first)
    back = cKDTree(first).query(second)[1]
    same = (back[nearest] == np.arange(len(first))) & (distance < SNAP_MM)
    return 2 * same.sum() / (len(first) + len(second)), distance[same]


def run_fits(study, learned, subject):
    """One participant: the second-pass fit to the holes of each run alone."""
    record = json.loads((study.output / "second" / f"{subject}.json").read_text())
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    inputs = study.inputs(subject)
    holes = {"all": prepared.holes["all"]}
    for run, (epi, transform) in enumerate(zip(inputs.epis, inputs.transforms, strict=True)):
        holes[run] = find_holes(scalp_layer([run_shell(prepared.head, epi, transform)]))
    runs = range(len(inputs.epis))
    caps = pipeline.build_caps(
        dataclasses.replace(prepared, holes=holes), pipeline.learned_layout(learned, subject)
    )

    all_pose = [record["pose"][name] for name in POSE_NAMES]
    all_model, all_reported, _ = positions(caps["all"], all_pose)
    poses = np.array([fit(caps[run], SECOND_BOUNDS, SECOND_SEEDS)[0] for run in runs])
    model, reported, has_hole = (
        np.array(values)
        for values in zip(*(positions(caps[run], poses[run]) for run in runs), strict=True)
    )

    pairs = list(combinations(runs, 2))
    model_apart, reported_apart = between_runs(model), between_runs(reported)
    both = np.array([has_hole[a] & has_hole[b] for a, b in pairs])
    either = np.array([has_hole[a] | has_hole[b] for a, b in pairs])
    holes_shared, holes_apart = zip(
        *(repeated(caps[a].hole_points, caps[b].hole_points) for a, b in pairs), strict=True
    )
    rng = np.random.default_rng(0)
    random = [
        np.median(np.linalg.norm(caps["all"].placed(drawn) - all_model, axis=1))
        for drawn in rng.uniform(*np.transpose(SECOND_BOUNDS), (RANDOM_POSES, len(SECOND_BOUNDS)))
    ]
    start = prepared.head.from_scanner(prepared.start_mm)
    row = dict(
        subject=subject,
        passed=int(record["passed"]),
        runs=len(runs),
        holes=float(np.median([len(caps[run].hole_points) for run in runs])),
        sites_with_hole=float(np.median(has_hole.sum(axis=1))),
        model_between_runs_median_mm=float(np.median(np.median(model_apart, axis=1))),
        model_between_runs_worst_pair_mm=float(np.median(model_apart, axis=1).max()),
        reported_between_runs_median_mm=float(np.median(np.median(reported_apart, axis=1))),
        reported_between_runs_worst_pair_mm=float(np.median(reported_apart, axis=1).max()),
        reported_between_runs_p90_mm=float(np.percentile(reported_apart, 90)),
        sites_with_hole_in_both_runs=float(np.median(both.sum(axis=1) / either.sum(axis=1))),
        same_site_holes_apart_median_mm=float(np.median(reported_apart[both])),
        holes_in_both_runs=float(np.median(holes_shared)),
        holes_apart_median_mm=float(np.median(np.concatenate(holes_apart))),
        reported_run_to_all_runs_median_mm=float(
            np.median(np.median(np.linalg.norm(reported - all_reported, axis=2), axis=1))
        ),
        random_pose_to_all_runs_median_mm=float(np.median(random)),
    )
    arrays = dict(
        poses=poses,
        model=model,
        reported=reported,
        has_hole=has_hole,
        all_reported=all_reported,
        start=start,
        lag=np.array([b - a for a, b in pairs]),
        model_apart=np.median(model_apart, axis=1),
        reported_apart=np.median(reported_apart, axis=1),
    )
    return row, arrays


def icc(values):
    """ICC(1,1) of single runs and the within-participant standard deviation.

    `values` is participants by runs, or participants by runs by axes; the mean squares of the
    one-way analysis of variance are then summed over the axes.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[..., None]
    participants, runs = values.shape[:2]
    means = values.mean(axis=1, keepdims=True)
    grand = values.mean(axis=(0, 1), keepdims=True)
    between = runs * ((means - grand) ** 2).sum() / (participants - 1)
    within = ((values - means) ** 2).sum() / (participants * (runs - 1))
    return (between - within) / (between + (runs - 1) * within), np.sqrt(within)


def print_numbers(results):
    """Every number the README quotes, from the rows and arrays of all participants."""
    rows = [row for row, _ in results]
    passing = [(row, arrays) for row, arrays in results if row["passed"]]
    print(f"{len(passing)} of {len(rows)} participants; median (range) over participants:")
    for key in rows[0]:
        if key not in ("subject", "passed"):
            values = [row[key] for row, _ in passing]
            print(f"  {key}: {np.median(values):.2f} ({min(values):.2f} to {max(values):.2f})")
    for row in rows:
        if not row["passed"]:
            print(
                f"  not among them, {row['subject']}: reported positions "
                f"{row['reported_between_runs_median_mm']:.2f} mm between runs"
            )

    print("positions over sites, pairs of runs and participants:")
    for key in ("model", "reported"):
        pooled = np.concatenate([between_runs(arrays[key]).ravel() for _, arrays in passing])
        within = ", ".join(f"{(pooled < mm).mean():.3f} within {mm:g} mm" for mm in WITHIN_MM)
        print(f"  {key} (n = {len(pooled)}): {within}")

    # A site's spread across runs, the three axes combined, by how often it has a hole.
    spread = np.array(
        [np.sqrt(arrays["reported"].var(axis=0, ddof=1).sum(axis=1)) for _, arrays in passing]
    )
    with_hole = np.array([arrays["has_hole"].mean(axis=0) for _, arrays in passing])
    print("s.d. across runs of a site's reported position, median over sites and participants:")
    for label, chosen in (
        ("a hole in every run", with_hole == 1),
        ("a hole in no run", with_hole == 0),
        ("a hole in some runs", (with_hole > 0) & (with_hole < 1)),
    ):
        print(f"  {label} ({chosen.mean():.3f} of the sites): {np.median(spread[chosen]):.2f} mm")

    near, far = (
        np.array([arrays["reported_apart"][chosen(arrays["lag"])].mean() for _, arrays in passing])
        for chosen in (lambda lag: lag == 1, lambda lag: lag >= FAR_APART)
    )
    print(
        f"reported positions of consecutive runs: {np.median(near):.2f} mm; of runs {FAR_APART} "
        f"or more apart: {np.median(far):.2f} mm, more than consecutive runs in "
        f"{(far > near).sum()} of {len(near)} participants"
    )

    if len({row["runs"] for row, _ in passing}) == 1:
        print("ICC(1,1) of single runs (within-participant s.d.):")
        poses = np.array([arrays["poses"] for _, arrays in passing])
        for index, name in enumerate(POSE_NAMES):
            value, sd = icc(poses[:, :, index])
            print(f"  {name}: {value:.3f} ({sd:.3f})")
        for key in ("model", "reported"):
            # The displacement of each channel from its starting position, about the head's centre.
            moved = np.array(
                [arrays[key][:, : len(arrays["start"])] - arrays["start"] for _, arrays in passing]
            )
            per_site = np.array([icc(moved[:, :, site]) for site in range(moved.shape[2])])
            print(
                f"  {key} positions, per channel: ICC median {np.median(per_site[:, 0]):.3f} "
                f"({per_site[:, 0].min():.3f} to {per_site[:, 0].max():.3f}); "
                f"s.d. median {np.median(per_site[:, 1]):.2f} mm "
                f"({per_site[:, 1].min():.2f} to {per_site[:, 1].max():.2f})"
            )


if __name__ == "__main__":
    study = Study.read(sys.argv[1])
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    learned = pipeline.lessons(study)
    with Pool(jobs) as pool:
        results = sorted(
            pool.imap_unordered(partial(run_fits, study, learned), study.subjects),
            key=lambda result: result[0]["subject"],
        )
    rows = [row for row, _ in results]
    with open(study.output / "run_retest.tsv", "w", newline="") as file:
        writer = csv.DictWriter(file, rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        study.output / "run_retest.npz",
        subjects=np.array([row["subject"] for row in rows]),
        passed=np.array([row["passed"] for row in rows], dtype=bool),
        **{
            f"{row['subject']}_{key}": value
            for row, arrays in results
            for key, value in arrays.items()
        },
    )

    print_numbers(results)
