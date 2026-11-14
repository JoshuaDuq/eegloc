"""Electrode positions of a cohort, in two passes.

First pass, per participant:
1. T1 (cap on): the head's outer boundary as a map about a centre in the head, and its floor.
2. Mean EPI (each run's reference image through its transform to the T1): the scalp layer, where
   each electrode leaves a hole.
3. The cap, laid out by arc length over the scalp from the montage's layout, fitted to the holes.
4. Checks (checks.py).

Second pass: every participant refitted with the cap layout learned from the others who passed
the first pass (layout.py), a narrower search set by what first passes find, run from two seeds.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eegloc import checks, layout, report
from eegloc.cap import Bounds, Cap, Layout, fit, montage_layout
from eegloc.config import (
    FIRST_BOUNDS,
    FIRST_SEEDS,
    LAYOUT_MAXIMUM_STRETCH,
    PASS_HALVES_MM,
    PASS_RETEST_MM,
    PASS_T1_P,
    PASS_Z,
    POSE_NAMES,
    SECOND_BOUNDS,
    SECOND_SEEDS,
    SNAP_MM,
    Inputs,
    Study,
)
from eegloc.epi import Holes, find_holes, run_shell, scalp_layer
from eegloc.geometry import Array
from eegloc.head import Head, read_head

RUNS = ("all", "early", "late")  # the holes of all runs, and of each half for the retest


@dataclass(frozen=True)
class Prepared:
    """What the images give one participant's fits: the scalp, the holes and the start."""

    head: Head
    holes: dict[str, Holes]  # by RUNS
    start_names: tuple[str, ...]
    start_mm: Array  # scanner coordinates

    def save(self, path: Path) -> None:
        arrays = {f"head_{key}": value for key, value in vars(self.head).items()}
        for runs, holes in self.holes.items():
            arrays |= {f"{runs}_{key}": value for key, value in vars(holes).items()}
        np.savez_compressed(
            path, start_names=np.array(self.start_names), start_mm=self.start_mm, **arrays
        )

    @classmethod
    def load(cls, path: Path) -> "Prepared":
        with np.load(path) as file:
            head = Head(**{key: file[f"head_{key}"] for key in Head.__dataclass_fields__})
            holes = {
                runs: Holes(**{key: file[f"{runs}_{key}"] for key in Holes.__dataclass_fields__})
                for runs in RUNS
            }
            return cls(head, holes, tuple(file["start_names"].tolist()), file["start_mm"])


def read_start(path: Path) -> tuple[tuple[str, ...], Array]:
    """Starting positions: a tab-separated table of name, x, y, z in the T1's scanner mm."""
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if not rows or not {"name", "x", "y", "z"} <= set(rows[0]):
        raise ValueError(f"{path}: expected tab-separated columns name, x, y, z")
    names = tuple(row["name"] for row in rows)
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: a channel is named more than once")
    return names, np.array([[float(row[axis]) for axis in "xyz"] for row in rows])


def prepare(inputs: Inputs) -> Prepared:
    """The scalp from the T1 and the holes from the EPI runs."""
    head = read_head(inputs.t1)
    shells = [
        run_shell(head, epi, xfm) for epi, xfm in zip(inputs.epis, inputs.transforms, strict=True)
    ]
    half = len(shells) // 2
    holes = {
        "all": find_holes(scalp_layer(shells)),
        "early": find_holes(scalp_layer(shells[:half])),
        "late": find_holes(scalp_layer(shells[half:])),
    }
    return Prepared(head, holes, *read_start(inputs.start))


def site_names(study: Study, start_names: tuple[str, ...]) -> tuple[str, ...]:
    """Every site on the cap: the channels of the start, then the reference and the ground."""
    repeated = set(study.extra_sites) & set(start_names)
    if repeated:
        raise ValueError(f"{sorted(repeated)} is both a channel of the start and an extra site")
    return (*start_names, *study.extra_sites)


def build_caps(prepared: Prepared, cap_layout: Layout) -> dict[str, Cap]:
    """The cap on the participant's scalp, against the holes of each set of runs."""
    start = prepared.head.from_scanner(prepared.start_mm)
    index = {name.lower(): i for i, name in enumerate(prepared.start_names)}
    missing = [name for name in ("Cz", "C3", "C4") if name.lower() not in index]
    if missing:
        raise ValueError(f"the starting positions lack {missing}")
    frame = {name: start[index[name.lower()]] for name in ("Cz", "C3", "C4")}
    return {
        runs: Cap(prepared.head, holes, cap_layout, frame) for runs, holes in prepared.holes.items()
    }


def locate(
    study: Study,
    subject: str,
    prepared: Prepared,
    cap_layout: Layout,
    bounds: Bounds,
    seeds: tuple[int, ...],
    folder: Path,
) -> dict:
    """Fit one participant's cap, check the fit and write the positions, record and figure."""
    caps = build_caps(prepared, cap_layout)
    cap, head = caps["all"], prepared.head
    pose, peak = fit(cap, bounds, seeds)
    sites = cap.placed(pose)
    # A site takes its hole's position where it has one; elsewhere it keeps the cap model's.
    has_hole, nearest = cap.matched(sites, SNAP_MM)
    final = np.where(has_hole[:, None], cap.hole_points[nearest], sites)
    to_hole = np.linalg.norm(sites - cap.hole_points[nearest], axis=1)

    null = checks.twisted_null(cap, bounds, seeds[:1])
    halves = checks.split_halves(cap, bounds, seeds)
    gaps = checks.retest(caps["early"], caps["late"], bounds, seeds)
    start = head.from_scanner(prepared.start_mm)
    final_mm = head.to_scanner(final)
    moved = np.linalg.norm(final_mm[: len(start)] - prepared.start_mm, axis=1)
    record = dict(
        subject=subject,
        holes=len(cap.hole_points),
        pose=dict(zip(POSE_NAMES, map(float, pose), strict=True)),
        at_bound=[
            bool(abs(value - low) < 1e-3 or abs(value - high) < 1e-3)
            for value, (low, high) in zip(pose, bounds, strict=True)
        ],
        score=float(peak),
        null_scores=null.tolist(),
        null_mean=float(null.mean()),
        null_sd=float(null.std()),
        null_max=float(null.max()),
        z=float((peak - null.mean()) / null.std()),
        in_view=int(cap.in_view(sites).sum()),
        matched=int(has_hole.sum()),
        matched_median_mm=float(np.median(to_hole[has_hole])) if has_hole.any() else None,
        halves_mm=halves.tolist(),
        halves_median_mm=float(np.median(halves)),
        halves_max_mm=float(halves.max()),
        retest_median_mm=float(np.median(gaps)),
        retest_max_mm=float(gaps.max()),
        t1=checks.t1_evidence(cap, pose, start),
        moved_from_start_median_mm=float(np.median(moved)),
        moved_from_start_max_mm=float(moved.max()),
    )
    record["passed"] = passed(record)

    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{subject}.json").write_text(json.dumps(record, indent=1))
    roles = ["channel"] * len(start) + list(study.extra_sites.values())
    report.write_electrodes(
        folder / f"{subject}_space-ScanRAS_electrodes.tsv", cap.names, final_mm, has_hole, roles
    )
    report.draw_fit(
        folder / f"{subject}.png", record, cap, prepared.holes["all"], sites, has_hole, start
    )
    return record


def passed(record: dict) -> bool:
    """Whether a fit passes all of its checks."""
    return bool(
        record["z"] > PASS_Z
        and record["halves_median_mm"] < PASS_HALVES_MM
        and record["retest_median_mm"] < PASS_RETEST_MM
        and record["t1"]["p"] < PASS_T1_P
        and not any(record["at_bound"])
    )


def teaches(record: dict) -> bool:
    """Whether a first-pass fit is sound enough to teach the cap's layout."""
    return bool(
        record["z"] > PASS_Z
        and record["retest_median_mm"] < PASS_RETEST_MM
        and record["t1"]["p"] < PASS_T1_P
        and not any(record["at_bound"])
        and record["pose"]["stretch_lr"] < LAYOUT_MAXIMUM_STRETCH
        and record["pose"]["stretch_ap"] < LAYOUT_MAXIMUM_STRETCH
    )


def first_pass(study: Study, subject: str) -> dict:
    """One participant from the images, with the montage's layout."""
    prepared = prepare(study.inputs(subject))
    (study.output / "maps").mkdir(parents=True, exist_ok=True)
    prepared.save(study.output / "maps" / f"{subject}.npz")
    cap_layout = montage_layout(site_names(study, prepared.start_names), study.montage)
    folder = study.output / "first"
    return locate(study, subject, prepared, cap_layout, FIRST_BOUNDS, FIRST_SEEDS, folder)


@dataclass(frozen=True)
class Lessons:
    """What the first pass teaches about the cap: each teaching participant's site offsets."""

    layout: Layout  # the montage's
    subjects: tuple[str, ...]
    east: Array  # participants by sites
    north: Array


def lessons(study: Study) -> Lessons:
    """The site offsets of every participant whose first-pass fit can teach the layout."""
    subjects, east, north, layouts = [], [], [], []
    for subject in study.subjects:
        record = json.loads((study.output / "first" / f"{subject}.json").read_text())
        prepared = Prepared.load(study.output / "maps" / f"{subject}.npz")
        layouts.append(montage_layout(site_names(study, prepared.start_names), study.montage))
        if teaches(record):
            cap = build_caps(prepared, layouts[-1])["all"]
            offsets = layout.offsets(cap, [record["pose"][name] for name in POSE_NAMES])
            subjects.append(subject)
            east.append(offsets[0])
            north.append(offsets[1])
    if any(other.names != layouts[0].names for other in layouts):
        raise ValueError("the participants' starting positions do not name the same channels")
    sites = len(layouts[0].names)
    return Lessons(
        layouts[0],
        tuple(subjects),
        np.array(east).reshape(-1, sites),
        np.array(north).reshape(-1, sites),
    )


def learned_layout(learned: Lessons, subject: str) -> Layout:
    """The layout a participant is refitted with: learned from the other participants."""
    others = np.array([other != subject for other in learned.subjects], dtype=bool)
    return layout.learn(learned.layout, learned.east[others], learned.north[others])


def second_pass(study: Study, learned: Lessons, subject: str) -> dict:
    """One participant refitted with the layout learned from the other participants."""
    prepared = Prepared.load(study.output / "maps" / f"{subject}.npz")
    cap_layout = learned_layout(learned, subject)
    folder = study.output / "second"
    return locate(study, subject, prepared, cap_layout, SECOND_BOUNDS, SECOND_SEEDS, folder)
