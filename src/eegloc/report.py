"""What a pass writes: each participant's positions and figure, and the cohort's summary."""

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from eegloc import geometry  # noqa: E402
from eegloc.cap import Cap  # noqa: E402
from eegloc.config import (  # noqa: E402
    PASS_HALVES_MM,
    PASS_RETEST_MM,
    PASS_T1_P,
    PASS_Z,
    POSE_NAMES,
)
from eegloc.epi import Holes  # noqa: E402
from eegloc.geometry import Array  # noqa: E402


def write_electrodes(
    path: Path,
    names: Sequence[str],
    positions: Array,
    has_hole: NDArray[np.bool_],
    roles: Sequence[str],
) -> None:
    """Positions in the T1's scanner coordinates (mm), and where each one comes from."""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(["name", "x", "y", "z", "source", "role"])
        for name, position, from_hole, role in zip(names, positions, has_hole, roles, strict=True):
            source = "hole" if from_hole else "cap model"
            writer.writerow([name, *(f"{value:.2f}" for value in position), source, role])


def draw_fit(
    path: Path,
    record: dict,
    cap: Cap,
    holes: Holes,
    sites: Array,
    has_hole: NDArray[np.bool_],
    start: Array,
) -> None:
    """The holes on the scalp map, the fitted cap over them and the starting positions."""
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(
        np.where(holes.covered, holes.depth, np.nan),
        origin="lower",
        cmap="gray_r",
        vmin=-0.4,
        vmax=0.6,
    )
    ax.scatter(holes.x, holes.y, s=420, facecolors="none", edgecolors="yellow", linewidths=1)
    ax.scatter(*_pixels(start), s=30, c="cyan", marker="+")
    x, y = _pixels(sites)
    ax.scatter(x, y, s=50, c=np.where(has_hole, "lime", "red"), marker="x")
    for xi, yi, name in zip(x, y, cap.names, strict=True):
        ax.text(xi + 6, yi + 6, name, color="red", fontsize=8)
    ax.set_title(
        f"{record['subject']}: EPI scalp holes (dark, circled); fitted cap (x; green = takes a "
        f"hole's position); cyan +: starting positions\n"
        f"z {record['z']:.1f}, halves {record['halves_median_mm']:.1f} mm, "
        f"retest {record['retest_median_mm']:.1f} mm, "
        f"matched {record['matched']}/{record['in_view']}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def _pixels(points: Array) -> tuple[Array, Array]:
    return geometry.to_pixels(points / np.linalg.norm(points, axis=1, keepdims=True))


def summarize(folder: Path) -> str:
    """Write the pass's summary.tsv, one row per participant, and describe the cohort's checks."""
    records = [json.loads(path.read_text()) for path in sorted(folder.glob("[!.]*.json"))]
    rows = [
        dict(
            subject=record["subject"],
            holes=record["holes"],
            in_view=record["in_view"],
            matched=record["matched"],
            matched_median_mm=record["matched_median_mm"],
            z=record["z"],
            halves_median_mm=record["halves_median_mm"],
            halves_max_mm=record["halves_max_mm"],
            retest_median_mm=record["retest_median_mm"],
            t1_p=record["t1"]["p"],
            **{name: record["pose"][name] for name in POSE_NAMES},
            at_bound=int(any(record["at_bound"])),
            moved_from_start_median_mm=record["moved_from_start_median_mm"],
            moved_from_start_max_mm=record["moved_from_start_max_mm"],
            passed=int(record["passed"]),
        )
        for record in records
    ]
    with open(folder / "summary.tsv", "w", newline="") as file:
        writer = csv.DictWriter(file, rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    checks = {
        f"z > {PASS_Z:g}": [row["z"] > PASS_Z for row in rows],
        f"halves < {PASS_HALVES_MM:g} mm": [
            row["halves_median_mm"] < PASS_HALVES_MM for row in rows
        ],
        f"retest < {PASS_RETEST_MM:g} mm": [
            row["retest_median_mm"] < PASS_RETEST_MM for row in rows
        ],
        f"T1 p < {PASS_T1_P:g}": [row["t1_p"] < PASS_T1_P for row in rows],
        "no parameter at a bound": [not row["at_bound"] for row in rows],
    }
    failed = {
        row["subject"]: [name for name, holds in checks.items() if not holds[i]]
        for i, row in enumerate(rows)
        if not row["passed"]
    }
    counts = "; ".join(f"{name}: {sum(holds)}" for name, holds in checks.items())
    lines = [
        f"{len(rows)} participants. {counts}; all five: {len(rows) - len(failed)}",
        *(f"  {subject} failed: {', '.join(names)}" for subject, names in failed.items()),
        f"summary: {folder / 'summary.tsv'}",
    ]
    return "\n".join(lines)
