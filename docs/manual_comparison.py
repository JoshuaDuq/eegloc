"""Fitted positions against positions marked by hand on the T1, on a study that has them.

    python docs/manual_comparison.py STUDY MARKED COHORT [JOBS]

STUDY is the study file written by docs/scrivener_data.py, on which both passes and
docs/epi_quality.py have been run; MARKED is its folder of marked positions (Brainsight targets in
the scanner coordinates of the T1; a file of MARKED/Updated replaces that of MARKED/Original;
participant sub-<nn> is S<n>). COHORT is the study file of a cohort recorded with the same cap, on
which both passes and docs/epi_quality.py have been run.

For every participant with marked positions:
- each pass: the distance along the scalp between each site's reported position and its marked
  position, both moved along their ray from the head's centre onto the scalp floor; the same for
  the starting positions;
- with no cap: how many of the marked electrodes within the EPI's coverage have a hole within
  5 mm, against the same electrodes turned about the vertical axis;
- the second pass repeated with the layout learned on COHORT in place of the study's own
  (written to <output>/transfer, with the files of a pass);
- the cap model fitted to the marked positions themselves, with the montage's layout and with
  the layout learned on COHORT: what is left is the least distance at which the model could place
  the sites, were every electrode detected.
Writes <output>/manual_comparison.tsv and docs/figures/quality.png, and prints the numbers the
README quotes.
"""

import csv
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import make_figures as figures  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from t1_comparison import on_floor  # noqa: E402

from eegloc import geometry, pipeline  # noqa: E402
from eegloc.cap import montage_layout  # noqa: E402
from eegloc.config import (  # noqa: E402
    PASS_Z,
    SEARCH,
    SECOND_BOUNDS,
    SECOND_SEEDS,
    Study,
)

NEAR_MM = 5.0
TURNS = 200
PASSES = ("first", "second", "transfer")
LAYOUTS = ("montage", "cohort")
# The poses of the cap model fitted to marked positions are searched well beyond those of a pass.
MODEL_BOUNDS = ((-30.0, 30.0), (-20.0, 20.0), (-25.0, 25.0), (0.7, 1.3), (0.7, 1.3))


def read_marked(folder, subject):
    """Marked positions by lower-case site name (scanner mm), or None where there are none."""
    name = f"S{int(subject.split('-')[1])}_electrodes.txt"
    path = next(
        (p for p in (folder / "Updated" / name, folder / "Original" / name) if p.exists()), None
    )
    if path is None:
        return None
    marked = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            site, x, y, z = line.split("\t")[:4]
            marked[site.strip().lower()] = [float(x), float(y), float(z)]
    # S5 names the target between Fz and Cz "fc7"; the cap has no FC7 and the file has no FCz.
    if "fc7" in marked and "fcz" not in marked:
        marked["fcz"] = marked.pop("fc7")
    return marked


def marked_on_floor(head, marked, names):
    """The marked positions of the named sites on the scalp floor, about the head's centre."""
    return on_floor(head, head.from_scanner(np.array([marked[name.lower()] for name in names])))


def along_scalp(head, names, reported, marked):
    """Distance (mm) of each reported position from its marked position, both on the floor."""
    return np.linalg.norm(
        on_floor(head, head.from_scanner(reported)) - marked_on_floor(head, marked, names), axis=1
    )


def transfer(study, learned, subject):
    """The second pass with the layout learned on another cohort of the same cap."""
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    cap_layout = pipeline.learned_layout(learned, subject)
    if cap_layout.names != pipeline.site_names(study, prepared.start_names):
        raise ValueError("the two studies do not name the same sites in the same order")
    folder = study.output / "transfer"
    return pipeline.locate(
        study, subject, prepared, cap_layout, SECOND_BOUNDS, SECOND_SEEDS, folder
    )


def model_fits(study, folder, learned, subject):
    """The cap model fitted to the marked positions, with each layout: distance left per site."""
    marked = read_marked(folder, subject)
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    names = pipeline.site_names(study, prepared.start_names)
    truth = marked_on_floor(prepared.head, marked, names)
    layouts = (montage_layout(names, study.montage), pipeline.learned_layout(learned, subject))
    left = {}
    for label, cap_layout in zip(LAYOUTS, layouts, strict=True):
        cap = pipeline.build_caps(prepared, cap_layout)["all"]

        def mean_square(pose, cap=cap):
            return float(np.mean(np.sum((cap.placed(pose) - truth) ** 2, axis=1)))

        pose = differential_evolution(mean_square, MODEL_BOUNDS, seed=0, **SEARCH).x
        left[label] = np.linalg.norm(cap.placed(pose) - truth, axis=1)
    return subject, names, left


def compare(study, folder, subject):
    """One participant's passes against the marked positions, and the holes against them."""
    marked = read_marked(folder, subject)
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    head, holes = prepared.head, prepared.holes["all"]
    row = dict(subject=subject)
    for stage in PASSES:
        record = json.loads((study.output / stage / f"{subject}.json").read_text())
        names, reported, from_hole = figures.read_electrodes(
            study.output / stage / f"{subject}_space-ScanRAS_electrodes.tsv"
        )
        apart = along_scalp(head, names, reported, marked)
        row |= {
            f"{stage}_passed": int(record["passed"]),
            f"{stage}_z": record["z"],
            f"{stage}_sites_with_hole": int(from_hole.sum()),
            f"{stage}_along_scalp_median_mm": float(np.median(apart)),
            f"{stage}_along_scalp_hole_median_mm": float(np.median(apart[from_hole])),
            f"{stage}_along_scalp_model_median_mm": float(np.median(apart[~from_hole])),
        }
    row["start_along_scalp_median_mm"] = float(
        np.median(along_scalp(head, prepared.start_names, prepared.start_mm, marked))
    )

    # With no cap: marked electrodes that have a hole nearby, and the same electrodes turned.
    truth = marked_on_floor(head, marked, list(marked))
    directions = geometry.to_directions(holes.x, holes.y)
    tree = cKDTree(directions * head.floor_radius(directions)[:, None])

    def nearby(points):
        seen = holes.covered[geometry.nearest_pixels(points)]
        return float((tree.query(points[seen])[0] < NEAR_MM).mean())

    rng = np.random.default_rng(0)
    turned = [
        nearby(on_floor(head, Rotation.from_rotvec([0, 0, np.deg2rad(angle)]).apply(truth)))
        for angle in rng.uniform(20, 340, TURNS)
    ]
    row["marked_with_hole_nearby"] = nearby(truth)
    row["turned_with_hole_nearby"] = float(np.mean(turned))
    return row


def table(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def typical(rows):
    """The participant of the median hole depth."""
    return sorted(rows, key=lambda row: float(row["hole_depth"]))[len(rows) // 2]["subject"]


def show_layer(ax, study, subject, points, colour):
    """A participant's EPI signal loss on the scalp map, with points about the head's centre."""
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    holes = prepared.holes["all"]
    ax.imshow(np.where(holes.covered, holes.depth, np.nan), origin="lower", **figures.LOSS)
    x, y = figures.pixels(points(prepared.head))
    ax.scatter(x, y, s=16, facecolors="none", edgecolors=colour, linewidths=0.6)
    figures.frame_map(ax)


def strip(ax, groups, label):
    """One value per participant of each cohort, and the cohort's median."""
    rng = np.random.default_rng(0)
    for k, (values, colour) in enumerate(zip(groups, (figures.BLUE, figures.ORANGE), strict=True)):
        ax.scatter(k + rng.uniform(-0.16, 0.16, len(values)), values, s=7, c=colour, linewidths=0)
        ax.plot([k - 0.28, k + 0.28], [np.median(values)] * 2, color="black", linewidth=1.0)
    ax.set_xticks([0, 1], labels=["Cohort", "Scrivener\net al."])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(label)


def figure_quality(study, folder, cohort, rows):
    """What the EPI of two cohorts shows of the electrodes, and the error where it shows little."""
    quality = [table(each.output / "epi_quality.tsv") for each in (cohort, study)]
    with_marks = {row["subject"] for row in rows}
    cohort_subject = typical(quality[0])
    subject = typical([row for row in quality[1] if row["subject"] in with_marks])
    fig = plt.figure(figsize=(figures.WIDTH, 128 * figures.MM))
    upper = fig.add_gridspec(1, 2, wspace=0.08)
    upper.update(left=0.03, right=0.97, top=0.95, bottom=0.44)
    lower = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.6], wspace=0.62)
    lower.update(left=0.075, right=0.985, top=0.34, bottom=0.09)
    a, b = (fig.add_subplot(upper[0, k]) for k in range(2))
    c, d, e, f = (fig.add_subplot(lower[0, k]) for k in range(4))

    def fitted(head):
        path = cohort.output / "second" / f"{cohort_subject}_space-ScanRAS_electrodes.tsv"
        return head.from_scanner(figures.read_electrodes(path)[1])

    def marked(head):
        positions = read_marked(folder, subject)
        return marked_on_floor(head, positions, list(positions))

    show_layer(a, cohort, cohort_subject, fitted, figures.BLUE)
    figures.title(a, 0, "Cohort: fitted sites", shift=-2)
    show_layer(b, study, subject, marked, figures.ORANGE)
    figures.title(b, 1, "Scrivener et al.: electrodes marked by hand", shift=-2)

    panels = (
        (c, "scalp_to_cortex", "Signal relative to cortex", "Scalp signal"),
        (d, "hole_depth", "Signal lost in a hole", "Depth of the holes"),
        (e, "z", "z against twisted caps", "Fit of the cap"),
    )
    for index, (ax, key, label, name) in enumerate(panels):
        strip(ax, [[float(row[key]) for row in each] for each in quality], label)
        figures.title(ax, 2 + index, name)
    e.axhline(PASS_Z, color="black", linewidth=0.4, linestyle=(0, (4, 3)))

    start = np.array([row["start_along_scalp_median_mm"] for row in rows])
    for stage, colour in (("second", figures.ORANGE), ("transfer", figures.BLUE)):
        z = np.array([row[f"{stage}_z"] for row in rows])
        if stage == "second":
            f.scatter(
                z,
                start,
                s=9,
                marker="D",
                facecolors="white",
                edgecolors=figures.GREY,
                linewidths=0.6,
            )
        met = np.array([bool(row[f"{stage}_passed"]) for row in rows])
        figures.mark(
            f, z, np.array([row[f"{stage}_along_scalp_median_mm"] for row in rows]), met, colour
        )
    f.axvline(PASS_Z, ymax=0.6, color="black", linewidth=0.4, linestyle=(0, (4, 3)))
    model = np.median([row["model_montage_layout_median_mm"] for row in rows])
    f.axhline(model, color=figures.GREY, linewidth=0.6)
    f.set_xlabel("z against twisted caps")
    f.set_ylabel("Distance from the marked\npositions (mm)")
    f.set_ylim(0, 44)
    f.set_yticks([0, 10, 20, 30])
    open_marker = dict(linestyle="", markerfacecolor="white")
    handles = [
        plt.Line2D(
            [], [], marker="o", markersize=3.4, markeredgecolor=figures.ORANGE, **open_marker
        ),
        plt.Line2D([], [], marker="o", markersize=3.4, markeredgecolor=figures.BLUE, **open_marker),
        plt.Line2D([], [], marker="D", markersize=3, markeredgecolor=figures.GREY, **open_marker),
        plt.Line2D([], [], color=figures.GREY, linewidth=0.6),
    ]
    f.legend(
        handles,
        ["Second pass", "With the cohort's layout", "Starting positions", "Cap model on the marks"],
        loc="upper right",
        handletextpad=0.3,
        handlelength=1.2,
        borderaxespad=0.0,
    )
    figures.title(f, 5, "Error and fit of the cap")
    fig.savefig(figures.FOLDER / "quality.png")
    plt.close(fig)
    return cohort_subject, subject


def spread(values):
    return f"{np.median(values):.2f} ({np.min(values):.2f} to {np.max(values):.2f})"


if __name__ == "__main__":
    study, folder, cohort = Study.read(sys.argv[1]), Path(sys.argv[2]), Study.read(sys.argv[3])
    jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    learned = pipeline.lessons(cohort)
    if set(learned.subjects) & set(study.subjects):
        raise ValueError("a participant of the study has the name of a participant of the cohort")
    subjects = [s for s in study.subjects if read_marked(folder, s) is not None]
    with Pool(jobs) as pool:
        if not all((study.output / "transfer" / f"{s}.json").exists() for s in study.subjects):
            for record in pool.imap_unordered(partial(transfer, study, learned), study.subjects):
                print(
                    f"{record['subject']}: {'passed' if record['passed'] else 'failed'}", flush=True
                )
        models = {
            subject: (names, left)
            for subject, names, left in pool.imap_unordered(
                partial(model_fits, study, folder, learned), subjects
            )
        }
    rows = [
        compare(study, folder, subject)
        | {
            f"model_{label}_layout_median_mm": float(np.median(models[subject][1][label]))
            for label in LAYOUTS
        }
        for subject in subjects
    ]
    with open(study.output / "manual_comparison.tsv", "w", newline="") as file:
        writer = csv.DictWriter(file, rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(figures.STYLE)
    shown = figure_quality(study, folder, cohort, rows)
    print(f"{len(rows)} participants with marked positions; figure: {shown}")
    print(f"layout of the cohort learned from {len(learned.subjects)} participants")
    for key in (
        "start_along_scalp_median_mm",
        "marked_with_hole_nearby",
        "turned_with_hole_nearby",
    ):
        print(f"{key}: {spread([row[key] for row in rows])}")
    for stage in PASSES:
        z = np.array([row[f"{stage}_z"] for row in rows])
        apart = np.array([row[f"{stage}_along_scalp_median_mm"] for row in rows])
        rho, p = stats.spearmanr(z, apart)
        print(
            f"{stage}: along the scalp {spread(apart)} mm; z {spread(z)}; z > {PASS_Z:g} in "
            f"{(z > PASS_Z).sum()}"
            + (f", {spread(apart[z > PASS_Z])} mm" if (z > PASS_Z).any() else "")
            + f"; Spearman of the distance with z {rho:.2f} (p = {p:.3f})"
        )
        met = np.array([bool(row[f"{stage}_passed"]) for row in rows])
        better = (apart < np.array([row["start_along_scalp_median_mm"] for row in rows])).sum()
        print(
            f"  all criteria met in {met.sum()}"
            + (f", {spread(apart[met])} mm" if met.any() else "")
            + f"; nearer than the starting positions in {better}; sites with a hole "
            f"{spread([row[f'{stage}_sites_with_hole'] for row in rows])}"
        )
    names = models[subjects[0]][0]
    for label in LAYOUTS:
        per_site = np.median([models[subject][1][label] for subject in subjects], axis=0)
        worst = ", ".join(f"{names[k]} {per_site[k]:.1f}" for k in np.argsort(-per_site)[:4])
        print(
            f"cap model fitted to the marked positions, {label} layout: "
            f"{spread([row[f'model_{label}_layout_median_mm'] for row in rows])} mm; "
            f"sites it leaves farthest (median over participants, mm): {worst}"
        )
    depth = {
        row["subject"]: float(row["hole_depth"]) for row in table(study.output / "epi_quality.tsv")
    }
    rho, p = stats.spearmanr(
        [depth[row["subject"]] for row in rows], [row["marked_with_hole_nearby"] for row in rows]
    )
    print(
        f"Spearman of marked_with_hole_nearby with the depth of the holes: {rho:.2f} (p = {p:.3f})"
    )
