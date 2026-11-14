"""The README's figures, from the outputs of a study on which both passes have been run.

    python docs/make_figures.py STUDY SUBJECT SYNTHETIC

STUDY is the study file and SUBJECT the participant shown in the figures of one participant; the
other figures show every participant of the study. SYNTHETIC is the folder written by
examples/synthetic_participant.py, once both passes have been run on its study file. The numbers
quoted in the README and its captions are printed.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import Delaunay, cKDTree  # noqa: E402

from eegloc import checks, geometry, layout, pipeline  # noqa: E402
from eegloc.config import (  # noqa: E402
    BOUNDARY_LEVEL,
    COVERAGE_SMOOTHING_MM,
    EPI_PERCENTILE,
    FAT_PERCENTILE,
    HOLE_DEPTH,
    LAYER_OFFSETS_MM,
    LAYOUT_MINIMUM_PARTICIPANTS,
    MAP_SIZE,
    MM_PER_PIXEL,
    NOMINAL_RADIUS_MM,
    PASS_HALVES_MM,
    PASS_RETEST_MM,
    PASS_T1_P,
    PASS_Z,
    T1_SMOOTHING_MM,
    Study,
)
from eegloc.epi import ItkAffine, run_shell, scalp_layer  # noqa: E402

FOLDER = Path(__file__).parent / "figures"
MM = 1 / 25.4
WIDTH = 183 * MM
# Journal style: sans-serif text of 6-7 pt, open axes, no line under 0.25 pt.
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 5.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
}
# Okabe and Ito's colour-blind-safe hues.
BLUE, ORANGE, GREY, PALE = "#0072B2", "#D55E00", "#8C8C8C", "#D9D9D9"
YELLOW, SKY = "#E69F00", "#56B4E9"
LOSS = dict(cmap="gray_r", norm=Normalize(-0.4, 0.6))  # the EPI's signal loss, one scale throughout
SECTION_U, SECTION_V = np.arange(-125, 125.2, 0.4), np.arange(-12, 140.2, 0.4)  # mm in the plane
SURFACE_MARGIN_MM = 30.0  # the mean head is drawn within this distance of a site
SURFACE_REACH_MM = 160.0  # rays start this far from the centre of the head
OBLIQUES = {  # azimuth from the front toward the participant's right, elevation (degrees)
    "Front left": (-40, 25),
    "Front right": (40, 25),
    "Back right": (140, 25),
    "Back left": (-140, 25),
}
SECTION_HALF_WIDTH = 5.0  # sites within this distance of the plane are marked, mm
VIEWS = {  # toward the viewer, toward the image's right, toward its top
    "Left": ([-1, 0, 0], [0, -1, 0], [0, 0, 1]),
    "Top": ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
    "Back": ([0, -1, 0], [1, 0, 0], [0, 0, 1]),
    "Right": ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
}


def title(ax, index, text, shift=-10):
    """A bold lowercase panel letter and a short title."""
    ax.set_title(text, loc="left")
    ax.annotate(
        "abcdefgh"[index],
        (0, 1),
        xytext=(shift, 5),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def read_records(folder):
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def read_electrodes(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    positions = np.array([[float(row[axis]) for axis in "xyz"] for row in rows])
    names = [row["name"] for row in rows]
    return names, positions, np.array([row["source"] == "hole" for row in rows])


def pixels(points):
    return geometry.to_pixels(points / np.linalg.norm(points, axis=1, keepdims=True))


def scale_bar(ax, length, label, where=(0.04, 0.02), colour="black"):
    """A bar of `length` data units at the lower left of the axes."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x, y = x0 + where[0] * (x1 - x0), y0 + where[1] * (y1 - y0)
    ax.plot([x, x + length], [y, y], color=colour, linewidth=1.0, solid_capstyle="butt")
    ax.annotate(
        label,
        (x + length / 2, y),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        color=colour,
    )


def frame_map(ax):
    """Limits and marks of a scalp map: front at the top, the participant's left on the left."""
    middle, reach = (MAP_SIZE - 1) / 2, 0.97 * (MAP_SIZE - 1) / 2
    ax.set_xlim(middle - reach, middle + reach)
    ax.set_ylim(middle - reach, middle + reach)
    ax.set_aspect("equal")
    ax.axis("off")
    for text, where, align in (("L", (0, 0.5), "right"), ("R", (1, 0.5), "left")):
        ax.annotate(text, where, xycoords="axes fraction", ha=align, va="center", color=GREY)
    scale_bar(ax, 50 / MM_PER_PIXEL, "50 mm")


def colour_bar(fig, ax, shown, label, ticks):
    inset = ax.inset_axes([0.3, -0.06, 0.4, 0.025])
    bar = fig.colorbar(shown, cax=inset, orientation="horizontal", ticks=ticks)
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(length=1.5, width=0.4, pad=1.5)
    bar.set_label(label, labelpad=2)


def cap_frame(names, positions):
    """Unit vectors toward the fitted Cz and along the cap from C3 to C4."""
    index = {name.lower(): i for i, name in enumerate(names)}
    cz = positions[index["cz"]] / np.linalg.norm(positions[index["cz"]])
    across = positions[index["c4"]] - positions[index["c3"]]
    across = across - (across @ cz) * cz
    return cz, across / np.linalg.norm(across)


def sample_t1(inputs, head, points):
    """The T1 at points about the head's centre."""
    data = nib.as_closest_canonical(nib.load(inputs.t1)).get_fdata()
    voxel = (head.centre + points.reshape(-1, 3)) / head.zooms
    return ndimage.map_coordinates(data, voxel.T, order=1).reshape(points.shape[:-1])


def sample_epi(inputs, head, points):
    """The mean over runs of the EPI reference images at points about the head's centre."""
    total = np.zeros(points.shape[:-1])
    for epi, transform in zip(inputs.epis, inputs.transforms, strict=True):
        image = nib.load(epi)
        data = image.get_fdata()
        inverse = np.linalg.inv(image.affine)
        in_run = ItkAffine.read(transform).apply(head.to_scanner(points.reshape(-1, 3)))
        voxel = in_run @ inverse[:3, :3].T + inverse[:3, 3]
        sampled = ndimage.map_coordinates(
            data / np.percentile(data, EPI_PERCENTILE), voxel.T, order=1
        )
        total += sampled.reshape(total.shape)
    return total / len(inputs.epis)


def show_section(ax, image, vmax, head, cz, across, curves, names, positions, from_hole):
    """An image in the plane through the head's centre, Cz and the cap's left-right bearing."""
    box = (SECTION_U[0], SECTION_U[-1], SECTION_V[0], SECTION_V[-1])
    ax.imshow(image, origin="lower", cmap="gray", vmin=0, vmax=vmax, extent=box)
    # The curves of the scalp in this plane, as far as the scalp map reaches.
    angle = np.deg2rad(np.arange(-125, 125.2, 0.25))
    directions = np.cos(angle)[:, None] * cz + np.sin(angle)[:, None] * across
    reached = np.arccos(directions[:, 2]) < np.deg2rad(120)
    floor = np.where(reached, head.floor_radius(directions), np.nan)
    for offset, colour, label in curves:
        if label == "Outer boundary":
            x, y = geometry.to_pixels(directions)
            radius = np.where(reached, ndimage.map_coordinates(head.outer, [y, x], order=1), np.nan)
        else:
            radius = floor + offset
        ax.plot(radius * np.sin(angle), radius * np.cos(angle), color=colour, linewidth=0.6)
    # The sites of the cap that lie in this plane, marked outside the head.
    near = np.abs(positions @ np.cross(cz, across)) < SECTION_HALF_WIDTH
    for name, position, hole in zip(
        np.array(names)[near], positions[near], from_hole[near], strict=True
    ):
        outward = np.array([position @ across, position @ cz]) / np.linalg.norm(position)
        tip, tail = outward * (np.linalg.norm(position) + 9), outward * (
            np.linalg.norm(position) + 15
        )
        ax.plot(*zip(tip, tail, strict=True), color=BLUE if hole else ORANGE, linewidth=1.0)
        ax.annotate(
            name,
            outward * (np.linalg.norm(position) + 21),
            ha="center",
            va="center",
            color="white",
            fontsize=6,
        )
    ax.set_xlim(box[:2])
    ax.set_ylim(box[2:])
    ax.set_aspect("equal")
    ax.axis("off")
    for text, where, align in (("L", (0.015, 0.5), "left"), ("R", (0.985, 0.5), "right")):
        ax.annotate(text, where, xycoords="axes fraction", ha=align, va="center", color="white")
    scale_bar(ax, 50, "50 mm", where=(0.03, 0.05), colour="white")


def show_profiles(upper, lower, head, holes, cz, across, names, positions, from_hole):
    """The two maps along the section: what the T1 and the EPI show at the sites of the cap."""
    angle = np.deg2rad(np.arange(-125, 125.2, 0.25))
    directions = np.cos(angle)[:, None] * cz + np.sin(angle)[:, None] * across
    reached = np.arccos(directions[:, 2]) < np.deg2rad(120)
    radius = head.floor_radius(directions)
    steps = np.hypot(radius * np.gradient(angle), np.gradient(radius))
    arc = np.cumsum(steps) - np.cumsum(steps)[np.argmin(np.abs(angle))]  # mm from Cz
    x, y = geometry.to_pixels(directions)
    region = checks.cap_region(head.bump)[np.rint(y).astype(int), np.rint(x).astype(int)]
    relief = ndimage.map_coordinates(checks.band(head.bump), [y, x], order=1)
    upper.plot(arc, np.where(reached & region, relief, np.nan), color="black", linewidth=0.8)
    upper.axhline(0, color=GREY, linewidth=0.4)
    upper.set_ylabel("T1 relief\n(mm)")
    loss = ndimage.map_coordinates(holes.depth, [y, x], order=1)
    covered = holes.covered[np.rint(y).astype(int), np.rint(x).astype(int)]
    lower.plot(arc, np.where(reached & covered, loss, np.nan), color="black", linewidth=0.8)
    lower.axhline(HOLE_DEPTH, color="black", linewidth=0.4, linestyle=(0, (4, 3)))
    lower.set_ylabel("EPI signal\nloss")
    lower.set_xlabel("Distance from Cz along the scalp floor, in the plane of a and b (mm)")
    near = np.abs(positions @ np.cross(cz, across)) < SECTION_HALF_WIDTH
    marks = []
    for name, position, hole in zip(
        np.array(names)[near], positions[near], from_hole[near], strict=True
    ):
        where = np.interp(np.arctan2(position @ across, position @ cz), angle, arc)
        marks.append(where)
        for ax in (upper, lower):
            ax.axvline(where, color=BLUE if hole else ORANGE, linewidth=0.8, alpha=0.55, zorder=0)
        upper.annotate(
            name,
            (where, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 1.5),
            textcoords="offset points",
            ha="center",
            fontsize=6,
        )
    upper.set_xlim(min(marks) - 20, max(marks) + 20)
    upper.tick_params(labelbottom=False)


def figure_images(study, subject):
    """The electrodes in the two images of one participant, and the scalp maps made of them."""
    inputs = study.inputs(subject)
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    head, holes = prepared.head, prepared.holes["all"]
    path = study.output / "second" / f"{subject}_space-ScanRAS_electrodes.tsv"
    names, final_mm, from_hole = read_electrodes(path)
    final = head.from_scanner(final_mm)
    cz, across = cap_frame(names, final)
    u, v = np.meshgrid(SECTION_U, SECTION_V)
    plane = u[..., None] * across + v[..., None] * cz

    fig = plt.figure(figsize=(WIDTH, 176 * MM))
    upper = fig.add_gridspec(1, 2, left=0.03, right=0.97, top=0.968, bottom=0.71, wspace=0.06)
    middle = fig.add_gridspec(2, 1, left=0.09, right=0.97, top=0.645, bottom=0.455, hspace=0.18)
    lower = fig.add_gridspec(1, 3, left=0.035, right=0.965, top=0.375, bottom=0.062, wspace=0.1)
    a, b = (fig.add_subplot(upper[0, k]) for k in range(2))
    profiles = middle.subplots(sharex=True)
    c, d, e = (fig.add_subplot(lower[0, k]) for k in range(3))

    t1 = sample_t1(inputs, head, plane)
    curves = [(None, YELLOW, "Outer boundary"), (0.0, SKY, "Scalp floor")]
    show_section(a, t1, np.percentile(t1, 99.5), head, cz, across, curves, names, final, from_hole)
    for (_, colour, label), height in zip(curves, (0.93, 0.86), strict=True):
        a.annotate(label, (0.015, height), xycoords="axes fraction", color=colour)
    title(a, 0, "T1 acquired with the cap on")

    curves = [(LAYER_OFFSETS_MM[0], SKY, "Shell"), (LAYER_OFFSETS_MM[-1], SKY, "Shell")]
    show_section(
        b, sample_epi(inputs, head, plane), 0.85, head, cz, across, curves, names, final, from_hole
    )
    b.annotate("Sampled shell", (0.015, 0.93), xycoords="axes fraction", color=SKY)
    title(b, 1, "Mean EPI of the runs")

    angle = np.deg2rad(np.arange(-125, 125.2, 0.25))
    directions = np.cos(angle)[:, None] * cz + np.sin(angle)[:, None] * across
    trace = geometry.to_pixels(directions[np.arccos(directions[:, 2]) < np.deg2rad(120)])
    region = checks.cap_region(head.bump)
    relief = np.where(region, checks.band(head.bump), np.nan)
    shown = c.imshow(relief, origin="lower", cmap="gray", vmin=-1.5, vmax=1.5)
    x, y = pixels(final)
    for chosen, colour in ((from_hole, BLUE), (~from_hole, ORANGE)):
        c.scatter(x[chosen], y[chosen], s=22, facecolors="none", edgecolors=colour, linewidths=0.6)
    frame_map(c)
    colour_bar(fig, c, shown, "Band-passed height of the boundary (mm)", [-1.5, 0, 1.5])
    title(c, 3, "Relief of the T1 boundary, and the fitted sites")

    shown = d.imshow(np.where(holes.covered, holes.depth, np.nan), origin="lower", **LOSS)
    d.scatter(holes.x, holes.y, s=40, facecolors="none", edgecolors=YELLOW, linewidths=0.7)
    frame_map(d)
    colour_bar(fig, d, shown, "Signal loss relative to the local mean", [-0.4, 0, 0.6])
    title(d, 4, "Scalp layer of the EPI, and its holes")
    for ax in (c, d):
        ax.plot(*trace, color=SKY, linewidth=0.6, linestyle=(0, (4, 3)))

    e.contourf(holes.covered.astype(float), levels=[0.5, 1.5], colors=["#F2F2F2"])
    labels = (
        f"Site at the position of its hole (n = {from_hole.sum()})",
        f"Site at the position from the cap model (n = {(~from_hole).sum()})",
    )
    for chosen, colour, label in zip((from_hole, ~from_hole), (BLUE, ORANGE), labels, strict=True):
        e.scatter(x[chosen], y[chosen], s=13, c=colour, linewidths=0, label=label)
    frame_map(e)
    e.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), handletextpad=0.1, labelspacing=0.3)
    title(e, 5, "Fitted cap")

    show_profiles(*profiles, head, holes, cz, across, names, final, from_hole)
    title(profiles[0], 2, "", shift=-34)
    fig.align_ylabels(profiles)
    fig.savefig(FOLDER / "images.png")
    plt.close(fig)


def render(radius, region, view, size=720, reach=130.0):
    """An orthographic, shaded view of a surface (its radius over the map) within a region."""
    toward, right, up = (np.array(axis, dtype=float) for axis in VIEWS[view])
    points = geometry.map_directions() * radius[..., None]
    normal = np.cross(np.gradient(points, axis=1), np.gradient(points, axis=0))
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = toward + 0.45 * up - 0.35 * right
    shade = 0.12 + 0.78 * np.clip(normal @ (light / np.linalg.norm(light)), 0, 1) ** 1.5
    seen = region & (normal @ toward > 0)
    col = np.rint((points[seen] @ right / reach + 1) / 2 * (size - 1)).astype(int)
    row = np.rint((points[seen] @ up / reach + 1) / 2 * (size - 1)).astype(int)
    order = np.argsort(points[seen] @ toward)  # the nearest point is drawn last
    image = np.full((size, size), np.nan)
    image[row[order], col[order]] = shade[seen][order]
    # Close the gaps the map's sampling leaves where the surface turns away from the viewer.
    distance, nearest = ndimage.distance_transform_edt(np.isnan(image), return_indices=True)
    image = np.where(distance <= 2.5, image[tuple(nearest)], np.nan)
    # Keep the head: the mask of ears and face leaves a few detached specks.
    pieces, count = ndimage.label(np.isfinite(image))
    sizes = ndimage.sum_labels(np.ones_like(pieces), pieces, np.arange(1, count + 1))
    return np.where(pieces == 1 + np.argmax(sizes), image, np.nan)


def view_axes(azimuth, elevation):
    """Unit vectors toward a viewer, toward the image's right and toward its top.

    The azimuth is from the front of the head toward the participant's right, the elevation
    above the horizontal, both in degrees.
    """
    azimuth, elevation = np.deg2rad(azimuth), np.deg2rad(elevation)
    toward = np.array(
        [
            np.sin(azimuth) * np.cos(elevation),
            np.cos(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ]
    )
    up = np.array([0.0, 0.0, 1.0]) - toward[2] * toward
    up /= np.linalg.norm(up)
    return toward, np.cross(up, toward), up


def cast(data, level, head, axes, u, v, step=0.5):
    """Depth of the head's outer boundary along a viewing direction, over the image (u, v).

    Each ray walks from the viewer toward the head and stops at the first sample above the level,
    placed between the last two samples by linear interpolation. Depths are mm toward the viewer
    from the centre of the head; NaN where a ray meets nothing.
    """
    toward, right, up = axes
    plane = u[None, :, None] * right + v[:, None, None] * up
    depth = np.full(plane.shape[:2], np.nan)
    before = np.zeros(plane.shape[:2])
    for w in np.arange(SURFACE_REACH_MM, -SURFACE_REACH_MM, -step):
        open_ = np.isnan(depth)
        voxel = (head.centre + plane[open_] + w * toward) / head.zooms
        value = ndimage.map_coordinates(data, voxel.T, order=1, cval=0)
        hit = value > level
        crossing = (value[hit] - level) / (value[hit] - before[open_][hit])
        rows, cols = np.nonzero(open_)
        depth[rows[hit], cols[hit]] = w + step * crossing
        before[open_] = value
    return depth


def figure_surface(study, subject):
    """The head's outer boundary in the T1 of one participant, and the fitted positions on it."""
    inputs = study.inputs(subject)
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    head = prepared.head
    path = study.output / "second" / f"{subject}_space-ScanRAS_electrodes.tsv"
    _, final_mm, from_hole = read_electrodes(path)
    final = head.from_scanner(final_mm)
    # The image and the level that give the method its outer boundary (head.py).
    image = nib.as_closest_canonical(nib.load(inputs.t1))
    data = ndimage.gaussian_filter(image.get_fdata(), T1_SMOOTHING_MM / head.zooms)
    level = BOUNDARY_LEVEL * np.percentile(data, FAT_PERCENTILE)
    # Only the head: what is above the level but detached from it (ghosts in the air) is left out.
    pieces, count = ndimage.label(data > level)
    sizes = ndimage.sum_labels(np.ones(data.shape), pieces, np.arange(1, count + 1))
    data = np.where(pieces == 1 + np.argmax(sizes), data, 0.0)
    pixel = 0.5
    u, v = np.arange(-125, 125, pixel), np.arange(-150, 125, pixel)
    box = (u[0], u[-1], v[0], v[-1])

    fig, axes = plt.subplots(2, len(OBLIQUES), figsize=(WIDTH, 112 * MM))
    fig.subplots_adjust(left=0.02, right=0.995, top=0.95, bottom=0.04, wspace=0.01, hspace=0.16)
    for column, (name, angles) in enumerate(OBLIQUES.items()):
        toward, right, up = view = view_axes(*angles)
        depth = cast(data, level, head, view, u, v)
        smooth = np.where(
            np.isfinite(depth), ndimage.gaussian_filter(np.nan_to_num(depth), 1.0 / pixel), np.nan
        )
        slope_v, slope_u = np.gradient(smooth, pixel)
        normal = np.stack([-slope_u, -slope_v, np.ones_like(smooth)], axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        light = np.array([-0.35, 0.45, 1.0])
        shade = 0.1 + 0.85 * np.clip(normal @ (light / np.linalg.norm(light)), 0, 1) ** 1.4
        # A site is drawn where it faces the viewer and nothing lies in front of it.
        cols = np.clip(np.rint((final @ right - u[0]) / pixel).astype(int), 0, len(u) - 1)
        rows = np.clip(np.rint((final @ up - v[0]) / pixel).astype(int), 0, len(v) - 1)
        visible = final @ toward > depth[rows, cols] - 8
        for ax in axes[:, column]:
            ax.imshow(shade, origin="lower", cmap="gray", vmin=0, vmax=1, extent=box)
            ax.set_aspect("equal")
            ax.axis("off")
        for chosen, colour in ((from_hole, BLUE), (~from_hole, ORANGE)):
            keep = visible & chosen
            axes[1, column].scatter(
                final[keep] @ right,
                final[keep] @ up,
                s=20,
                facecolors="none",
                edgecolors=colour,
                linewidths=0.7,
            )
        axes[1, column].annotate(
            name, (0.5, 0), xycoords="axes fraction", ha="center", va="top", color=GREY
        )
    scale_bar(axes[0, 0], 50, "50 mm", where=(0.0, 0.0))
    title(axes[0, 0], 0, "Outer boundary of the head in the T1", shift=-2)
    title(axes[1, 0], 1, "Fitted positions on the outer boundary", shift=-2)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=0.8,
        )
        for colour in (BLUE, ORANGE)
    ]
    fig.legend(
        handles,
        ["Position of a hole", "Position from the cap model"],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.5),
        ncols=2,
        handletextpad=0.2,
        columnspacing=2.0,
    )
    fig.savefig(FOLDER / "surface.png")
    plt.close(fig)


def mark(ax, x, y, holds, colour=BLUE):
    """One value per participant: filled where the criterion is met, open where it is not."""
    ax.scatter(x[holds], y[holds], s=12, c=colour, linewidths=0, zorder=3)
    ax.scatter(
        x[~holds], y[~holds], s=10, facecolors="white", edgecolors=colour, linewidths=0.8, zorder=3
    )


def criterion(ax, value, label):
    ax.axhline(value, color="black", linewidth=0.4, linestyle=(0, (4, 3)))
    ax.annotate(
        label,
        (1, value),
        xycoords=("axes fraction", "data"),
        xytext=(0, 1.5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6,
    )


def figure_checks(records):
    """The four graded checks of the second pass, in every participant."""
    order = sorted(records, key=lambda subject: -records[subject]["z"])
    rows = [records[subject] for subject in order]
    x = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 112 * MM), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.09, hspace=0.34, wspace=0.2)
    a, b, c, d = axes.ravel()

    null = np.array([row["null_scores"] for row in rows])
    a.scatter(np.repeat(x, null.shape[1]), null.ravel(), s=3, c=GREY, linewidths=0)
    mark(a, x, np.array([row["score"] for row in rows]), np.array([r["z"] > PASS_Z for r in rows]))
    a.set_ylabel("Agreement of sites and holes")
    a.set_ylim(0, None)
    title(a, 0, "Fitted cap against twisted caps")

    halves = np.array([row["halves_mm"] for row in rows])
    b.scatter(np.repeat(x, halves.shape[1]), halves.ravel(), s=3, c=GREY, linewidths=0)
    median = np.median(halves, axis=1)
    mark(b, x, median, median < PASS_HALVES_MM)
    criterion(b, PASS_HALVES_MM, f"{PASS_HALVES_MM:g} mm")
    b.set_ylabel("Distance between halves (mm)")
    title(b, 1, "Two halves of the sites")

    retest = np.array([row["retest_median_mm"] for row in rows])
    largest = np.array([row["retest_max_mm"] for row in rows])
    c.vlines(x, retest, largest, color=GREY, linewidth=0.6)
    mark(c, x, retest, retest < PASS_RETEST_MM)
    criterion(c, PASS_RETEST_MM, f"{PASS_RETEST_MM:g} mm")
    c.set_ylabel("Distance between halves (mm)")
    title(c, 2, "Two halves of the runs")

    t1 = {key: np.array([row["t1"][key] for row in rows]) for key in rows[0]["t1"]}
    d.axhspan(-1, 1, color=PALE, linewidth=0)
    d.axhline(0, color=GREY, linewidth=0.4)
    start = (t1["start"] - t1["null_mean"]) / t1["null_sd"]
    d.scatter(x, start, s=9, marker="D", facecolors="white", edgecolors=GREY, linewidths=0.6)
    mark(d, x, (t1["fitted"] - t1["null_mean"]) / t1["null_sd"], t1["p"] < PASS_T1_P)
    d.set_ylabel("T1 relief under the sites\n(s.d. above twisted caps)")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=3.6, color=BLUE),
        plt.Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markersize=3,
            markerfacecolor="white",
            markeredgecolor=GREY,
            markeredgewidth=0.6,
        ),
    ]
    d.legend(
        handles,
        ["Fitted positions", "Starting positions"],
        loc="upper right",
        ncols=2,
        handletextpad=0.1,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    title(d, 3, "Fitted cap against twisted caps, T1")

    for ax in (b, c):
        ax.set_yscale("log")
        ax.set_ylim(0.2, 60)
        ax.set_yticks([0.3, 1, 3, 10, 30], labels=["0.3", "1", "3", "10", "30"])
    for ax in (c, d):
        ax.set_xlabel("Participant (ordered by z of the twisted-cap null)")
    a.set_xticks(x[::2] if len(x) > 12 else x)
    a.set_xlim(0.3, len(x) + 0.7)
    fig.savefig(FOLDER / "checks.png")
    plt.close(fig)


def cohort(study, records):
    """Every passing participant's scalp floor, starting and fitted positions about the centre.

    The starting positions are projected onto the scalp floor (nearest point), as a forward model
    projects electrodes onto the scalp: what remains is their distance along the scalp.
    """
    directions = geometry.map_directions()[geometry.map_disc()]
    floors, starts, fitted, from_hole = [], [], [], []
    for subject, record in records.items():
        if not record["passed"]:
            continue
        prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
        path = study.output / "second" / f"{subject}_space-ScanRAS_electrodes.tsv"
        _, final_mm, holes = read_electrodes(path)
        floors.append(prepared.head.floor)
        surface = directions * prepared.head.floor[geometry.map_disc()][:, None]
        nearest = cKDTree(surface).query(prepared.head.from_scanner(prepared.start_mm))[1]
        starts.append(surface[nearest])
        fitted.append(prepared.head.from_scanner(final_mm))
        from_hole.append(holes)
    return np.array(floors), np.array(starts), np.array(fitted), np.array(from_hole)


def figure_template(study, records):
    """Where the fitted positions lie relative to the starting (template) positions."""
    floors, starts, fitted, _ = cohort(study, records)
    channels = starts.shape[1]
    fitted = fitted[:, :channels]
    mean_start, mean_fitted = starts.mean(axis=0), fitted.mean(axis=0)
    floor = floors.mean(axis=0)
    surface = geometry.map_directions() * floor[..., None]
    near = cKDTree(np.vstack([mean_start, mean_fitted])).query(surface.reshape(-1, 3))[0]
    region = near.reshape(floor.shape) < SURFACE_MARGIN_MM
    outward = mean_fitted / np.linalg.norm(mean_fitted, axis=1, keepdims=True)

    fig = plt.figure(figsize=(WIDTH, 60 * MM))
    grid = fig.add_gridspec(1, 3, left=0.03, right=0.68, top=0.88, bottom=0.1, wspace=0.03)
    views = [fig.add_subplot(grid[0, k]) for k in range(3)]
    reach = 130.0
    for ax, view in zip(views, ("Top", "Left", "Back"), strict=True):
        toward, right, up = (np.array(axis, dtype=float) for axis in VIEWS[view])
        shaded = render(floor, region, view, reach=reach)
        ax.imshow(
            shaded, origin="lower", cmap="gray", vmin=-0.6, vmax=1, extent=(-reach, reach) * 2
        )
        seen = outward @ toward > 0.1
        ax.scatter(mean_start[seen] @ right, mean_start[seen] @ up, s=5, c="black", linewidths=0)
        ax.quiver(
            mean_start[seen] @ right,
            mean_start[seen] @ up,
            (mean_fitted - mean_start)[seen] @ right,
            (mean_fitted - mean_start)[seen] @ up,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=BLUE,
            width=0.008,
            headwidth=3.5,
            headlength=4,
        )
        ax.set_xlim(-118, 118)
        ax.set_ylim((-118, 118) if view == "Top" else (-75, 118))
        ax.set_aspect("equal")
        ax.set_anchor("S")
        ax.axis("off")
        ax.annotate(view, (0.5, 0), xycoords="axes fraction", ha="center", va="top", color=GREY)
    scale_bar(views[1], 50, "50 mm", where=(0.0, 0.0))
    title(views[0], 0, f"Mean displacement from the template (n = {len(floors)})")

    ax = fig.add_axes([0.775, 0.19, 0.205, 0.69])
    layout_ = pipeline.lessons(study).layout
    angle = np.degrees(layout_.angle[:channels])
    distance = np.linalg.norm(fitted - starts, axis=2)  # participants by channels
    low, median, high = np.percentile(distance, [25, 50, 75], axis=0)
    ax.vlines(angle, low, high, color=PALE, linewidth=1.0)
    ax.scatter(angle, median, s=8, c=BLUE, linewidths=0, zorder=3)
    ax.set_xlabel("Angle from Cz in the montage (degrees)")
    ax.set_ylabel("Distance from the template (mm)")
    ax.set_xlim(-4, 125)
    ax.set_ylim(0, None)
    title(ax, 1, "Displacement by channel", shift=-26)
    per_participant = np.median(distance, axis=1)
    print(
        f"distance along the scalp from the template: median over participants "
        f"{np.median(per_participant):.1f} mm (range {per_participant.min():.1f} to "
        f"{per_participant.max():.1f}); median per channel from {median.min():.1f} to "
        f"{median.max():.1f} mm"
    )
    fig.savefig(FOLDER / "template.png")
    plt.close(fig)


def paired(ax, first, second, label, log=False):
    """Each participant's value in the first and in the second pass, and the medians."""
    ax.plot([0, 1], [first, second], color=GREY, linewidth=0.5, alpha=0.8)
    ax.scatter(np.zeros(len(first)), first, s=7, c=GREY, linewidths=0, zorder=3)
    ax.scatter(np.ones(len(second)), second, s=7, c=GREY, linewidths=0, zorder=3)
    ax.plot([0, 1], [np.median(first), np.median(second)], color=BLUE, linewidth=1.6, zorder=4)
    ax.set_xticks([0, 1], labels=["First", "Second"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_xlabel("Pass")
    ax.set_ylabel(label)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.7, 60)
        ax.set_yticks([1, 3, 10, 30], labels=["1", "3", "10", "30"])


def frame_layout(ax):
    """Limits and marks of the cap's layout: the nose at the top, 50 mm on the unstretched cap."""
    ax.set_xlim(-205, 205)
    ax.set_ylim(-205, 205)
    ax.set_aspect("equal")
    ax.set_anchor("NW")
    ax.axis("off")
    for text, where, align in (("L", (0, 0.5), "right"), ("R", (1, 0.5), "left")):
        ax.annotate(text, where, xycoords="axes fraction", ha=align, va="center", color=GREY)
    scale_bar(ax, 50, "50 mm", where=(0.0, 0.02))


def figure_passes(study, first, second):
    """The sites across the cohort: which have a hole, the learned layout, and the two passes."""
    lessons = pipeline.lessons(study)
    learned = layout.learn(lessons.layout, lessons.east, lessons.north)
    enough = np.isfinite(lessons.east).sum(axis=0) >= LAYOUT_MINIMUM_PARTICIPANTS
    base = lessons.layout
    x = NOMINAL_RADIUS_MM * base.angle * np.cos(base.bearing)
    y = NOMINAL_RADIUS_MM * base.angle * np.sin(base.bearing)
    dx = NOMINAL_RADIUS_MM * learned.angle * np.cos(learned.bearing) - x
    dy = NOMINAL_RADIUS_MM * learned.angle * np.sin(learned.bearing) - y
    from_hole = cohort(study, second)[3]

    fig = plt.figure(figsize=(WIDTH, 70 * MM))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.55, 1.55, 1, 1], wspace=0.5)
    grid.update(left=0.03, right=0.985, top=0.89, bottom=0.15)
    a, b, c, d = (fig.add_subplot(grid[0, k]) for k in range(4))

    shown = a.scatter(
        x, y, s=26, c=from_hole.mean(axis=0), cmap="viridis", vmin=0, vmax=1, linewidths=0
    )
    frame_layout(a)
    inset = a.inset_axes([0.42, 0.0, 0.5, 0.03])
    bar = fig.colorbar(shown, cax=inset, orientation="horizontal", ticks=[0, 0.5, 1])
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(length=1.5, width=0.4, pad=1.5)
    bar.set_label("Fraction of participants", labelpad=2)
    title(a, 0, f"Sites with a hole (n = {len(from_hole)})")

    magnified = 5
    b.scatter(x[~enough], y[~enough], s=9, facecolors="white", edgecolors=GREY, linewidths=0.6)
    b.scatter(x[enough], y[enough], s=5, c=GREY, linewidths=0)
    arrows = b.quiver(
        x[enough],
        y[enough],
        dx[enough],
        dy[enough],
        angles="xy",
        scale_units="xy",
        scale=1 / magnified,
        color=BLUE,
        width=0.006,
        headwidth=3.5,
        headlength=4,
        zorder=3,
    )
    b.quiverkey(arrows, 0.8, 0.02, 5, "5 mm", labelpos="E", coordinates="axes", labelsep=0.04)
    frame_layout(b)
    title(b, 1, f"Learned layout (n = {len(lessons.subjects)})")

    both = [subject for subject in second if subject in first]
    for ax, key, label in (
        (c, "matched", "Sites with a hole of their own"),
        (d, "halves_median_mm", "Distance between halves (mm)"),
    ):
        values = [np.array([records[s][key] for s in both]) for records in (first, second)]
        paired(ax, *values, label, log=key == "halves_median_mm")
    title(c, 2, "Sites with a hole")
    d.axhline(PASS_HALVES_MM, color="black", linewidth=0.4, linestyle=(0, (4, 3)))
    title(d, 3, "Two halves of the sites")
    fig.savefig(FOLDER / "passes.png")
    plt.close(fig)


def figure_synthetic(folder):
    """Fitted against true positions on the synthetic participant of the examples."""
    study, subject = Study.read(folder / "study.toml"), "sub-01"
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    head, holes = prepared.head, prepared.holes["all"]
    path = study.output / "second" / f"{subject}_space-ScanRAS_electrodes.tsv"
    _, fitted_mm, _ = read_electrodes(path)
    truth_mm = np.loadtxt(folder / f"{subject}_truth.tsv", skiprows=1, usecols=(1, 2, 3))
    error = np.linalg.norm(fitted_mm - truth_mm, axis=1)

    fig = plt.figure(figsize=(WIDTH, 74 * MM))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.28)
    grid.update(left=0.04, right=0.985, top=0.9, bottom=0.15)
    a, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

    shown = a.imshow(np.where(holes.covered, holes.depth, np.nan), origin="lower", **LOSS)
    true = a.scatter(
        *pixels(head.from_scanner(truth_mm)),
        s=38,
        facecolors="none",
        edgecolors=ORANGE,
        linewidths=0.7,
    )
    fitted = a.scatter(*pixels(head.from_scanner(fitted_mm)), s=5, c=BLUE, linewidths=0)
    frame_map(a)
    a.legend(
        [true, fitted],
        ["True position", "Fitted position"],
        loc="upper left",
        bbox_to_anchor=(0.86, 1.02),
        handletextpad=0.1,
    )
    colour_bar(fig, a, shown, "Signal loss relative to the local mean", [-0.4, 0, 0.6])
    title(a, 0, "Synthetic head, scalp layer of the mean EPI")

    b.step(np.sort(error), np.arange(1, len(error) + 1) / len(error), where="post", color=BLUE)
    median = np.median(error)
    b.axvline(median, color="black", linewidth=0.4, linestyle=(0, (4, 3)))
    b.annotate(
        f"median {median:.2f} mm",
        (median, 0.03),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=6,
    )
    b.set_xlim(0, 3)
    b.set_ylim(0, 1)
    b.set_xlabel("Distance from the true position (mm)")
    b.set_ylabel(f"Cumulative fraction of sites (n = {len(error)})")
    title(b, 1, "Error of the fitted positions")
    fig.savefig(FOLDER / "synthetic.png")
    plt.close(fig)


def unassigned_holes(prepared, final, from_hole):
    """Which holes were assigned to no site, and which holes lie within the outline of the cap."""
    head, holes = prepared.head, prepared.holes["all"]
    points = geometry.to_directions(holes.x, holes.y)
    points = points * head.floor_radius(points)[:, None]
    taken = cKDTree(final[from_hole]).query(points)[0] < 0.1
    outline = Delaunay(np.column_stack(pixels(final)))
    return ~taken, outline.find_simplex(np.column_stack([holes.x, holes.y])) >= 0


def hole_widths(study, records):
    """Full width at half depth (mm) of every hole assigned to a site, in the passing fits.

    The scalp layer is divided by its local mean without the smoothing that detection applies;
    the depth is averaged around each hole and followed outward to half its central value.
    """
    radii = np.arange(0, 20.5, 0.5)
    around = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    widths = []
    for subject, record in records.items():
        if not record["passed"]:
            continue
        inputs = study.inputs(subject)
        prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
        head, holes = prepared.head, prepared.holes["all"]
        shells = [
            run_shell(head, epi, xfm)
            for epi, xfm in zip(inputs.epis, inputs.transforms, strict=True)
        ]
        layer = scalp_layer(shells)
        broad = ndimage.gaussian_filter(layer, COVERAGE_SMOOTHING_MM / MM_PER_PIXEL)
        loss = 1 - layer / np.maximum(broad, 1e-6)
        path = study.output / "second" / f"{subject}_space-ScanRAS_electrodes.tsv"
        _, final_mm, from_hole = read_electrodes(path)
        unassigned, _ = unassigned_holes(prepared, head.from_scanner(final_mm), from_hole)
        for x, y in zip(holes.x[~unassigned], holes.y[~unassigned], strict=True):
            step = radii[:, None] / MM_PER_PIXEL
            rows, cols = y + step * np.sin(around), x + step * np.cos(around)
            ring = ndimage.map_coordinates(loss, [rows.ravel(), cols.ravel()], order=1)
            profile = ring.reshape(len(radii), -1).mean(axis=1)
            below = np.nonzero(profile < profile[0] / 2)[0]
            if profile[0] > 0 and len(below):
                k = below[0]
                half = np.interp(profile[0] / 2, profile[[k, k - 1]], radii[[k, k - 1]])
                widths.append(2 * half)
    return np.array(widths)


def print_numbers(study, subject, first, second):
    """Every number the README and its captions quote."""
    spread = lambda v: f"{np.median(v):.2f} ({np.min(v):.2f} to {np.max(v):.2f})"  # noqa: E731
    for name, records in (("first", first), ("second", second)):
        rows = list(records.values())
        print(
            f"{name} pass: {sum(r['passed'] for r in rows)} of {len(rows)} fits met all criteria; "
            f"medians: sites with a hole {np.median([r['matched'] for r in rows]):.0f}, "
            f"z {np.median([r['z'] for r in rows]):.1f}, "
            f"split halves {np.median([r['halves_median_mm'] for r in rows]):.2f} mm, "
            f"test-retest {np.median([r['retest_median_mm'] for r in rows]):.2f} mm"
        )
    t1 = {key: np.array([r["t1"][key] for r in second.values()]) for key in ("fitted", "start")}
    null = np.array([[r["t1"]["null_mean"], r["t1"]["null_sd"]] for r in second.values()]).T
    for key, values in t1.items():
        print(
            f"T1 relief under the {key} positions, s.d. above twisted caps: "
            f"{spread((values - null[0]) / null[1])}"
        )
    fraction = cohort(study, second)[3].mean(axis=0)
    names = read_electrodes(study.output / "second" / f"{subject}_space-ScanRAS_electrodes.tsv")[0]
    print(
        f"fraction of passing participants in which a site has a hole: {fraction.min():.2f} to "
        f"{fraction.max():.2f}; at least 0.8 for {(fraction >= 0.8).sum()} sites; below 0.3 for "
        f"{[names[k] for k in np.nonzero(fraction < 0.3)[0]]}"
    )
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    _, final_mm, from_hole = read_electrodes(
        study.output / "second" / f"{subject}_space-ScanRAS_electrodes.tsv"
    )
    unassigned, inside = unassigned_holes(prepared, prepared.head.from_scanner(final_mm), from_hole)
    print(
        f"{subject}: {len(unassigned)} holes; {from_hole.sum()} sites at a hole, "
        f"{(~from_hole).sum()} from the cap model; {unassigned.sum()} holes assigned to no site, "
        f"{(unassigned & ~inside).sum()} of them outside the outline of the cap"
    )
    widths = hole_widths(study, second)
    low, high = np.percentile(widths, [25, 75])
    print(
        f"width of the holes at half depth: median {np.median(widths):.1f} mm "
        f"(interquartile range {low:.1f} to {high:.1f}; n = {len(widths)})"
    )


if __name__ == "__main__":
    study, subject, synthetic = Study.read(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    plt.rcParams.update(STYLE)
    FOLDER.mkdir(exist_ok=True)
    first, second = (read_records(study.output / stage) for stage in ("first", "second"))
    figure_images(study, subject)
    figure_surface(study, subject)
    figure_checks(second)
    figure_template(study, second)
    figure_passes(study, first, second)
    figure_synthetic(synthetic)
    print_numbers(study, subject, first, second)
