"""Every setting of the method, and the study file that names a cohort's inputs."""

import tomllib
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np

# The scalp map: an azimuthal equidistant projection about a centre in the head (geometry.py).
MAP_SIZE = 900  # pixels along each side
MAP_EXTENT = np.deg2rad(125)  # angle from the superior axis at the map's edge
NOMINAL_RADIUS_MM = 90.0  # head radius that converts map pixels and montage angles to mm
MM_PER_PIXEL = 2 * MAP_EXTENT / (MAP_SIZE - 1) * NOMINAL_RADIUS_MM  # along a meridian

# The head's outer boundary in the cap-on T1 (head.py). Levels are fractions of scalp fat.
T1_SMOOTHING_MM = 0.5
FAT_PERCENTILE = 99
HEAD_LEVEL = 0.3  # tissue bright enough to locate the head
BOUNDARY_LEVEL = 0.07  # the outermost signal along a ray: skin, or gel where there is any
SLAB_MM = 110.0  # the top of the head whose centre of mass sets the centre left-right, front-back
CENTRE_DEPTH_MM = 95.0  # the centre lies this far below the top of the head
RAY_RADII_MM = np.arange(40, 150, 0.5)
FLOOR_PERCENTILE = 20  # the scalp floor: a low percentile of the boundary in a window, smoothed
FLOOR_WINDOW_PIXELS = 55
FLOOR_SMOOTHING_PIXELS = 10

# The scalp layer of the mean EPI and its holes (epi.py).
LAYER_OFFSETS_MM = np.arange(-6, 0.5, 1.0)  # depths about the scalp floor
EPI_PERCENTILE = 99  # each run is scaled by this percentile of its own signal
COVERAGE_SMOOTHING_MM = 14.0
COVERAGE_LEVEL = 0.25  # of the 90th percentile of the smoothed layer
COVERAGE_MARGIN_MM = 6.0
HOLE_SMOOTHING_MM = 4.0
HOLE_SPACING_MM = 16.0  # a hole is the deepest point within this window
HOLE_DEPTH = 0.12  # signal lost relative to the local mean

# The cap fit (cap.py). A pose is pitch, roll, twist (degrees), stretch left-right and front-back.
POSE_NAMES = ("pitch_deg", "roll_deg", "twist_deg", "stretch_lr", "stretch_ap")
SECTION_ANGLES = np.deg2rad(np.arange(0, 150.5, 1.0))  # along a scalp section from Cz
MATCH_SIGMA_MM = 5.0  # width of the site-to-hole agreement
MINIMUM_SITES_IN_VIEW = 10
SNAP_MM = 8.0  # a site takes its hole's position within this distance
FIRST_BOUNDS = ((-20.0, 20.0), (-12.0, 12.0), (-15.0, 15.0), (0.8, 1.3), (0.8, 1.3))
FIRST_SEEDS = (0,)
SECOND_BOUNDS = ((-12.0, 12.0), (-8.0, 8.0), (-10.0, 10.0), (0.75, 1.0), (0.75, 1.0))
SECOND_SEEDS = (0, 1)
SEARCH = dict(popsize=25, maxiter=200, tol=1e-8, polish=True)

# The checks (checks.py).
NULL_TWISTS_DEG = np.r_[np.arange(40, 141, 20.0), np.arange(220, 321, 20.0)]
NULL_HALF_WIDTH_DEG = 10.0
SPLIT_HALVES = 6
T1_DRAWS = 1000
T1_JITTER_DEG = 8.0
T1_TWIST_RANGE_DEG = (20.0, 340.0)
T1_HEIGHT_RANGE_MM = (-4.0, 6.0)
T1_BAND_MM = (2.0, 6.0)  # bumps and dents of this size
T1_WINDOW_MM = 6.0  # the strongest bump or dent within this window of a site
T1_TALL_MM = 6.0  # ears and face stand this far above the scalp floor
T1_TALL_MARGIN_MM = 8.0
T1_REGION_EXTENT = np.deg2rad(118)
T1_MINIMUM_SITES = 10

# A fit passes when all of these hold.
PASS_Z = 3.0
PASS_HALVES_MM = 6.0
PASS_RETEST_MM = 5.0
PASS_T1_P = 0.05

# The cap layout learned across participants (layout.py).
LAYOUT_MATCH_MM = 10.0
LAYOUT_MINIMUM_PARTICIPANTS = 5
LAYOUT_MAXIMUM_STRETCH = 1.0  # a first-pass fit teaches the layout only below this stretch


@dataclass(frozen=True)
class Inputs:
    """One participant's files."""

    subject: str
    t1: Path
    epis: tuple[Path, ...]
    transforms: tuple[Path, ...]
    start: Path


@dataclass(frozen=True)
class Study:
    """A cohort that wore caps of one layout: where its files are and what the cap is."""

    output: Path
    subjects: tuple[str, ...]
    montage: str
    reference: str | None
    ground: str | None
    patterns: dict[str, str]

    @classmethod
    def read(cls, path: str | Path) -> "Study":
        with open(path, "rb") as file:
            table = tomllib.load(file)
        unknown = set(table) - {"output", "subjects", "montage", "reference", "ground", "inputs"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        patterns = table["inputs"]
        if set(patterns) != {"t1", "epi", "transform", "start"}:
            raise ValueError(f"{path}: [inputs] must name exactly t1, epi, transform and start")
        return cls(
            output=Path(table["output"]),
            subjects=tuple(table["subjects"]),
            montage=table["montage"],
            reference=table.get("reference"),
            ground=table.get("ground"),
            patterns=patterns,
        )

    def inputs(self, subject: str) -> Inputs:
        """The participant's files: each pattern with {subject} filled in, wildcards expanded."""
        found = {
            key: tuple(Path(p) for p in sorted(glob(pattern.format(subject=subject))))
            for key, pattern in self.patterns.items()
        }
        for key in ("t1", "start"):
            if len(found[key]) != 1:
                raise FileNotFoundError(
                    f"{subject}: expected one {key} file, found {len(found[key])}"
                )
        if len(found["epi"]) < 2:
            raise FileNotFoundError(
                f"{subject}: at least two EPI runs are needed (the test-retest check splits "
                f"them), found {len(found['epi'])}"
            )
        if len(found["transform"]) != len(found["epi"]):
            raise FileNotFoundError(
                f"{subject}: {len(found['epi'])} EPI runs but {len(found['transform'])} transforms"
            )
        return Inputs(subject, found["t1"][0], found["epi"], found["transform"], found["start"][0])

    @property
    def extra_sites(self) -> dict[str, str]:
        """Sites on the cap that record no channel of their own, and their role."""
        sites = {self.reference: "reference", self.ground: "ground"}
        return {name: role for name, role in sites.items() if name is not None}
