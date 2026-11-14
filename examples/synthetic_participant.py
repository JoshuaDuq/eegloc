"""A synthetic participant with electrodes at known positions.

The head is an ellipsoid: brain, a bright layer of scalp fat, and a blob of dim gel on the scalp at
each electrode. Each EPI run sees the same head a little displaced, with a signal void in the scalp
at each electrode. The electrodes sit where the cap model puts them for a chosen pose.

    python examples/synthetic_participant.py FOLDER

writes the participant's images, transforms and starting positions, the true positions
(sub-01_truth.tsv) and a study file (study.toml) into FOLDER.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from eegloc.cap import Cap, montage_layout
from eegloc.epi import LPS, Holes
from eegloc.head import read_head

CHANNELS = (  # noqa: SIM905
    "Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 Fz Cz Pz Oz FC1 FC2 CP1 CP2 FC5 FC6 CP5 "
    "CP6 TP9 TP10 POz F1 F2 C1 C2 P1 P2 AF3 AF4 FC3 FC4 CP3 CP4 PO3 PO4 F5 F6 C5 C6 P5 P6 AF7 "
    "AF8 FT7 FT8 TP7 TP8 PO7 PO8 FT9 FT10 Fpz CPz"
).split()
MONTAGE, REFERENCE, GROUND = "easycap-M1", "FCz", "AFz"
POSE = (5.0, -3.0, 4.0, 0.9, 0.87)  # the cap slid, twisted and stretched away from the start

HEAD_CENTRE = np.array([0.0, 0.0, -10.0])  # scanner mm
HEAD_AXES = np.array([72.0, 90.0, 95.0])
SCALP = 0.92  # the scalp fat lies between this fraction of the ellipsoid and its surface
GEL_MM, VOID_MM = 4.5, 7.5
SHIFTS_MM = ((1.0, 0.5, -0.5), (-0.8, 1.2, 0.3), (0.4, -1.0, 0.9), (-1.1, -0.3, -0.7))


def _level(points):
    return np.linalg.norm((points - HEAD_CENTRE) / HEAD_AXES, axis=-1)


def _near(points, electrodes, within):
    distance = cKDTree(electrodes).query(points.reshape(-1, 3), distance_upper_bound=within)[0]
    return np.isfinite(distance).reshape(points.shape[:-1])


def _grid(shape, affine):
    voxels = np.stack(np.meshgrid(*map(np.arange, shape), indexing="ij"), axis=-1)
    return voxels @ affine[:3, :3].T + affine[:3, 3]


def _t1(electrodes):
    shape, affine = (180, 220, 200), np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-90.0, -110.0, -100.0]
    points = _grid(shape, affine)
    level = _level(points)
    data = np.where(level < SCALP, 0.45, np.where(level <= 1, 1.0, 0.0))
    if len(electrodes):
        data[(level > 1) & _near(points, electrodes, GEL_MM)] = 0.25
    return nib.Nifti1Image(data.astype(np.float32), affine)


def _epi(electrodes, shift, rng):
    shape, affine = (64, 76, 68), np.diag([3.0, 3.0, 3.0, 1.0])
    affine[:3, 3] = [-94.5, -112.5, -100.5]
    points = _grid(shape, affine) - shift  # this run's scanner coordinates, back to the T1's
    level = _level(points)
    data = np.where(level < SCALP, 1.0, np.where(level <= 1, 0.7, 0.0))
    data[(level >= SCALP - 0.05) & _near(points, electrodes, VOID_MM)] = 0.03
    data += rng.normal(0, 0.02, shape) * (level <= 1)
    return nib.Nifti1Image(data.astype(np.float32), affine)


def _table(path, names, positions):
    rows = ["name\tx\ty\tz"] + [
        f"{name}\t{x:.3f}\t{y:.3f}\t{z:.3f}"
        for name, (x, y, z) in zip(names, positions, strict=True)
    ]
    path.write_text("\n".join(rows) + "\n")


def make_participant(folder: Path, subject: str, pose=POSE, seed: int = 0) -> dict[str, np.ndarray]:
    """Write a participant's T1, EPI runs, transforms, starting and true positions.

    Returns the true scanner position (mm) of every site on the cap.
    """
    folder.mkdir(parents=True, exist_ok=True)
    bare = folder / "bare.nii.gz"
    nib.save(_t1(np.empty((0, 3))), bare)
    head = read_head(bare)
    bare.unlink()
    nowhere = Holes(np.array([450.0]), np.array([450.0]), np.zeros((2, 2)), np.ones((2, 2), bool))
    layout = montage_layout([*CHANNELS, REFERENCE, GROUND], MONTAGE)

    # The start: the montage laid on the scalp upright, unstretched.
    upright = dict(
        Cz=np.array([0.0, 0.0, 1.0]), C3=np.array([-1.0, 0, 0]), C4=np.array([1.0, 0, 0])
    )
    start = Cap(head, nowhere, layout, upright).placed([0, 0, 0, 1, 1])[: len(CHANNELS)]
    _table(folder / f"{subject}_start.tsv", CHANNELS, head.to_scanner(start))

    # The truth: the same cap, moved and stretched by the pose.
    frame = {name: start[CHANNELS.index(name)] for name in ("Cz", "C3", "C4")}
    electrodes = head.to_scanner(Cap(head, nowhere, layout, frame).placed(pose))
    _table(folder / f"{subject}_truth.tsv", layout.names, electrodes)
    nib.save(_t1(electrodes), folder / f"{subject}_T1w.nii.gz")
    rng = np.random.default_rng(seed)
    for run, shift in enumerate(SHIFTS_MM, start=1):
        nib.save(
            _epi(electrodes, np.array(shift), rng), folder / f"{subject}_run-{run}_boldref.nii.gz"
        )
        translation = " ".join(f"{value:g}" for value in np.array(shift) * LPS)
        (folder / f"{subject}_run-{run}_xfm.txt").write_text(
            "#Insight Transform File V1.0\n#Transform 0\n"
            "Transform: AffineTransform_double_3_3\n"
            f"Parameters: 1 0 0 0 1 0 0 0 1 {translation}\n"
            "FixedParameters: 0 0 0\n"
        )
    return dict(zip(layout.names, electrodes, strict=True))


def write_study(path: Path, folder: Path, subjects) -> None:
    """A study file for participants written by make_participant into `folder`."""
    path.write_text(
        f'output = "{folder / "eegloc"}"\n'
        f"subjects = {list(subjects)}\n"
        f'montage = "{MONTAGE}"\nreference = "{REFERENCE}"\nground = "{GROUND}"\n\n'
        "[inputs]\n"
        f't1 = "{folder}/{{subject}}_T1w.nii.gz"\n'
        f'epi = "{folder}/{{subject}}_run-*_boldref.nii.gz"\n'
        f'transform = "{folder}/{{subject}}_run-*_xfm.txt"\n'
        f'start = "{folder}/{{subject}}_start.tsv"\n'
    )


if __name__ == "__main__":
    folder = Path(sys.argv[1]).resolve()
    make_participant(folder, "sub-01")
    write_study(folder / "study.toml", folder, ["sub-01"])
