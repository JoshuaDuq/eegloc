"""Starting positions for eegloc from a template montage, with MNE-Python.

A standard montage is fitted to the scalp surface of the participant's FreeSurfer reconstruction:
its fiducials to those estimated from fsaverage, then its electrodes to the surface by iterative
closest point. The channel positions are written in the scanner coordinates of the T1.

The reconstruction must hold a scalp surface (for example from `mne make_scalp_surfaces` or
`mne watershed_bem`).

    python starting_positions.py SUBJECTS_DIR SUBJECT MONTAGE CHANNELS OUTPUT

CHANNELS is a text file with one recorded channel name per line; OUTPUT is the table to write.
"""

import sys
from pathlib import Path

import mne
import nibabel as nib
import numpy as np

subjects_dir, subject, montage, channels, output = sys.argv[1:]

names = Path(channels).read_text().split()
info = mne.create_info(names, 1000.0, "eeg")
info.set_montage(montage)
coregistration = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials="estimated")
coregistration.fit_fiducials()
coregistration.fit_icp(n_iterations=20, eeg_weight=1.0)

# MNE's head coordinates (m) to FreeSurfer's surface coordinates, then to the scanner's (mm).
head = np.array([channel["loc"][:3] for channel in info["chs"]])
surface = mne.transforms.apply_trans(coregistration.trans, head) * 1e3
header = nib.load(Path(subjects_dir, subject, "mri", "orig.mgz")).header
scanner = mne.transforms.apply_trans(
    header.get_vox2ras() @ np.linalg.inv(header.get_vox2ras_tkr()), surface
)

rows = [
    "name\tx\ty\tz",
    *(f"{name}\t{x:.3f}\t{y:.3f}\t{z:.3f}" for name, (x, y, z) in zip(names, scanner, strict=True)),
]
Path(output).write_text("\n".join(rows) + "\n")
