"""How much the EPI of a study shows of the scalp and of the electrodes, per participant.

    python docs/epi_quality.py STUDY [PASS]

For every participant of a study on which PASS (first or second; by default second) has been run:
- sites_in_view: fraction of the fitted sites within the EPI's coverage of the scalp;
- scalp_signal: median of the scalp layer within that coverage, in units of each run's 99th
  percentile (the layer is the mean over runs and over the 6 mm below the scalp floor);
- scalp_to_cortex: that signal relative to the same shell 15 to 25 mm below the floor;
- holes and hole_depth: number of holes, and the median of their depth (signal lost relative to
  the local mean);
- z and passed: of the fit.
Writes <output>/epi_quality.tsv and prints each quantity's median and range over participants.
"""

import csv
import json
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

from eegloc import geometry, pipeline
from eegloc.config import EPI_PERCENTILE, LAYER_OFFSETS_MM, Study
from eegloc.epi import ItkAffine

CORTEX_OFFSETS_MM = (-25.0, -20.0, -15.0)


def shell(head, epi, transform, offsets):
    """One run on the map, averaged over depths about the scalp floor (as epi.run_shell)."""
    image = nib.load(epi)
    data = image.get_fdata()
    data = data / np.percentile(data, EPI_PERCENTILE)
    to_run, inverse = ItkAffine.read(transform), np.linalg.inv(image.affine)
    directions = geometry.map_directions()
    total = np.zeros(directions.shape[:2])
    for offset in offsets:
        in_t1 = head.to_scanner(directions * (head.floor + offset)[..., None]).reshape(-1, 3)
        voxel = to_run.apply(in_t1) @ inverse[:3, :3].T + inverse[:3, 3]
        total += ndimage.map_coordinates(data, voxel.T, order=1, cval=0).reshape(total.shape)
    return total / len(offsets)


def quality(study, stage, subject):
    """One participant's quantities."""
    inputs = study.inputs(subject)
    prepared = pipeline.Prepared.load(study.output / "maps" / f"{subject}.npz")
    head, holes = prepared.head, prepared.holes["all"]
    record = json.loads((study.output / stage / f"{subject}.json").read_text())
    runs = list(zip(inputs.epis, inputs.transforms, strict=True))
    scalp = np.mean([shell(head, e, x, LAYER_OFFSETS_MM) for e, x in runs], axis=0)
    cortex = np.mean([shell(head, e, x, CORTEX_OFFSETS_MM) for e, x in runs], axis=0)
    seen = holes.covered
    depth = holes.depth[holes.y.astype(int), holes.x.astype(int)]
    return dict(
        subject=subject,
        runs=len(runs),
        sites_in_view=record["in_view"] / (len(prepared.start_names) + len(study.extra_sites)),
        scalp_signal=float(np.median(scalp[seen])),
        scalp_to_cortex=float(np.median(scalp[seen]) / np.median(cortex[seen])),
        holes=len(holes.x),
        hole_depth=float(np.median(depth)),
        z=record["z"],
        passed=int(record["passed"]),
    )


if __name__ == "__main__":
    study = Study.read(sys.argv[1])
    stage = sys.argv[2] if len(sys.argv) > 2 else "second"
    rows = [quality(study, stage, subject) for subject in study.subjects]
    with open(study.output / "epi_quality.tsv", "w", newline="") as file:
        writer = csv.DictWriter(file, rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} participants; median (range) over participants:")
    for key in rows[0]:
        if key != "subject":
            values = [row[key] for row in rows]
            print(f"  {key}: {np.median(values):.3f} ({min(values):.3f} to {max(values):.3f})")
