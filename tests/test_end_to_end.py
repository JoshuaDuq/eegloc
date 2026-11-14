"""Both passes from the command line, on a synthetic participant whose electrodes are known."""

import csv
import json

import numpy as np
import pytest
from synthetic_participant import POSE, make_participant, write_study

from eegloc.cli import main

TOLERANCE = [1, 1, 1, 0.02, 0.02]


@pytest.mark.slow
def test_positions_are_recovered(tmp_path):
    truth = make_participant(tmp_path, "sub-01")
    study = tmp_path / "study.toml"
    write_study(study, tmp_path, ["sub-01"])
    main(["fit", str(study)])
    main(["refine", str(study)])  # one participant teaches no layout: the montage's is kept

    for stage in ("first", "second"):
        folder = tmp_path / "eegloc" / stage
        record = json.loads((folder / "sub-01.json").read_text())
        assert record["passed"], record
        off = np.abs(np.array(list(record["pose"].values())) - POSE)
        assert (off < TOLERANCE).all(), record["pose"]
        with open(folder / "sub-01_space-ScanRAS_electrodes.tsv", newline="") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
        assert [row["name"] for row in rows] == list(truth)
        assert [row["role"] for row in rows[-2:]] == ["reference", "ground"]
        found = np.array([[float(row[axis]) for axis in "xyz"] for row in rows])
        error = np.linalg.norm(found - np.array(list(truth.values())), axis=1)
        assert np.median(error) < 1.5 and error.max() < 4, error
        assert (folder / "sub-01.png").exists() and (folder / "summary.tsv").exists()
