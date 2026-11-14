"""Each step on inputs whose answer is known."""

import numpy as np
import pytest
from synthetic_participant import CHANNELS, GROUND, MONTAGE, REFERENCE

from eegloc import geometry, layout
from eegloc.cap import Cap, fit, montage_layout
from eegloc.config import FIRST_BOUNDS, FIRST_SEEDS, MAP_SIZE, MM_PER_PIXEL, Study
from eegloc.epi import Holes, ItkAffine, find_holes
from eegloc.head import Head

UPRIGHT = dict(Cz=np.array([0.0, 0.0, 1.0]), C3=np.array([-1.0, 0, 0]), C4=np.array([1.0, 0, 0]))


def sphere_head(radius: float = 90.0) -> Head:
    """A head whose scalp is a sphere."""
    floor = np.full((MAP_SIZE, MAP_SIZE), radius)
    bump = np.where(geometry.map_disc(), 0.0, np.nan)
    return Head(np.eye(4), np.ones(3), np.zeros(3), floor.copy(), floor, bump)


def test_map_round_trip():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(200, 700, (2, 100))
    back = geometry.to_pixels(geometry.to_directions(x, y))
    np.testing.assert_allclose(back, (x, y), atol=1e-9)


def test_map_orientation():
    # Vertex at the middle, the nose up, the participant's right on the right.
    middle = (MAP_SIZE - 1) / 2
    np.testing.assert_allclose(geometry.to_pixels(np.array([0.0, 0, 1])), (middle, middle))
    assert geometry.to_pixels(np.array([1.0, 0, 0]))[0] > middle
    assert geometry.to_pixels(np.array([0.0, 1, 0]))[1] > middle


def test_itk_affine_works_in_lps(tmp_path):
    # A quarter turn about the superior axis around a centre, then a shift, written in ITK's axes.
    path = tmp_path / "xfm.txt"
    path.write_text(
        "#Insight Transform File V1.0\nTransform: AffineTransform_double_3_3\n"
        "Parameters: 0 -1 0 1 0 0 0 0 1 5 0 2\nFixedParameters: 10 0 0\n"
    )
    moved = ItkAffine.read(path).apply(np.array([[-11.0, 0.0, 3.0]]))  # LPS (11, 0, 3)
    np.testing.assert_allclose(moved, [[-15.0, -1.0, 5.0]])  # LPS (15, 1, 5)


def test_itk_affine_rejects_other_files(tmp_path):
    path = tmp_path / "xfm.txt"
    path.write_text("Parameters: 1 2 3\nFixedParameters: 0 0 0\n")
    with pytest.raises(ValueError, match="not an ITK affine"):
        ItkAffine.read(path)


def test_holes_are_found_where_the_signal_drops():
    rows, cols = np.mgrid[:MAP_SIZE, :MAP_SIZE]
    planted = np.array([[450, 450], [380, 520], [600, 300], [300, 400]])  # x, y
    layer = np.ones((MAP_SIZE, MAP_SIZE))
    for x, y in planted:
        layer[np.hypot(cols - x, rows - y) < 7.5 / MM_PER_PIXEL] = 0.1
    holes = find_holes(layer)
    found = np.column_stack([holes.x, holes.y])
    assert len(found) == len(planted)
    for hole in planted:
        assert np.abs(found - hole).max(axis=1).min() <= 1


def test_montage_layout_frame():
    sites = montage_layout(["Cz", "C4", "Fz"], MONTAGE)
    np.testing.assert_allclose(sites.angle[0], 0, atol=1e-6)
    np.testing.assert_allclose(sites.bearing[1], 0, atol=1e-6)  # C4: toward the right ear
    np.testing.assert_allclose(sites.bearing[2], np.pi / 2, atol=1e-6)  # Fz: toward the nose
    with pytest.raises(ValueError, match="no site named"):
        montage_layout(["Cz", "Nowhere"], MONTAGE)


def _cap_with_holes_at(pose):
    """A cap on a spherical head whose holes lie exactly under the sites at a pose."""
    head = sphere_head()
    sites = montage_layout([*CHANNELS, REFERENCE, GROUND], MONTAGE)
    everywhere = np.ones((MAP_SIZE, MAP_SIZE), bool)
    blank = Holes(np.array([450.0]), np.array([450.0]), np.zeros((2, 2)), everywhere)
    truth = Cap(head, blank, sites, UPRIGHT).placed(pose)
    x, y = geometry.to_pixels(truth / np.linalg.norm(truth, axis=1, keepdims=True))
    return Cap(head, Holes(x, y, np.zeros((2, 2)), everywhere), sites, UPRIGHT), truth


def test_fit_recovers_a_known_pose():
    pose = [6.0, -4.0, 5.0, 0.9, 0.86]
    cap, truth = _cap_with_holes_at(pose)
    assert cap.score(pose) == pytest.approx(1.0)
    fitted, score = fit(cap, FIRST_BOUNDS, FIRST_SEEDS)
    assert score == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(cap.placed(fitted) - truth, axis=1).max() < 0.5
    has_hole, nearest = cap.matched(cap.placed(fitted), 8.0)
    assert has_hole.all() and (nearest == np.arange(len(truth))).all()


def test_mirror_name():
    assert [layout.mirror_name(name) for name in ("Fp1", "FT10", "TP9", "Cz", "Fpz")] == [
        "Fp2",
        "FT9",
        "TP10",
        "Cz",
        "Fpz",
    ]


def test_layout_offsets_vanish_for_a_perfect_fit():
    pose = [2.0, 1.0, -3.0, 0.9, 0.9]
    cap, _ = _cap_with_holes_at(pose)
    east, north = layout.offsets(cap, pose)
    np.testing.assert_allclose(east, 0, atol=1e-6)
    np.testing.assert_allclose(north, 0, atol=1e-6)


def test_layout_learning_is_symmetric_and_needs_enough_participants():
    sites = montage_layout(["Cz", "C3", "C4", "Fz"], MONTAGE)
    # Six participants: C3 and C4 both sit 0.02 rad further out, and 0.01 rad to the front.
    east = np.tile([0.0, -0.02, 0.02, 0.0], (6, 1))
    north = np.tile([0.0, 0.01, 0.01, 0.0], (6, 1))
    learned = layout.learn(sites, east, north)
    np.testing.assert_allclose(learned.angle[1], learned.angle[2])
    np.testing.assert_allclose(learned.bearing[1], np.pi - learned.bearing[2])
    assert learned.angle[2] > sites.angle[2] and learned.bearing[2] > 0
    np.testing.assert_allclose(learned.angle[[0, 3]], sites.angle[[0, 3]], atol=1e-12)
    too_few = layout.learn(sites, east[:4], north[:4])
    np.testing.assert_allclose(too_few.angle, sites.angle, atol=1e-12)
    nobody = layout.learn(sites, east[:0], north[:0])
    np.testing.assert_allclose(nobody.angle, sites.angle, atol=1e-12)


def test_study_file_is_validated(tmp_path):
    path = tmp_path / "study.toml"
    body = (
        f'output = "{tmp_path}"\nsubjects = ["sub-01"]\nmontage = "easycap-M1"\n'
        f'[inputs]\nt1 = "{tmp_path}/{{subject}}_T1w.nii.gz"\n'
        f'epi = "{tmp_path}/{{subject}}_*_epi.nii.gz"\n'
        f'transform = "{tmp_path}/{{subject}}_*.txt"\nstart = "{tmp_path}/{{subject}}.tsv"\n'
    )
    path.write_text(body)
    study = Study.read(path)
    assert study.extra_sites == {}
    with pytest.raises(FileNotFoundError, match="expected one t1"):
        study.inputs("sub-01")
    for name in ("sub-01_T1w.nii.gz", "sub-01.tsv", "sub-01_1_epi.nii.gz", "sub-01_1.txt"):
        (tmp_path / name).touch()
    with pytest.raises(FileNotFoundError, match="at least two EPI runs"):
        study.inputs("sub-01")
    (tmp_path / "sub-01_2_epi.nii.gz").touch()
    with pytest.raises(FileNotFoundError, match="2 EPI runs but 1 transforms"):
        study.inputs("sub-01")
    path.write_text(body.replace("montage", "montages"))
    with pytest.raises(ValueError, match="unknown keys"):
        Study.read(path)
