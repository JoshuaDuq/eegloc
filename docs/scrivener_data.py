"""The EEG-fMRI data of Scrivener et al. as a study of eegloc, with electrodes marked by hand.

    python docs/scrivener_data.py FOLDER

Scrivener, Malik, Lindner and Roesch shared a simultaneous EEG-fMRI experiment (osf.io/w6bh3)
recorded with a cap of the layout `easycap-M1` (63 channels, reference FCz, ground AFz), and
Scrivener and Reader (2022) marked every electrode by hand on the gel artifacts of the structural
images (osf.io/853kw). This script fetches what a study needs and prepares its inputs:

    FOLDER/raw/EBRCB1_<n>/    the defaced structural image and, of each of the five blocks, the 20
                              volumes in its middle (read from the archives without downloading
                              them whole: about 60 MB per participant instead of 1.6 GB)
    FOLDER/manual/            the marked positions, in the scanner coordinates of the structural
    FOLDER/derived/sub-<nn>/  the T1, and each block's reference image (the mean of its volumes,
                              each through the realignment SPM left in its header) with its rigid
                              transform to the T1 (mutual information, from the scanner's own
                              alignment of the two images)
    FOLDER/derived/start/     starting positions: the montage fitted to the scalp of fsaverage as
                              in examples/starting_positions.py, through an affine registration of
                              fsaverage's T1 to the participant's
    FOLDER/derived/study.toml the study file

The registrations need antspyx (`pip install antspyx`), which eegloc itself does not; they sample
the images at random, and a transform moves by up to about 1 mm at the scalp from one run of this
script to the next. No field map is applied to the EPI: the shared images do not carry the
parameters of the sequence that this would take. The marked positions are not used.
"""

import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
from scipy import ndimage

from eegloc.epi import LPS, ItkAffine

IMAGES, MARKED = "v6s2e", "xw2nd"  # components of the two OSF projects
VOLUMES = 20  # raw volumes taken from the middle of each block
MONTAGE, REFERENCE, GROUND = "easycap-M1", "FCz", "AFz"
CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "FC1",
    "FC2",
    "CP1",
    "CP2",
    "FC5",
    "FC6",
    "CP5",
    "CP6",
    "TP9",
    "TP10",
    "POz",
    "F1",
    "F2",
    "C1",
    "C2",
    "P1",
    "P2",
    "AF3",
    "AF4",
    "FC3",
    "FC4",
    "CP3",
    "CP4",
    "PO3",
    "PO4",
    "F5",
    "F6",
    "C5",
    "C6",
    "P5",
    "P6",
    "AF7",
    "AF8",
    "FT7",
    "FT8",
    "TP7",
    "TP8",
    "PO7",
    "PO8",
    "FT9",
    "FT10",
    "Fpz",
    "CPz",
]


def fetch(url, headers=None, tries=8):
    """The answer to a request, asked again after a pause when the server refuses or fails."""
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.url, response.headers, response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            print(f"   {error}; again in {20 * (attempt + 1)} s", flush=True)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"no answer from {url}")


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url):
        self.source, self.position = url, 0
        self.resolve()

    def resolve(self):
        self.url, headers, _ = fetch(self.source, {"Range": "bytes=0-0"})  # after redirections
        self.size = int(headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = {0: 0, 1: self.position, 2: self.size}[whence] + offset
        return self.position

    def readinto(self, buffer):
        if self.position >= self.size:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        span = {"Range": f"bytes={self.position}-{end}"}
        try:
            data = fetch(self.url, span, tries=2)[2]
        except RuntimeError:  # the signed address has expired
            self.resolve()
            data = fetch(self.url, span)[2]
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def listing(node, folder):
    """Name and identifier of the files in a folder of an OSF component."""
    url = f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/?page%5Bsize%5D=100"
    top = next(
        item for item in json.loads(fetch(url)[2])["data"] if item["attributes"]["name"] == folder
    )
    url, files = top["relationships"]["files"]["links"]["related"]["href"], []
    while url:
        page = json.loads(fetch(url)[2])
        files += [(item["attributes"]["name"], item["id"]) for item in page["data"]]
        url = page["links"].get("next")
    return top["attributes"]["path"], files


def fetch_images(raw):
    """The defaced structural image and the middle volumes of each block, of every participant."""
    for name, identifier in sorted(listing(IMAGES, "fMRI")[1]):
        if not name.endswith(".zip"):
            continue
        address = f"https://osf.io/download/{identifier}/"
        archive = zipfile.ZipFile(io.BufferedReader(HttpFile(address), 8 << 20))
        blocks = {}
        for info in archive.infolist():
            *_, folder, file = [""] + info.filename.split("/")
            if re.fullmatch(r"B\d+", folder) and re.fullmatch(r"f\w[\w-]*\.nii", file):
                blocks.setdefault(folder, []).append(info)
            elif re.fullmatch(r"s\w+_deface\.nii", file):
                blocks["anat"] = [info]
        for folder, infos in blocks.items():
            infos.sort(key=lambda info: info.filename)
            first = max((len(infos) - VOLUMES) // 2, 0)
            for info in infos[first : first + VOLUMES]:
                target = raw / name.removesuffix(".zip") / folder / Path(info.filename).name
                if not target.exists() or target.stat().st_size != info.file_size:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(info))
        print(name, flush=True)


def fetch_marked(manual):
    """The positions marked by hand: S<n>_electrodes.txt, as first shared and as updated."""
    path = listing(MARKED, "Co-ordinate files")[0]
    address = f"https://files.osf.io/v1/resources/{MARKED}/providers/osfstorage{path}?zip="
    archive = zipfile.ZipFile(io.BytesIO(fetch(address)[2]))
    for info in archive.infolist():
        *_, folder, file = [""] + info.filename.split("/")
        if folder in ("Original", "Updated") and re.fullmatch(r"S\d+_electrodes\.txt", file):
            (manual / folder).mkdir(parents=True, exist_ok=True)
            (manual / folder / file).write_bytes(archive.read(info))


def visible(folder, pattern):
    """The files of a folder that match, without the hidden ones some volumes add."""
    return sorted(path for path in folder.glob(pattern) if not path.name.startswith("."))


def reference(volumes):
    """Mean of a block's volumes on the grid of the first, each through its header's affine."""
    first = nib.load(volumes[0])
    total = np.zeros(first.shape)
    for path in volumes:
        image = nib.load(path)
        to_voxel = np.linalg.inv(image.affine) @ first.affine
        total += ndimage.affine_transform(
            image.get_fdata(), to_voxel[:3, :3], to_voxel[:3, 3], order=1, mode="nearest"
        )
    return nib.Nifti1Image((total / len(volumes)).astype(np.float32), first.affine)


def to_ants(image):
    """An ANTs image with the physical space nibabel reads, so that no header is read twice."""
    import ants

    image = nib.as_closest_canonical(image)
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    return ants.from_numpy(
        np.asarray(image.get_fdata(), dtype=np.float32),
        origin=(image.affine[:3, 3] * LPS).tolist(),
        spacing=spacing.tolist(),
        direction=LPS[:, None] * (image.affine[:3, :3] / spacing),
    )


def register(fixed, moving, kind, target, **options):
    """Register and keep the transform as ITK text: it maps fixed points to moving points."""
    import ants

    result = ants.registration(fixed, moving, type_of_transform=kind, random_seed=0, **options)
    ants.write_transform(ants.read_transform(result["fwdtransforms"][0]), str(target))


def template_positions():
    """The montage fitted to the scalp of fsaverage, in the scanner coordinates of its T1 (mm)."""
    subjects_dir = Path(mne.datasets.fetch_fsaverage(verbose=False)).parent
    info = mne.create_info(CHANNELS, 1000.0, "eeg")
    info.set_montage(MONTAGE)
    coregistration = mne.coreg.Coregistration(
        info, "fsaverage", subjects_dir, fiducials="estimated"
    )
    coregistration.fit_fiducials()
    coregistration.fit_icp(n_iterations=20, eeg_weight=1.0)
    head = np.array([channel["loc"][:3] for channel in info["chs"]])
    surface = mne.transforms.apply_trans(coregistration.trans, head) * 1e3
    t1 = nib.load(subjects_dir / "fsaverage" / "mri" / "T1.mgz")
    to_scanner = t1.header.get_vox2ras() @ np.linalg.inv(t1.header.get_vox2ras_tkr())
    return mne.transforms.apply_trans(to_scanner, surface), t1


def prepare(raw, derived):
    """The inputs of every participant, and the study file."""
    template_mm, template = template_positions()
    template = to_ants(nib.Nifti1Image(template.get_fdata().astype(np.float32), template.affine))
    subjects = []
    for folder in sorted(raw.glob("EBRCB1_*"), key=lambda path: int(path.name.split("_")[1])):
        subject = f"sub-{int(folder.name.split('_')[1]):02d}"
        stem = derived / subject / subject
        stem.parent.mkdir(parents=True, exist_ok=True)
        (shared,) = visible(folder / "anat", "s*_deface.nii")
        nib.save(nib.load(shared), f"{stem}_T1w.nii.gz")
        t1 = to_ants(nib.load(shared))
        for block in visible(folder, "B*"):
            mean = reference(visible(block, "f*.nii"))
            name = f"{stem}_block-{block.name[1:]}"
            nib.save(mean, f"{name}_boldref.nii.gz")
            register(
                t1,
                to_ants(mean),
                "Rigid",
                f"{name}_from-boldref_to-T1w_xfm.txt",
                initial_transform="Identity",
            )
        register(t1, template, "Affine", f"{stem}_from-template_to-T1w_xfm.txt")

        # The transform maps the participant's points to the template's, p' = R (p - c) + c + t in
        # ITK's axes: solved for p, it brings the template's positions to the participant.
        to_template = ItkAffine.read(f"{stem}_from-template_to-T1w_xfm.txt")
        moved = template_mm * LPS - to_template.centre - to_template.translation
        start = (moved @ np.linalg.inv(to_template.rotation).T + to_template.centre) * LPS
        (derived / "start").mkdir(exist_ok=True)
        rows = ["name\tx\ty\tz"] + [
            f"{name}\t{x:.3f}\t{y:.3f}\t{z:.3f}"
            for name, (x, y, z) in zip(CHANNELS, start, strict=True)
        ]
        (derived / "start" / f"{subject}_electrodes.tsv").write_text("\n".join(rows) + "\n")
        subjects.append(subject)
        print(subject, flush=True)

    files = f"{derived}/{{subject}}/{{subject}}"
    (derived / "study.toml").write_text(
        f'output = "{derived}/eegloc"\n'
        f"subjects = {json.dumps(subjects)}\n"
        f'montage = "{MONTAGE}"\nreference = "{REFERENCE}"\nground = "{GROUND}"\n\n'
        "[inputs]\n"
        f't1 = "{files}_T1w.nii.gz"\n'
        f'epi = "{files}_block-*_boldref.nii.gz"\n'
        f'transform = "{files}_block-*_from-boldref_to-T1w_xfm.txt"\n'
        f'start = "{derived}/start/{{subject}}_electrodes.tsv"\n'
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    fetch_marked(root / "manual")
    fetch_images(root / "raw")
    prepare(root / "raw", root / "derived")
