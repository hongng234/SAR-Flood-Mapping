"""Data: download Sen1Floods11, read chips/labels, preprocess, and the PyTorch Dataset.

Local layout (inside paths.data_dir):
    S1Hand/{Event}_{id}_S1Hand.tif        2 bands (VV, VH), float32, already in dB
    LabelHand/{Event}_{id}_LabelHand.tif  1 band, int16: -1 = no data, 0 = not water, 1 = water
    splits/flood_{train,valid,test,bolivia}_data.csv
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- download
BUCKET = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"  # public bucket, list API
FILES = "https://storage.googleapis.com/sen1floods11/"  # public bucket, file download
REMOTE = {  # local folder -> folder inside the bucket
    "S1Hand": "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "LabelHand": "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "splits": "v1.1/splits/flood_handlabeled/",
}


def download(data_dir: str | Path) -> None:
    """Download the hand-labeled chips, labels and split CSVs (skips files that already exist)."""
    data_dir = Path(data_dir)
    for local, prefix in REMOTE.items():
        # The list API returns at most 1000 names per call; this subset has < 1000 files per folder.
        url = BUCKET + "?" + urllib.parse.urlencode({"prefix": prefix, "fields": "items(name)"})
        names = [o["name"] for o in json.loads(urllib.request.urlopen(url).read()).get("items", [])]
        for name in names:
            dest = data_dir / local / Path(name).name
            if dest.exists() or name.endswith("/"):
                continue  # already downloaded, or a folder placeholder
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(urllib.request.urlopen(FILES + urllib.parse.quote(name)).read())
        print(f"{local}: {len(list((data_dir / local).glob('*')))} files")


# --------------------------------------------------------------------------- reading
@dataclass(frozen=True)
class Sample:
    """One chip and its label."""

    chip_id: str  # e.g. "Ghana_103272"
    image_path: Path
    label_path: Path

    @property
    def event(self) -> str:
        return self.chip_id.split("_")[0]  # "Ghana_103272" -> "Ghana"


def get_split(cfg: dict, split: str) -> list[Sample]:
    """Read an official split CSV. Each row is "chip.tif,label.tif"."""
    data_dir = Path(cfg["paths"]["data_dir"])
    samples = []
    with open(data_dir / "splits" / cfg["data"]["splits"][split], newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].endswith(".tif"):
                continue  # skip empty lines
            chip_id = "_".join(Path(row[0]).stem.split("_")[:2])  # "Ghana_103272_S1Hand" -> "Ghana_103272"
            samples.append(Sample(chip_id,
                                  data_dir / "S1Hand" / f"{chip_id}_S1Hand.tif",
                                  data_dir / "LabelHand" / f"{chip_id}_LabelHand.tif"))
    return samples


def read_chip(path: str | Path) -> np.ndarray:
    """SAR chip as float32 array (2, H, W): band 0 = VV, band 1 = VH, in dB. May contain NaN."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def read_label(path: str | Path) -> np.ndarray:
    """Label as int16 array (H, W) with values -1 / 0 / 1."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int16)


# --------------------------------------------------------------------------- preprocessing
def clean_db(img: np.ndarray, clip=(-50.0, 1.0)) -> np.ndarray:
    """Clip extreme values and replace NaN with the band median.

    The median (not e.g. -50 dB) is used so missing pixels do not look like dark water.
    """
    out = np.clip(img, clip[0], clip[1]).astype(np.float32)  # np.clip returns a copy and keeps NaN
    for band in out:  # each band is a view, so editing it edits `out`
        bad = ~np.isfinite(band)
        if bad.any():
            band[bad] = np.median(band[~bad]) if (~bad).any() else clip[0]
    return out


def compute_norm_stats(samples: list[Sample], clip=(-50.0, 1.0)) -> dict:
    """Mean and std of each band over all pixels of the given (TRAIN) chips.

    Uses running sums (sum and sum of squares) so all chips never have to be in memory at once:
    mean = sum / n, std = sqrt(sum_sq / n - mean^2).
    """
    total, total_sq, n = np.zeros(2), np.zeros(2), 0
    for s in samples:
        x = clean_db(read_chip(s.image_path), clip).reshape(2, -1).astype(np.float64)
        total += x.sum(axis=1)
        total_sq += (x**2).sum(axis=1)
        n += x.shape[1]
    mean = total / n
    return {"mean": mean.tolist(), "std": np.sqrt(total_sq / n - mean**2).tolist()}


def normalize(img: np.ndarray, stats: dict) -> np.ndarray:
    """Z-score each band: (x - mean) / std. [:, None, None] lets (2,) broadcast over (2, H, W)."""
    mean = np.array(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.array(stats["std"], dtype=np.float32)[:, None, None]
    return (img - mean) / std


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Random flips and 90-degree rotations, applied identically to image (C,H,W) and mask (H,W)."""
    if rng.random() < 0.5:  # horizontal flip
        img, mask = img[:, :, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:  # vertical flip
        img, mask = img[:, ::-1, :], mask[::-1, :]
    k = rng.integers(0, 4)  # rotate k * 90 degrees
    img, mask = np.rot90(img, k, axes=(1, 2)), np.rot90(mask, k)
    # Flips/rotations create strided views; torch needs contiguous memory.
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class FloodDataset(Dataset):
    """Returns (image [2,H,W] float32 normalised, mask [H,W] int64, chip_id)."""

    def __init__(self, samples: list[Sample], stats: dict, train: bool = False, seed: int = 0):
        self.samples = samples
        self.stats = stats  # normalisation statistics from the train split
        self.train = train  # augmentation only for training
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = normalize(clean_db(read_chip(s.image_path)), self.stats)
        mask = read_label(s.label_path)
        if self.train:
            img, mask = augment(img, mask, self.rng)
        return torch.from_numpy(img), torch.from_numpy(mask.astype(np.int64)), s.chip_id
