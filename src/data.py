"""Data handling: download Sen1Floods11, read chips and labels, preprocess them, and feed them to PyTorch.

Where this file sits in the pipeline (download -> baselines -> train -> evaluate -> figures):
    - `python main.py download` calls `download()` to fetch the dataset once.
    - Every other step uses `get_split()` to know which chips belong to train / val / test / Bolivia,
      `read_chip()` / `read_label()` to load them, and `clean_db()` to remove bad values.
    - `src/train.py` and `src/evaluate.py` use `compute_norm_stats()`, `normalize()` and `FloodDataset`
      to turn chips into the tensors the U-Net needs.

Background for readers new to SAR (Synthetic Aperture Radar):
    - Sentinel-1 is a radar satellite. It sends microwaves down and measures how much comes back ("backscatter").
    - VV and VH are two "polarisations": VV = sent vertical, received vertical; VH = sent vertical, received
      horizontal. Each chip therefore has 2 bands (2 grey-scale images of the same place).
    - Values are in decibels (dB), a log scale: 0 dB = strong echo, -25 dB = very weak echo.
      Calm water acts like a mirror, so it returns very little signal and is DARK (very negative dB).
    - "Speckle" is the grainy noise that every radar image has.

Local layout on disk (inside paths.data_dir from config.yaml):
    S1Hand/{Event}_{id}_S1Hand.tif        2 bands (VV, VH), float32, already in dB
    LabelHand/{Event}_{id}_LabelHand.tif  1 band, int16: -1 = no data, 0 = not water, 1 = water
    splits/flood_{train,valid,test,bolivia}_data.csv   lists of which chips go in which split
"""

# Lets us write type hints like `str | Path` on older Python versions too.
from __future__ import annotations

import csv  # reads the split CSV files
import json  # parses the JSON answer of the Google Cloud Storage list API
import urllib.parse  # builds URLs safely (query strings, escaping file names)
import urllib.request  # downloads files over HTTP (standard library, no extra package)
from dataclasses import dataclass  # quick way to define a small record class (Sample)
from pathlib import Path  # object-oriented file paths that work on Windows and Linux

import numpy as np  # fast arrays; every image here is a numpy array
import rasterio  # reads GeoTIFF files (the image format satellite data comes in)
import torch  # PyTorch: deep-learning library used for the U-Net
from torch.utils.data import Dataset  # base class PyTorch expects for a dataset

# --------------------------------------------------------------------------- download
# URL of the Google Cloud Storage "list objects" API for the public bucket "sen1floods11".
# Calling it with a prefix returns the names of all files inside that folder.
BUCKET = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"  # public bucket, list API
# Base URL to download a single file from the same bucket: FILES + "<file name inside bucket>".
FILES = "https://storage.googleapis.com/sen1floods11/"  # public bucket, file download
# Mapping: name of our local sub-folder -> the folder (prefix) inside the bucket it comes from.
REMOTE = {  # local folder -> folder inside the bucket
    "S1Hand": "v1.1/data/flood_events/HandLabeled/S1Hand/",  # the SAR images (VV, VH)
    "LabelHand": "v1.1/data/flood_events/HandLabeled/LabelHand/",  # the hand-drawn water masks
    "splits": "v1.1/splits/flood_handlabeled/",  # CSV files defining train/val/test/Bolivia
}


def download(data_dir: str | Path) -> None:
    """Download the hand-labeled chips, labels and split CSVs from the public Google bucket.

    How: for each of the 3 remote folders, ask the bucket's list API for every file name, then download
    each file that is not on disk yet. Running it twice is safe: existing files are skipped.

    Args:
        data_dir: local folder to save into (e.g. "data/sen1floods11"). Sub-folders are created automatically.
    Returns:
        Nothing; files are written to disk and a file count is printed per folder.
    """
    # Accept either a string or a Path and turn it into a Path so we can use the "/" operator to join paths.
    data_dir = Path(data_dir)
    # Loop over the 3 folders: `local` is our folder name (e.g. "S1Hand"), `prefix` is the bucket folder.
    for local, prefix in REMOTE.items():
        # Build the list-API URL. "prefix" filters to one folder; "fields=items(name)" asks only for file names.
        # The list API returns at most 1000 names per call; this subset has < 1000 files per folder.
        url = BUCKET + "?" + urllib.parse.urlencode({"prefix": prefix, "fields": "items(name)"})
        # Call the API, parse the JSON reply, and keep just the "name" of each item.
        # `.get("items", [])` returns an empty list if the folder is empty instead of crashing.
        names = [o["name"] for o in json.loads(urllib.request.urlopen(url).read()).get("items", [])]
        # Download every file in this folder, one by one.
        for name in names:
            # Local destination: data_dir / local folder / file name only (drop the long bucket path).
            dest = data_dir / local / Path(name).name
            # Skip the file if we already have it, or if the "name" is a folder marker (ends with "/").
            if dest.exists() or name.endswith("/"):
                continue  # already downloaded, or a folder placeholder
            # Make sure the local folder exists (create parents too; no error if it already exists).
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Download the file bytes (quote() escapes special characters in the name) and write them to disk.
            dest.write_bytes(urllib.request.urlopen(FILES + urllib.parse.quote(name)).read())
        # Report how many files are now in this local folder.
        print(f"{local}: {len(list((data_dir / local).glob('*')))} files")


# --------------------------------------------------------------------------- reading
# @dataclass auto-writes __init__ / __repr__ for us; frozen=True makes instances read-only (cannot be changed).
@dataclass(frozen=True)
class Sample:
    """One chip (a 512x512 SAR image tile) and the paths to its image and label files.

    Example: Sample("Ghana_103272", ".../S1Hand/Ghana_103272_S1Hand.tif", ".../LabelHand/Ghana_103272_LabelHand.tif")
    """

    chip_id: str  # unique chip name "<Event>_<number>", e.g. "Ghana_103272"
    image_path: Path  # path to the 2-band SAR GeoTIFF (VV, VH in dB)
    label_path: Path  # path to the 1-band label GeoTIFF (-1 / 0 / 1)

    # @property lets us write `sample.event` (no brackets) although it is computed on the fly.
    @property
    def event(self) -> str:
        """Name of the flood event (country) the chip comes from, used for per-event scores."""
        # Split the id at "_" and keep the first part.
        return self.chip_id.split("_")[0]  # "Ghana_103272" -> "Ghana"


def get_split(cfg: dict, split: str) -> list[Sample]:
    """Read one official split CSV and return the list of chips in it.

    Each CSV row looks like "Ghana_103272_S1Hand.tif,Ghana_103272_LabelHand.tif".

    Args:
        cfg: the config dict from config.yaml (needs paths.data_dir and data.splits).
        split: one of "train", "val", "test", "bolivia" (keys of data.splits in config.yaml).
    Returns:
        List of Sample objects, in the order of the CSV.
    """
    # Root folder of the dataset on disk.
    data_dir = Path(cfg["paths"]["data_dir"])
    # The list we will fill with one Sample per CSV row.
    samples = []
    # Open the CSV for this split; config.yaml maps e.g. "train" -> "flood_train_data.csv".
    # newline="" is the recommended way to open files for the csv module.
    with open(data_dir / "splits" / cfg["data"]["splits"][split], newline="") as f:
        # csv.reader gives each line as a list of strings, e.g. ["Ghana_103272_S1Hand.tif", "...LabelHand.tif"].
        for row in csv.reader(f):
            # Ignore blank lines or lines whose first column is not a .tif file name.
            if not row or not row[0].endswith(".tif"):
                continue  # skip empty lines
            # Path(...).stem drops ".tif"; split("_")[:2] keeps "<Event>" and "<number>"; join them back with "_".
            chip_id = "_".join(Path(row[0]).stem.split("_")[:2])  # "Ghana_103272_S1Hand" -> "Ghana_103272"
            # Build the full paths of the image and label from the chip id and store them as a Sample.
            samples.append(Sample(chip_id,
                                  data_dir / "S1Hand" / f"{chip_id}_S1Hand.tif",
                                  data_dir / "LabelHand" / f"{chip_id}_LabelHand.tif"))
    return samples


def read_chip(path: str | Path) -> np.ndarray:
    """Load a SAR chip from a GeoTIFF file.

    Args:
        path: path to an S1Hand .tif file.
    Returns:
        float32 array of shape (2, H, W): band 0 = VV, band 1 = VH, values in dB. May contain NaN (missing pixels).
    """
    # `with` opens the file and closes it automatically when the block ends.
    with rasterio.open(path) as src:
        # src.read() with no argument reads ALL bands -> shape (bands, height, width).
        return src.read().astype(np.float32)


def read_label(path: str | Path) -> np.ndarray:
    """Load a hand-drawn label from a GeoTIFF file.

    Args:
        path: path to a LabelHand .tif file.
    Returns:
        int16 array (H, W) with values -1 = no data (ignore), 0 = not water, 1 = water.
    """
    with rasterio.open(path) as src:
        # src.read(1) reads only band 1 (rasterio counts bands from 1) -> shape (height, width).
        return src.read(1).astype(np.int16)


# --------------------------------------------------------------------------- preprocessing
def clean_db(img: np.ndarray, clip=(-50.0, 1.0)) -> np.ndarray:
    """Clip extreme dB values and replace missing values (NaN) with the band median.

    Why clip: a few pixels have absurd values (e.g. -80 dB or +20 dB) that would distort statistics.
    Why the median for NaN (and not e.g. -50 dB): a very low value would look like dark water and fool the models.

    Args:
        img: (2, H, W) float32 SAR chip in dB, may contain NaN.
        clip: (low, high) range in dB; values outside are set to the nearest limit.
    Returns:
        New (2, H, W) float32 array with no NaN and every value inside `clip`.
    """
    # np.clip limits every value to [low, high]; it returns a new array (the input is untouched) and keeps NaN.
    out = np.clip(img, clip[0], clip[1]).astype(np.float32)  # np.clip returns a copy and keeps NaN
    # Loop over the 2 bands (VV then VH). Each `band` is a view of `out`, so editing it edits `out`.
    for band in out:  # each band is a view, so editing it edits `out`
        # Boolean mask (H, W): True where the value is NaN or infinite ("~" means NOT).
        bad = ~np.isfinite(band)
        # Only do work if this band actually has bad pixels.
        if bad.any():
            # If there is at least one good pixel, fill bad pixels with the median of the good ones;
            # otherwise (the whole band is missing) fall back to the low clip value.
            band[bad] = np.median(band[~bad]) if (~bad).any() else clip[0]
    return out


def compute_norm_stats(samples: list[Sample], clip=(-50.0, 1.0)) -> dict:
    """Mean and standard deviation of each band (VV, VH) over all pixels of the given chips.

    Called with the TRAIN chips only, so no information from val/test leaks into training.
    How: keep running sums (sum and sum of squares) so all chips never have to be in memory at once:
    mean = sum / n, std = sqrt(sum_sq / n - mean^2).

    Args:
        samples: chips to compute the statistics on (the train split).
        clip: same dB range as in clean_db.
    Returns:
        {"mean": [mean_VV, mean_VH], "std": [std_VV, std_VH]} as plain lists (so they can be saved to JSON/checkpoint).
    """
    # total = running sum per band, total_sq = running sum of squares per band, n = number of pixels per band.
    total, total_sq, n = np.zeros(2), np.zeros(2), 0
    # Go through every training chip once.
    for s in samples:
        # Load + clean the chip, flatten each band to one long row: shape (2, H*W).
        # float64 avoids rounding errors when summing millions of values.
        x = clean_db(read_chip(s.image_path), clip).reshape(2, -1).astype(np.float64)
        # Add this chip's per-band sum (axis=1 sums along the pixels).
        total += x.sum(axis=1)
        # Add this chip's per-band sum of squared values.
        total_sq += (x**2).sum(axis=1)
        # Count the pixels (x.shape[1] = H*W).
        n += x.shape[1]
    # Average value of each band.
    mean = total / n
    # Variance = E[x^2] - (E[x])^2; std is its square root. .tolist() turns numpy arrays into plain Python lists.
    return {"mean": mean.tolist(), "std": np.sqrt(total_sq / n - mean**2).tolist()}


def normalize(img: np.ndarray, stats: dict) -> np.ndarray:
    """Z-score each band: (x - mean) / std, so both bands have roughly mean 0 and std 1.

    Neural networks train better when inputs are centred around 0 with a similar scale.

    Args:
        img: (2, H, W) cleaned chip in dB.
        stats: output of compute_norm_stats (train statistics).
    Returns:
        (2, H, W) float32 normalised chip.
    """
    # Turn the 2 means into shape (2, 1, 1). [:, None, None] adds two size-1 axes so numpy "broadcasts"
    # the per-band value over every pixel of the (2, H, W) image.
    mean = np.array(stats["mean"], dtype=np.float32)[:, None, None]
    # Same for the standard deviations.
    std = np.array(stats["std"], dtype=np.float32)[:, None, None]
    return (img - mean) / std


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Data augmentation: random flips and 90-degree rotations, applied identically to image and mask.

    Why: it shows the model "new" versions of the same chips, so it learns that water is water whatever
    its orientation, and overfits less. Image and mask must get the SAME transform so they stay aligned.

    Args:
        img: (C, H, W) image. mask: (H, W) label. rng: numpy random generator (seeded -> reproducible).
    Returns:
        (img, mask) transformed, both as contiguous arrays.
    """
    # 50% chance: horizontal flip (reverse the width axis; "::-1" means "read backwards").
    if rng.random() < 0.5:  # horizontal flip
        img, mask = img[:, :, ::-1], mask[:, ::-1]
    # 50% chance: vertical flip (reverse the height axis).
    if rng.random() < 0.5:  # vertical flip
        img, mask = img[:, ::-1, :], mask[::-1, :]
    # Random integer 0, 1, 2 or 3 = how many quarter turns to rotate.
    k = rng.integers(0, 4)  # rotate k * 90 degrees
    # Rotate the image in the (height, width) plane (axes 1 and 2) and the mask in its only plane.
    img, mask = np.rot90(img, k, axes=(1, 2)), np.rot90(mask, k)
    # Flips/rotations create strided views; torch needs contiguous memory.
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class FloodDataset(Dataset):
    """PyTorch Dataset: gives one ready-to-use (image, mask, chip_id) triple per index.

    A DataLoader (in train.py / evaluate.py) asks this class for items by index and groups them into batches.
    Item i = (image tensor [2, H, W] float32 normalised, mask tensor [H, W] int64 with -1/0/1, chip_id string).
    """

    def __init__(self, samples: list[Sample], stats: dict, train: bool = False, seed: int = 0):
        """Store what is needed to load items later (nothing is read from disk here).

        Args:
            samples: chips of one split. stats: train normalisation statistics.
            train: True = apply random augmentation (only for the training split).
            seed: seed of the random generator used for augmentation.
        """
        self.samples = samples  # list of Sample (paths to image + label)
        self.stats = stats  # normalisation statistics from the train split
        self.train = train  # augmentation only for training
        self.rng = np.random.default_rng(seed)  # random generator for augment(); seeded for reproducibility

    def __len__(self) -> int:
        """Number of chips; PyTorch uses this to know how many items exist."""
        return len(self.samples)

    def __getitem__(self, i: int):
        """Load, clean, normalise (and maybe augment) chip number i, and convert it to tensors."""
        # The i-th chip record.
        s = self.samples[i]
        # Read the SAR image -> clip/fill NaN -> z-score with the train statistics. Shape (2, H, W).
        img = normalize(clean_db(read_chip(s.image_path)), self.stats)
        # Read the label (H, W) with -1 / 0 / 1.
        mask = read_label(s.label_path)
        # Only the training dataset is augmented; val/test chips are used exactly as they are.
        if self.train:
            img, mask = augment(img, mask, self.rng)
        # torch.from_numpy wraps the arrays as tensors (no copy). Masks become int64, the type the loss expects.
        return torch.from_numpy(img), torch.from_numpy(mask.astype(np.int64)), s.chip_id
