"""SAR Flood Mapping - command line entry point.

    python main.py download     # fetch the Sen1Floods11 hand-labeled data
    python main.py baselines    # fit Otsu threshold + Random Forest on the train split
    python main.py train        # train the U-Net (early stopping on validation IoU)
    python main.py evaluate     # score all methods on val / test / Bolivia
    python main.py figures      # data, training and prediction figures
    python main.py all          # every step in order
"""

# Note for readers: the text above is the module docstring. It is also shown by `python main.py --help`
# (it is passed to argparse below), so it is kept short. This file only reads the settings and calls the
# right function in src/ for each step; all real work happens in src/.

# Lets us write type hints like `str | Path` on older Python versions.
from __future__ import annotations

import argparse  # reads the command-line arguments (e.g. "train")
import random  # Python's built-in random generator (seeded in set_seed)
from pathlib import Path  # file paths

import numpy as np  # numpy's random generator (seeded in set_seed)
import torch  # PyTorch's random generator (seeded in set_seed)
import yaml  # reads config.yaml

# Absolute path of the folder that contains this file (the project root).
# __file__ = path of main.py; resolve() makes it absolute; .parent = its folder.
ROOT = Path(__file__).resolve().parent  # project folder
# The pipeline steps, in the order "all" runs them.
STEPS = ("download", "baselines", "train", "evaluate", "figures")


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    """Read config.yaml and turn the relative paths in it into absolute paths.

    Absolute paths mean the project works no matter which folder you run it from.

    Args:
        path: the YAML file to read (default: config.yaml in the project root).
    Returns:
        The settings as a nested dict, e.g. cfg["train"]["epochs"].
    """
    # Open the file as UTF-8 text; `with` closes it automatically.
    with open(path, encoding="utf-8") as f:
        # safe_load turns YAML text into Python dicts/lists/numbers (and never runs code from the file).
        cfg = yaml.safe_load(f)
    # For every entry under "paths:" (data_dir, output_dir) ...
    for key, value in cfg["paths"].items():
        # ... join it to the project root and store it back as a string.
        cfg["paths"][key] = str(ROOT / value)  # e.g. "outputs" -> <project>/outputs
    return cfg


def set_seed(seed: int) -> None:
    """Seed every random number generator so runs are reproducible (same seed -> same results).

    Randomness is used for pixel sampling, data shuffling, augmentation and the model's starting weights.
    """
    random.seed(seed)  # Python's own random module
    np.random.seed(seed)  # numpy's global generator
    torch.manual_seed(seed)  # also seeds CUDA


def run_step(step: str, cfg: dict) -> None:
    """Run one pipeline step. Imports are inside each branch so each step only loads what it needs.

    Args:
        step: one of STEPS. cfg: settings from load_config().
    """
    # Same seed before every step, so a step gives the same result whether run alone or within "all".
    set_seed(cfg["seed"])
    # Step 1: download the dataset into paths.data_dir.
    if step == "download":
        from src.data import download

        download(cfg["paths"]["data_dir"])
    # Step 2: fit the Otsu threshold and the Random Forest on the train split.
    elif step == "baselines":
        from src.baselines import fit_baselines

        fit_baselines(cfg)
    # Step 3: train the U-Net and print its summary (best epoch, best val IoU, minutes).
    elif step == "train":
        from src.train import train

        print(train(cfg))
    # Step 4: score all methods; print the table rounded to 3 decimals without row numbers.
    elif step == "evaluate":
        from src.evaluate import evaluate

        print(evaluate(cfg).round(3).to_string(index=False))
    # Step 5: draw the figures into outputs/figures/.
    elif step == "figures":
        from src.visualize import figures

        figures(cfg)


def main() -> None:
    """Parse the command line and run the requested step (or all steps)."""
    # The parser shows this file's docstring in --help; RawDescriptionHelpFormatter keeps its line breaks.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # One required argument: the step name, or "all". argparse rejects anything else with an error message.
    parser.add_argument("step", choices=(*STEPS, "all"))
    # Read sys.argv; args.step holds the chosen word.
    args = parser.parse_args()
    # Settings from config.yaml.
    cfg = load_config()
    # "all" runs every step in order; otherwise only the requested step.
    for step in STEPS if args.step == "all" else [args.step]:
        print(f"\n=== {step} ===")
        run_step(step, cfg)


# This block runs only when the file is started directly (python main.py ...), not when it is imported (e.g. by tests).
if __name__ == "__main__":
    main()
