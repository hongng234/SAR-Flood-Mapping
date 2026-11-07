# CLAUDE.md

Flood-water segmentation in Sentinel-1 SAR (VV, VH) on Sen1Floods11 hand-labeled chips.
Methods: Otsu threshold, per-pixel Random Forest, U-Net (ResNet18). Evaluated on test + held-out Bolivia event.

## Commands
```bash
python main.py download | baselines | train | evaluate | figures | all
python -m pytest                       # fake data, no download needed, ~10 s
python -m ruff check src tests main.py
```

## Layout
- `main.py`: CLI, `load_config`, `set_seed`. `config.yaml` holds every setting.
- `src/data.py` (download, reading, preprocessing, Dataset), `src/baselines.py`, `src/model.py` (U-Net + loss),
  `src/train.py`, `src/evaluate.py` (metrics + scoring all methods), `src/visualize.py`.
- `tests/conftest.py` writes a fake dataset with the real folder layout.

## Rules
1. No leakage: training reads only the train and val splits (a test enforces this). Normalisation, the Otsu threshold
   and the RF are fitted on train. The checkpoint is chosen by val IoU.
2. Label -1 = no data. It is ignored in the loss, the metrics and the baselines.
3. Chips are already in dB. Do not apply log10 again.
4. Keep the project small. Every function gets an English docstring/comment saying what it does and how.
