# SAR Flood Mapping

Finds flooded areas in **Sentinel-1 radar (SAR) satellite images**. Radar sees through clouds and works at night, so it
is the standard sensor for mapping floods. Calm water reflects the radar signal away from the satellite and looks
**dark**. This project compares three methods for turning that signal into a water / not-water map, from simple to complex:

1. **Otsu threshold**: every pixel darker than one threshold is water (the classic method).
2. **Random Forest**: classifies each pixel from 5 simple features.
3. **U-Net**: a deep-learning model that also uses the shape and context around each pixel.

Data: [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11), 446 hand-labeled 512×512 image chips from 11 floods.
The model is scored on the official test split and on **Bolivia**, a flood event that is never used for training.

![predictions](outputs/figures/predictions_test.png)
*Blue = water found, orange = false alarm, purple = missed water.*

## Results
IoU of the water class (1.0 = perfect overlap with the hand label):

| Method | Test | Bolivia (unseen flood) |
|---|---|---|
| Otsu threshold, per chip | 0.21 | 0.35 |
| Otsu threshold, one global value (−12.45 dB) | 0.33 | 0.59 |
| Random Forest | 0.54 | 0.42 |
| **U-Net** | **0.68** | **0.64** |

![metrics](outputs/figures/metrics.png)

- The U-Net doubles the IoU of the threshold method on the test floods, mostly by removing false alarms (dark land such
  as smooth soil, sand or radar shadow).
- On the unseen Bolivia flood the gap is smaller, and the U-Net result depends on the training run: in earlier
  runs with different random seeds, Bolivia IoU ranged from 0.35 to 0.68. Simple physics-based thresholds are
  surprisingly strong on new events.

## Run it
```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python main.py download    # ~700 MB
python main.py all         # baselines -> train (~6 min on an RTX 4060 Ti) -> evaluate -> figures
python -m pytest           # 27 tests on fake data, no download needed
```

## Files
| File | What it does |
|---|---|
| `config.yaml` | all settings (paths, model, training) |
| `main.py` | runs the steps: `download`, `baselines`, `train`, `evaluate`, `figures`, `all` |
| `src/data.py` | download, read images/labels, clean and normalise, augmentation, PyTorch dataset |
| `src/baselines.py` | Otsu threshold and Random Forest |
| `src/model.py` | U-Net (ResNet18 encoder) and the loss function |
| `src/train.py` | training loop, keeps the model with the best validation IoU |
| `src/evaluate.py` | IoU / F1 / precision / recall, scores every method → `outputs/metrics.csv` |
| `src/visualize.py` | figures in `outputs/figures/` |
| `tests/` | tests for data, metrics, baselines, model and the full pipeline |
| `docs/GIAI_THICH.md` | detailed explanation (Vietnamese) |

## Limitations
- Sentinel-1 has 10 m pixels, so narrow rivers and water under vegetation are often invisible.
- Only one held-out flood event, and results on it vary between training runs.

## Data
Bonafilia et al. (2020), *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for
Sentinel-1*, CVPR Workshops. Contains modified Copernicus Sentinel data.
