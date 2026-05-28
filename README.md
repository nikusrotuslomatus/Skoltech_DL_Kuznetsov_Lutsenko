# Wheel Candidate Filtering

Lightweight binary classifier for filtering wheel candidate crops before a more
expensive virtual wheel try-on stage.

The model does not detect wheels on a full vehicle image. It receives an already
proposed crop and decides whether that crop is a usable wheel candidate.

```text
input:  candidate crop
output: 0 = invalid_candidate, 1 = valid_wheel_candidate
```

## Final Result

The selected model is `mobilenet_v3_large`. It has the lowest false-negative
count among the trained deep models, which is the main product constraint: a
false negative can remove a real wheel before downstream try-on processing.

Test set, fixed threshold chosen on validation:

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sklearn baseline | 0.125 | 0.669 | 0.646 | 0.761 | 0.699 | 0.702 | 0.715 | 76 | 56 | 32 | 102 |
| MobileNetV3-Large | 0.460 | 0.868 | 0.828 | 0.933 | 0.877 | 0.947 | 0.948 | 106 | 26 | 9 | 125 |
| EfficientNet-B0 | 0.405 | 0.865 | 0.836 | 0.910 | 0.871 | 0.950 | 0.951 | 108 | 24 | 12 | 122 |

Checkpoints:

```text
artifacts/deep/mobilenet_v3_large/best.pt    16.23 MB
artifacts/deep/efficientnet_b0/best.pt       15.57 MB
```

## Dataset

The project uses two cleaned crop exports:

```text
../labeled_crops_70
../labeled_crops/labeled_crops
```

Expected layout:

```text
labels.csv
valid_wheel_candidate/*.jpg
invalid_candidate/*.jpg
```

Final manifest:

```text
total samples:        1767
valid candidates:      884
invalid candidates:    883
source images:         586
missing files:           0
corrupt images:          0
label conflicts:         0
```

Grouped split by `source_relative_path` prevents crops from the same source
image from leaking across train, validation, and test.

```text
train:       1228 samples, 615 valid / 613 invalid
validation:  273 samples, 135 valid / 138 invalid
test:        266 samples, 134 valid / 132 invalid
```

## Repository Layout

```text
configs/default.yaml                 training and data configuration
src/wheel_filter/                    reusable pipeline code
scripts/01_audit_dataset.py          dataset audit
scripts/02_prepare_manifest.py       manifest and grouped split creation
scripts/03_train_sklearn_baseline.py baseline training
scripts/03_train_deep.py             PyTorch fine-tuning
scripts/04_evaluate_deep.py          metric recomputation
scripts/05_visualize_results.py      confusion matrix, ROC/PR, error grids
scripts/06_predict_deep.py           inference on crop files
scripts/07_plot_training_history.py  loss and validation-quality curves
scripts/08_benchmark_latency.py      optional latency benchmark
scripts/run_all.py                   end-to-end runner
artifacts/                           trained models, metrics, predictions
reports/                             figures and audit summaries
REPORT.md                            submission report
```

## Setup

Create an environment with Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch separately for the target machine. Example for CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Reproduce the Pipeline

Run from the repository root:

```bash
python scripts/run_all.py --models mobilenet_v3_large efficientnet_b0
```

For a quick data and baseline check without PyTorch:

```bash
python scripts/run_all.py --skip-deep
```

Windows PowerShell wrapper:

```powershell
.\scripts\run_pipeline.ps1 --models mobilenet_v3_large efficientnet_b0
```

## Generate Figures

```bash
python scripts/05_visualize_results.py \
  --predictions artifacts/deep/mobilenet_v3_large/test_predictions.csv \
  --out-dir reports/deep_mobilenet_v3_large

python scripts/07_plot_training_history.py
```

Key report figures:

```text
reports/deep_mobilenet_v3_large/confusion_matrix.png
reports/deep_mobilenet_v3_large/roc_curve.png
reports/deep_mobilenet_v3_large/pr_curve.png
reports/deep_mobilenet_v3_large/false_positives.jpg
reports/deep_mobilenet_v3_large/false_negatives.jpg
reports/training_history.png
```

## Predict on New Crops

```bash
python scripts/06_predict_deep.py \
  --checkpoint artifacts/deep/mobilenet_v3_large/best.pt \
  --input path/to/crop_or_folder \
  --out predictions.csv
```

## Optional Latency Benchmark

Run this on the target CPU/GPU after regenerating the manifest on that machine:

```bash
python scripts/08_benchmark_latency.py \
  --checkpoint artifacts/deep/mobilenet_v3_large/best.pt \
  --batch-size 32 \
  --samples 128
```

The script writes a JSON report to `reports/`.
