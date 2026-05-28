# Lightweight Wheel-Candidate Filtering for Virtual Wheel Try-On

## Problem

The project addresses one narrow step in a virtual wheel try-on pipeline. A
candidate-generation stage may propose many crop regions around possible wheels.
Some crops are real, usable wheels; others are false positives such as vehicle
body parts, background patches, reflections, partial objects, or badly localized
regions.

This project trains a compact classifier:

```text
f(crop) -> {invalid_candidate, valid_wheel_candidate}
```

The model is a candidate filter, not a full wheel detector. A full vehicle image
must first be converted into candidate crops by another stage.

## Motivation

Downstream try-on steps can use larger vision-language or image-generation
models. They are slower and more expensive than a local CNN. A lightweight
classifier can reject bad candidate crops early, reducing unnecessary downstream
calls and preserving valid wheel regions for later processing.

The key product risk is a false negative: rejecting a real wheel can break the
try-on pipeline. Precision still matters, but the final threshold is chosen with
valid-wheel recall as the priority.

## Dataset

Two manually labeled crop exports were cleaned and merged into one manifest.

```text
total samples:       1767
valid candidates:     884
invalid candidates:   883
unique source images: 586
```

Audit results:

```text
missing files:                 0
corrupt images:                0
duplicate label conflicts:     0
path/label mismatches:         0
tiny crops flagged:          123
huge crops flagged:           30
```

The split is grouped by `source_relative_path`, so crops from the same original
image cannot appear in different splits.

| Split | Samples | Valid | Invalid | Source images |
| --- | ---: | ---: | ---: | ---: |
| Train | 1228 | 615 | 613 | 410 |
| Validation | 273 | 135 | 138 | 88 |
| Test | 266 | 134 | 132 | 88 |

## Method

The pipeline consists of:

1. Dataset audit and label consistency checks.
2. Canonical manifest creation.
3. Leakage-safe grouped train/validation/test split.
4. Sklearn baseline with simple image features and logistic regression.
5. Fine-tuning compact pretrained CNN backbones.
6. Threshold selection on validation data.
7. Held-out test evaluation and visual error analysis.

Preprocessing:

```text
RGB conversion
square padding
resize to 224 x 224
ImageNet normalization
```

Training augmentations:

```text
random resized crop
horizontal flip
small rotation
brightness/contrast/saturation jitter
mild Gaussian blur
```

Deep training setup:

```text
optimizer: AdamW
learning rate: 1e-4
weight decay: 1e-4
loss: weighted cross entropy
scheduler: cosine
early stopping patience: 4
mixed precision: enabled on CUDA
threshold target: validation recall >= 0.90
```

## Models

The main comparison uses:

```text
sklearn logistic-regression baseline
MobileNetV3-Large
EfficientNet-B0
```

MobileNetV3-Large is selected as the final model because it has the lowest
false-negative count on the test set.

## Results

Test metrics:

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sklearn baseline | 0.125 | 0.669 | 0.646 | 0.761 | 0.699 | 0.702 | 0.715 | 76 | 56 | 32 | 102 |
| MobileNetV3-Large | 0.460 | 0.868 | 0.828 | 0.933 | 0.877 | 0.947 | 0.948 | 106 | 26 | 9 | 125 |
| EfficientNet-B0 | 0.405 | 0.865 | 0.836 | 0.910 | 0.871 | 0.950 | 0.951 | 108 | 24 | 12 | 122 |

MobileNetV3-Large improves the valid-wheel F1 score from `0.699` to `0.877`
compared with the baseline. It also reduces false negatives from `32` to `9`.

## Training Dynamics

Both deep models reach their best validation performance early.

MobileNetV3-Large:

```text
best epoch: 2
validation F1 at threshold 0.5: 0.867
validation recall at threshold 0.5: 0.896
```

EfficientNet-B0:

```text
best epoch: 2
validation F1 at threshold 0.5: 0.874
validation recall at threshold 0.5: 0.874
```

After the second epoch, training loss keeps decreasing while validation loss
starts rising. This is expected for a small and visually ambiguous dataset, and
early stopping prevents using the overfitted checkpoints.

## Error Analysis

Most false negatives are visually difficult valid wheels:

```text
small wheels
dark crops
blurred wheels
occluded wheels
unusual viewpoints
poorly localized regions
```

Most false positives are hard negatives:

```text
wheel-like circular structures
partial wheels marked as invalid
tire or rim fragments
vehicle body parts near the wheel
ambiguous crops where the label definition is debatable
```

This suggests that the next improvements should focus on relabeling ambiguous
samples and collecting more hard negatives/positives, rather than only trying a
larger model.

## Limitations

The current model classifies candidate crops only. It does not localize wheels
in a full vehicle image. A complete product pipeline would need a candidate
generator or object detector before this classifier.

The dataset is clean but still small. The current test set is suitable for a
course project, but a production system would need a larger external holdout
set, more vehicle types, and repeated evaluation across multiple random seeds.

The proposal also listed latency as a deployment-oriented metric. A benchmark
script is included as `scripts/08_benchmark_latency.py`; it should be run on the
target CPU/GPU after regenerating the manifest paths on that machine.

## Conclusion

The implemented pipeline matches the original proposal: it builds a cleaned
binary crop dataset, trains compact transfer-learning models, tunes the decision
threshold using validation data, and reports held-out test metrics with visual
error analysis.

The final MobileNetV3-Large model reaches `0.877` valid-wheel F1 and `0.933`
valid-wheel recall on the test set. It is suitable as a lightweight first-stage
candidate filter before expensive virtual wheel try-on components.
