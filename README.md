# 🔊 Gunshot Detection from Audio

A deep learning pipeline for **binary gunshot audio classification** using PyTorch. The project combines multiple audio datasets, performs dataset auditing and label harmonization, extracts **PCEN-based acoustic features**, uses **group-aware stratified train/validation/test splitting**, and trains a **CBAM-CNN** classifier with GPU acceleration and mixed-precision training.

The final pipeline supports:

- Dataset path and file verification
- Audio metadata auditing
- Per-dataset label mapping
- Dataset combination and balancing
- Canonical PCEN feature extraction
- Group-aware stratified splitting
- PyTorch `Dataset` / `DataLoader`
- CBAM-CNN model training
- Validation and checkpoint selection
- Test-set evaluation
- Confusion matrix analysis
- ROC and Precision-Recall curves
- Source-level diagnostic analysis
- Single-file inference
- Folder-level batch inference

> **Project scope:** This repository is intended for research and educational experimentation in acoustic event classification. It is not a certified safety-critical gunshot detection system.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Definition](#problem-definition)
- [Key Features](#key-features)
- [System Pipeline](#system-pipeline)
- [Dataset Sources](#dataset-sources)
- [Dataset Preparation](#dataset-preparation)
- [Audio Metadata Audit](#audio-metadata-audit)
- [Label Mapping](#label-mapping)
- [Dataset Combination and Balancing](#dataset-combination-and-balancing)
- [Feature Extraction](#feature-extraction)
- [Train/Validation/Test Split](#trainvalidationtest-split)
- [PyTorch Dataset and DataLoader](#pytorch-dataset-and-dataloader)
- [CBAM-CNN Architecture](#cbam-cnn-architecture)
- [Training Configuration](#training-configuration)
- [GPU and Mixed Precision](#gpu-and-mixed-precision)
- [Model Selection and Checkpointing](#model-selection-and-checkpointing)
- [Evaluation](#evaluation)
- [Confusion Matrix](#confusion-matrix)
- [ROC Curve](#roc-curve)
- [Precision-Recall Curve](#precision-recall-curve)
- [Source-Level Diagnostic](#source-level-diagnostic)
- [Inference](#inference)
- [Generated Outputs](#generated-outputs)
- [Repository Structure](#repository-structure)
- [Running the Project on Kaggle](#running-the-project-on-kaggle)
- [Installation](#installation)
- [Reproducibility](#reproducibility)
- [Performance Reporting](#performance-reporting)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Responsible Use](#responsible-use)
- [Acknowledgements](#acknowledgements)
- [Author](#author)
- [License](#license)

---

## Project Overview

Gunshot detection from audio is an **acoustic event classification** problem. The objective of this project is to determine whether an audio sample contains a gunshot event.

The classifier uses two output classes:

| Label | Class |
|---:|---|
| `0` | Non-Gunshot |
| `1` | Gunshot |

The project was developed as a complete end-to-end machine-learning workflow rather than as a single model-training experiment. Particular attention is given to dataset consistency, source diversity, leakage prevention, evaluation, and inference.

---

## Problem Definition

### Input

An audio recording containing either a gunshot or a non-gunshot sound.

### Output

A binary prediction:

```text
0 → Non-Gunshot
1 → Gunshot
```

The inference pipeline also returns a class probability/confidence value.

Example:

```text
Prediction : Gunshot
Confidence : 97.42%
```

The confidence value should not be interpreted as a guaranteed probability of real-world correctness; it is the model's softmax output for the selected class.

---

## Key Features

### Dataset Engineering

- Multiple audio datasets can be combined into a master dataset.
- File paths are explicitly verified before training.
- Audio metadata is audited.
- Per-dataset label conventions are mapped into a common binary label space.
- Dataset/source information is retained for later diagnostic analysis.
- Group-aware splitting is used to reduce related-recording leakage.

### Audio Processing

- Audio files are loaded from the configured dataset paths.
- Acoustic features are extracted using a consistent feature-extraction pipeline.
- PCEN is used as the canonical feature representation.
- Features are converted into PyTorch tensors.

### Deep Learning

- PyTorch-based training.
- CBAM-CNN architecture.
- GPU/CUDA support.
- Automatic mixed precision when CUDA is available.
- Best-model checkpointing based on validation F1 score.

### Evaluation

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Average Precision
- Confusion matrix
- Normalized confusion matrix
- ROC curve
- Precision-Recall curve
- Source-level accuracy/error diagnostics

### Inference

- Single audio-file prediction.
- Optional feature visualization.
- Folder-level batch inference.
- CSV export of inference results.

---

## System Pipeline

```text
                  ┌────────────────────────┐
                  │   Multiple Audio       │
                  │       Datasets         │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Path / File Verification│
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Audio Metadata Audit    │
                  │ Sample Rate / Channels  │
                  │ Duration / Format       │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Per-Dataset Label       │
                  │ Mapping                 │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Combine + Balance       │
                  │ Master Dataset          │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ PCEN Feature Extraction │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Group-Aware Stratified  │
                  │ Train / Val / Test      │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ PyTorch Dataset /       │
                  │ DataLoader              │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │       CBAM-CNN          │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ GPU + Mixed Precision   │
                  │ Training                │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Best Model by Val F1    │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Test Evaluation         │
                  │ Metrics + Curves        │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Single / Folder         │
                  │ Inference               │
                  └────────────────────────┘
```

---

## Dataset Sources

The Kaggle working environment used during development contains multiple audio datasets. The visible dataset inputs include:

- `clapping testing1`
- `DEMAND`
- `Esc-50`
- `GunFire_Dataset`
- `Gunshot audio dataset`
- `GunShot Audio Dataset for ShotGraphNet`
- `Gunshot Detection Dataset (Custom)`
- `gunshot testing1`
- `gunshot testing2`
- `newly gun`
- `UrbanSound8K`
- `Firearms Audio Dataset – 58 Gun Types`

The exact subset used in a particular experiment is determined by the dataset configuration and filtering performed in the notebook.

### Important Dataset Note

The repository should **not** assume that all of the above datasets are used in every experiment.

The notebook's dataset-audit and source columns should be treated as the authoritative record of which files entered a particular experiment.

Dataset licenses and redistribution terms must be checked independently before redistributing any audio files.

---

## Dataset Preparation

The dataset preparation stage performs the following operations:

1. Define dataset paths.
2. Verify that paths exist.
3. Enumerate supported audio files.
4. Inspect audio metadata.
5. Assign a canonical source identifier.
6. Map source-specific labels to:
   - `0 = Non-Gunshot`
   - `1 = Gunshot`
7. Combine the datasets.
8. Remove invalid/unusable records as required.
9. Balance or control source/class composition where appropriate.
10. Retain source and grouping information for leakage-aware splitting and diagnostics.

Example file verification:

```python
import os

path = "/kaggle/input/..."

print(os.path.exists(path))
```

A valid dataset file should return:

```text
True
```

---

## Audio Metadata Audit

Before model training, audio metadata should be inspected for consistency.

Typical metadata checks include:

- Sample rate
- Number of channels
- Duration
- Audio format
- File readability
- Missing/corrupt files

This step is important because combining recordings with substantially different acquisition properties can introduce unwanted dataset/source cues.

---

## Label Mapping

Different datasets may encode their classes using different folder names, labels, or naming conventions.

The project therefore converts source-specific labels into one canonical label space:

```text
0 → Non-Gunshot
1 → Gunshot
```

The mapping is performed before the master dataset is created.

This avoids accidentally treating different source-label conventions as if they were already identical.

---

## Dataset Combination and Balancing

After individual dataset auditing and label mapping, the valid records can be combined into a master dataframe.

A typical record contains information such as:

```text
filepath
label
source
group
```

The `source` field identifies the originating dataset.

The `group` field can be used to keep related recordings together during dataset splitting.

### Why source information is retained

A model can sometimes learn characteristics associated with a particular recording source instead of learning the intended acoustic event.

Examples of source-related characteristics can include:

- Microphone response
- Background noise
- Recording environment
- Compression characteristics
- Sampling conditions
- Consistent acoustic background

Source-level evaluation is therefore included in this project.

---

## Feature Extraction

The canonical feature extraction stage produces an acoustic representation suitable for the neural network.

The project uses **Per-Channel Energy Normalization (PCEN)** as part of the feature-extraction pipeline.

The resulting feature representation used by the dataset pipeline has the expected shape:

```text
[125, 64, 1]
```

It is converted to a PyTorch tensor and permuted to:

```text
[1, 125, 64]
```

For a batch size of 64:

```text
[64, 1, 125, 64]
```

where:

- `64` = batch size
- `1` = feature channel
- `125` = time dimension
- `64` = feature/frequency dimension

---

## Train/Validation/Test Split

The project uses a **group-aware stratified splitting strategy**.

The purpose is to preserve class balance while reducing the possibility that highly related recordings appear across training and evaluation sets.

Conceptually:

```text
Master Dataset
      │
      ├───────────────┐
      │               │
      ▼               ▼
   Groups          Labels
      │               │
      └───────┬───────┘
              ▼
     Group-aware stratification
              │
              ▼
       ┌──────┼──────┐
       ▼      ▼      ▼
     Train    Val    Test
```

The test set is not shuffled during evaluation, which allows the prediction order to remain aligned with the corresponding test dataframe when source-level diagnostics are generated.

---

## PyTorch Dataset and DataLoader

A custom `GunshotDataset` loads the prepared dataframe and returns:

```text
feature_tensor
label
```

The feature tensor for one sample is:

```text
[1, 125, 64]
```

With:

```python
BATCH_SIZE = 64
```

the expected batch shape is:

```text
[64, 1, 125, 64]
```

### Kaggle DataLoader Configuration

During debugging and notebook execution, the DataLoader uses:

```python
NUM_WORKERS = 0
```

This avoids multiprocessing cleanup issues observed in notebook environments.

Once the pipeline is stable, the number of workers can be benchmarked separately.

### Feature caching

Because audio feature extraction can be CPU-intensive, repeatedly calculating features inside `__getitem__()` can become a major training bottleneck.

A recommended optimization is to extract/cache features once before repeated multi-epoch training.

---

## CBAM-CNN Architecture

The project uses a **CBAM-CNN** architecture.

CBAM stands for **Convolutional Block Attention Module**.

The purpose of attention is to help the convolutional network emphasize informative feature regions.

Conceptually:

```text
Input Feature Map
       │
       ▼
   Convolution
       │
       ▼
   Feature Maps
       │
       ▼
  CBAM Attention
   ┌──────┴──────┐
   ▼             ▼
Channel       Spatial
Attention     Attention
   └──────┬──────┘
          ▼
   Refined Features
          │
          ▼
   Classification Head
          │
          ▼
   Non-Gunshot / Gunshot
```

The exact layer configuration should be taken from the model definition in the notebook rather than inferred from this README.

---

## Training Configuration

The demonstrated training configuration is:

```python
EPOCHS = 30
BATCH_SIZE = 64
NUM_WORKERS = 0
```

The optimizer, learning rate, scheduler, and loss function are configured in the notebook.

The training loop records:

- Training loss
- Validation loss
- Accuracy
- Precision
- Recall
- F1 score

Training history is exported to:

```text
training_history_corrected.csv
```

---

## GPU and Mixed Precision

The training pipeline automatically selects CUDA when available:

```python
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The model is moved to the selected device:

```python
model = model.to(DEVICE)
```

Input tensors are also moved to the device:

```python
images = images.to(
    DEVICE,
    non_blocking=True
)

labels = labels.to(
    DEVICE,
    non_blocking=True
)
```

### Mixed precision

When CUDA is available, automatic mixed precision can be enabled to reduce memory usage and potentially improve training throughput.

The Kaggle environment used during development provided NVIDIA T4 GPUs.

### Important performance consideration

GPU availability does not guarantee high GPU utilization.

Audio decoding and PCEN feature extraction are CPU-side operations. If features are generated during every `__getitem__()` call, the GPU may spend substantial time waiting for the CPU.

For faster training, cache/precompute features before multi-epoch training.

---

## Model Selection and Checkpointing

The best model is selected using **validation F1 score**.

Example:

```python
if f1 > best_f1:
    best_f1 = f1
    torch.save(
        model.state_dict(),
        SAVE_PATH
    )
```

The checkpoint is saved as:

```text
/kaggle/working/best_model_corrected.pth
```

This means the final model used for test evaluation is the model that achieved the highest validation F1 during training, rather than necessarily the final epoch.

---

## Evaluation

The test set is evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall fraction of correct predictions |
| Precision | Fraction of predicted gunshots that are actually gunshots |
| Recall | Fraction of actual gunshots detected |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Threshold-independent ranking/discrimination measure |
| Average Precision | Summary of the precision-recall curve |

### Test results

The final metrics are exported to:

```text
test_results_corrected.csv
```

Example structure:

```text
Metric,Value
Accuracy,0.xxxx
Precision,0.xxxx
Recall,0.xxxx
F1,0.xxxx
ROC_AUC,0.xxxx
Average_Precision,0.xxxx
```

> Replace the example values above with the actual values generated by your final experiment. Do not report placeholder values as measured results.

---

## Confusion Matrix

The confusion matrix is generated from:

```python
cm = confusion_matrix(
    all_labels,
    all_preds
)
```

The class order is:

```text
0 → Non-Gunshot
1 → Gunshot
```

Conceptually:

```text
                    Predicted
                 Non-Gunshot  Gunshot

Actual
Non-Gunshot          TN          FP

Gunshot              FN          TP
```

For this application:

- **True Negative (TN):** non-gunshot correctly classified.
- **False Positive (FP):** non-gunshot classified as gunshot.
- **False Negative (FN):** gunshot classified as non-gunshot.
- **True Positive (TP):** gunshot correctly classified.

The false-negative count is especially important when assessing missed gunshot events.

### Generated files

```text
confusion_matrix_corrected.png
confusion_matrix_normalized.png
```

---

## ROC Curve

The ROC curve is generated from the probability assigned to class `1`:

```text
class 1 = Gunshot
```

It plots:

```text
True Positive Rate
against
False Positive Rate
```

The area under the ROC curve is reported as ROC-AUC.

Generated file:

```text
roc_curve_corrected.png
```

---

## Precision-Recall Curve

The Precision-Recall curve is generated using the Gunshot probability.

It shows the relationship between:

- Precision
- Recall

Average Precision is also calculated.

Generated file:

```text
pr_curve_corrected.png
```

Precision-Recall analysis is useful when the class distribution is not perfectly balanced.

---

## Source-Level Diagnostic

The project includes a diagnostic that evaluates accuracy separately for each dataset source.

Example output fields:

```text
source
accuracy
n_samples
n_errors
accuracy_percent
```

This analysis is important because a high overall test score can hide large differences between sources.

### Why this matters

If one source has substantially lower accuracy than others, possible explanations include:

- Different microphone characteristics
- Different background conditions
- Domain shift
- Recording artifacts
- Label inconsistencies
- Insufficient diversity
- Source-specific shortcuts learned by the model

This diagnostic does **not** by itself prove dataset leakage. It is a signal for further investigation.

---

## Inference

The trained checkpoint can be used for single-file inference.

Example:

```python
pred, confidence = predict(
    audio_path,
    show_plot=True
)
```

The inference function:

1. Checks that the audio file exists.
2. Extracts the same feature representation used during training.
3. Converts it to a tensor.
4. Adds the batch dimension.
5. Moves the tensor to CPU/GPU.
6. Runs the trained model.
7. Applies softmax.
8. Returns the predicted class and confidence.
9. Optionally visualizes the feature representation.

### Example output

```text
============================================================
AUDIO PREDICTION
============================================================
File: example.wav
Prediction : Gunshot
Confidence : 97.42%
Non-Gunshot probability : 2.58%
Gunshot probability     : 97.42%
```

The numeric values above are illustrative only.

---

## Folder-Level Inference

The project also supports inference over an entire folder.

```python
results_df = predict_folder(
    folder_path
)
```

The output is saved as:

```text
inference_results_corrected.csv
```

Example structure:

| File | Prediction | Confidence | Confidence_Percent |
|---|---|---:|---:|
| audio_001.wav | Gunshot | 0.97 | 97.0 |
| audio_002.wav | Non-Gunshot | 0.91 | 91.0 |

---

## Generated Outputs

The Kaggle working directory contains the following main artifacts:

```text
/kaggle/working/
│
├── best_model_corrected.pth
├── confusion_matrix_corrected.png
├── inference_results_corrected.csv
├── pr_curve_corrected.png
├── roc_curve_corrected.png
├── test_results_corrected.csv
└── training_history_corrected.csv
```

Additional files may be generated depending on notebook execution and experiment configuration.

---

## Repository Structure

A recommended GitHub structure is:

```text
Gunshot-Detection/
│
├── README.md
├── Gunshot_Detection.ipynb
├── requirements.txt
├── .gitignore
│
├── results/
│   ├── confusion_matrix_corrected.png
│   ├── confusion_matrix_normalized.png
│   ├── roc_curve_corrected.png
│   ├── pr_curve_corrected.png
│   ├── test_results_corrected.csv
│   └── training_history_corrected.csv
│
└── models/
    └── best_model_corrected.pth
```

### Recommended Git policy

Do **not** commit the complete raw audio datasets to GitHub unless their licenses explicitly permit redistribution.

Large model checkpoints should also be handled carefully. If the checkpoint is large, use Git LFS or an appropriate model/artifact hosting service rather than ordinary Git blobs.

---

## Running the Project on Kaggle

The current notebook workflow is designed around the Kaggle environment.

### Step 1 — Create/open a Kaggle Notebook

Upload or open the project notebook.

### Step 2 — Add datasets

Use the Kaggle notebook's **Add Input** option to attach the required datasets.

The notebook should then expose them under:

```text
/kaggle/input/
```

### Step 3 — Verify paths

Run the dataset verification section before training.

For example:

```python
import os

print(os.path.exists(audio_path))
```

Expected result:

```text
True
```

### Step 4 — Enable GPU

In Kaggle Notebook settings, enable a GPU accelerator.

Verify:

```python
import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### Step 5 — Run the notebook sections

The notebook is organized into the following major sections:

```text
Section 1  — Setup
Section 2  — Dataset paths & verification
Section 3  — Audio metadata audit
Section 4  — Per-dataset label mapping
Section 5  — Combine + balance dataset sources
Section 6  — Canonical PCEN feature extraction
Section 7  — Group-aware stratified train/val/test split
Section 8  — PyTorch Dataset / DataLoader
Section 9  — Model: CBAM-CNN
Section 10 — Loss, optimizer, scheduler
Section 11 — Training loop
Section 12 — Test-set evaluation
Section 13 — Inference
```

This ordering is intentional: data quality and splitting are handled before model training and inference.

---

## Installation

For a local Python environment, install the dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The project uses packages including:

- Python
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Librosa
- tqdm

For CUDA-enabled PyTorch, install the PyTorch build appropriate for your GPU/CUDA environment rather than blindly relying on a generic CPU package.

---

## Reproducibility

For reproducible experiments, record:

- Dataset versions
- Dataset source names
- Dataset paths/configuration
- Label mappings
- Train/validation/test split strategy
- Random seeds
- Feature-extraction parameters
- Model architecture
- Optimizer
- Learning rate
- Scheduler
- Batch size
- Number of epochs
- GPU/CUDA environment
- Best checkpoint
- Final test metrics

A trained model should only be compared fairly against another model when the evaluation protocol and test set are held constant.

---

## Performance Reporting

When publishing final results, report at minimum:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
Average Precision
```

Also include:

- Confusion matrix
- ROC curve
- Precision-Recall curve
- Number of test samples
- Class distribution
- Source-level diagnostic results

### Recommended results table

| Metric | Score |
|---|---:|
| Accuracy | `YOUR_VALUE` |
| Precision | `YOUR_VALUE` |
| Recall | `YOUR_VALUE` |
| F1 | `YOUR_VALUE` |
| ROC-AUC | `YOUR_VALUE` |
| Average Precision | `YOUR_VALUE` |

Replace `YOUR_VALUE` with the values from `test_results_corrected.csv`.

---

## Limitations

Audio gunshot classification is affected by domain shift and recording conditions.

Potential sources of error include:

- Background noise
- Reverberation
- Distance from microphone
- Microphone/device differences
- Audio compression
- Sample-rate differences
- Multiple simultaneous sound events
- Fireworks and other impulsive sounds
- Dataset imbalance
- Dataset/source bias
- Label noise
- Differences between controlled and real-world recordings

A strong benchmark score does not automatically imply reliable real-world performance.

The source-level diagnostic is therefore an important part of the evaluation.

---

## Future Improvements

Potential future improvements include:

### Data

- Increase recording diversity.
- Add more environmental conditions.
- Add more microphone/device types.
- Improve label verification.
- Perform cross-dataset testing.
- Add difficult negative examples.

### Feature Engineering

- Compare PCEN with log-Mel spectrograms.
- Evaluate alternative time-frequency representations.
- Cache/precompute features.
- Investigate augmentation in feature and waveform domains.

### Modeling

- Compare CBAM-CNN against baseline CNN architectures.
- Evaluate CRNN architectures.
- Experiment with attention-based audio models.
- Tune model depth and receptive fields.
- Perform systematic hyperparameter optimization.

### Evaluation

- Cross-source validation.
- Leave-one-dataset-out testing.
- Calibration analysis.
- Threshold optimization.
- Confidence/error analysis.
- Per-class and per-source reporting.

### Deployment

- Optimize inference latency.
- Export the trained model to a deployment-friendly format.
- Build a lightweight inference API.
- Evaluate edge-device deployment.
- Add streaming/chunked audio inference.

---

## Responsible Use

This project is designed for **research and educational purposes**.

An audio classifier should not be treated as definitive evidence that a gunshot occurred. Real-world systems require extensive validation under varied acoustic conditions, careful threshold selection, monitoring of false positives and false negatives, and appropriate human oversight.

The model should not be used as the sole basis for emergency, law-enforcement, medical, or other high-stakes decisions without a dedicated safety and validation process.

---

## Acknowledgements

This project uses the Python scientific-computing and machine-learning ecosystem, including PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, Librosa, and tqdm.

The project also relies on publicly available audio datasets and custom data sources. Dataset owners and licenses should be credited according to the terms applicable to each dataset.

---

## Author

**Vaibhav Chaudhary**

GitHub: `vaibhavchau37`

---

## License

Add the license that matches the code and data you are legally able to distribute.

For example, if you choose the MIT License for your original source code, add a `LICENSE` file containing the standard MIT License text.

**Important:** A software license applied to this repository does not automatically grant redistribution rights for third-party audio datasets. Dataset licenses remain separate.

---

## Project Status

**Status:** Active research / development

The current pipeline includes:

- [x] Dataset verification
- [x] Audio metadata audit
- [x] Label harmonization
- [x] Dataset combination
- [x] PCEN feature extraction
- [x] Group-aware splitting
- [x] PyTorch DataLoader
- [x] CBAM-CNN
- [x] GPU training
- [x] Mixed precision
- [x] Best-model checkpointing
- [x] Test-set metrics
- [x] Confusion matrix
- [x] ROC curve
- [x] Precision-Recall curve
- [x] Source-level diagnostics
- [x] Single-file inference
- [x] Folder inference
- [ ] Cross-dataset generalization benchmark
- [ ] Real-time streaming inference
- [ ] Production deployment

---

## Citation / Reference

If this project is used in academic or research work, cite this repository and the original datasets used in the experiment according to their respective citation requirements.

For any dataset used by the project, the original dataset provider's citation and license should take precedence over a generic repository citation.
