# Pose-Aware Multimodal Automatic Tagging on Greek Traditional Music

This repository contains the code developed for the thesis project **“Pose-Aware Multimodal Automatic Tagging on Greek Traditional Music”**.

The project investigates whether **pose-derived motion information**, together with **audio** and **video**, can improve the **automatic tagging of Greek traditional music performances**. More specifically, it explores how embodied performance cues such as **dance movement**, **posture**, and **visual context** can complement acoustic information in a culturally grounded music information retrieval setting.

The repository includes code for:

- **dance-scene detection**,
- **skeleton extraction from in-the-wild videos**,
- **unimodal processing pipelines** for audio, video, and skeleton modalities, and
- **multimodal fusion** across all combinations of audio, video, and skeleton representations.

The implemented framework supports automatic tagging on the **top-28 labels of the Lyra Dataset**, using both unimodal and multimodal models.

---

## Research Motivation

Automatic tagging is a core task in Music Information Retrieval (MIR), yet most existing systems operate on **audio alone**. This is often sufficient for many tagging tasks, but it can be limiting in performance-centered musical traditions, where semantic information is conveyed not only through sound, but also through **movement**, **gesture**, **dance**, **costume**, and **visual scene context**.

This is particularly relevant in **Greek traditional music**, where musical meaning is frequently embodied in performance practice. Regional dance styles, body motion, stage setting, and visible instruments may all carry information related to tags such as **genre**, **instrumentation**, or **geographic origin**.

The **Lyra Dataset** provides an appropriate testbed for such a study, as it contains videos of Greek traditional music performances annotated with multilabel tags. While previous work on Lyra has established strong **audio-based baselines**, its **visual** and **pose-related** dimensions remain largely unexplored.

This repository addresses that gap by introducing a **pose-aware multimodal setting** for automatic tagging on Greek traditional music.

---

## Main Contributions

This work makes the following main contributions:

1. **A pose-aware multimodal extension of Lyra**  
   The repository extends the prior audio-only use of the Lyra Dataset by introducing aligned **video representations** and **pose-derived skeleton streams**, enabling the study of audio, video, and motion jointly.

2. **An automated skeleton extraction pipeline for in-the-wild dance footage**  
   A full processing pipeline is provided for extracting primary-dancer skeleton sequences from unconstrained broadcast recordings. The pipeline combines:
   - dance-scene detection,
   - multi-person tracking,
   - primary-dancer selection,
   - pose estimation, and
   - quality-aware skeleton filtering.

3. **A multimodal analysis of the contribution of dancer pose**  
   The repository supports experiments with unimodal, bimodal, and trimodal systems, allowing the study of whether skeleton-based motion contributes complementary information beyond audio and raw video.

4. **Robust handling of missing or noisy skeleton information**  
   Since skeletons are extracted from real-world videos and are often noisy or partially missing, the multimodal setting incorporates a **missing-data mechanism based on learnable missing tokens**.

---

## Overview

Greek traditional music performances are inherently **multimodal**. In addition to the musical signal itself, they often include rich visual and embodied information such as:

- dance movement,
- posture and gesture,
- visible instruments,
- stage composition,
- costume and regional visual cues.

This repository focuses on building computational tools that can process such heterogeneous information and support **automatic semantic analysis and tagging**.

A particular emphasis is placed on **skeleton-based representations** as an additional modality. Because these skeletons are extracted from **in-the-wild broadcast videos**, they are often noisy, incomplete, or unavailable for some clips. For this reason, the project treats skeletons as a challenging but potentially informative modality in multimodal fusion.

The audio processing pipeline builds upon prior work conducted on the Lyra Dataset:

[https://github.com/pxaris/ccml/tree/main](https://github.com/pxaris/ccml/tree/main)

At its current stage, the repository includes:

- dance-scene detection scripts,
- skeleton extraction utilities,
- skeleton-based unimodal modeling,
- video embedding extraction and video classification modules,
- multimodal fusion pipelines for audio, video, and skeleton modalities.

---

## Repository Structure

```text
.
├── detect_dance_scenes/
├── extract_skeletons/
├── skeletons/
├── video/
├── multimodal/
└── README.md
```

### Folder Description

- **detect_dance_scenes/**  
  Training and inference code for identifying dance-related scenes in video recordings.

- **extract_skeletons/**  
  Utilities for pose estimation, person tracking, primary-dancer selection, and skeleton extraction from video material.

- **skeletons/**  
  Skeleton-based unimodal processing, embedding generation, training, and evaluation.

- **video/**  
  Video embedding extraction, unimodal video training, evaluation, and late-fusion utilities.

- **multimodal/**  
  Architectures and training scripts for multimodal fusion across audio, video, and skeleton representations.

---

## Methodological Pipeline

The overall workflow implemented in this repository can be summarized as follows:

1. **Dance-scene detection**
2. **Skeleton / pose extraction**
3. **Video preprocessing**
4. **Feature extraction**
5. **Unimodal modeling**
6. **Multimodal fusion**
7. **Automatic tagging / classification**

---

## Dataset

All experiments in this thesis are based on the **Lyra Dataset**, a dataset of Greek traditional music performances annotated with multilabel semantic tags.

The full Lyra collection contains **1570 videos**. Among them, **767 videos include dancing** and form the initial **dance subset**. Since the pose-aware setting requires at least one valid extracted skeleton clip per video, the skeleton extraction and filtering stage reduces this subset to **749 videos**, referred to as the **skeleton subset**.

Following prior work on Lyra, this project begins from the **top-30 most frequent labels**. However, in the skeleton subset, two of these labels have zero positive support, so all experiments in this repository are conducted on the remaining **28 labels**.

Depending on the experiment, the code may require:

- video recordings,
- annotation or label files,
- pretrained checkpoints,
- metadata associated with the Lyra Dataset.

This repository does **not necessarily include the raw datasets** used in the experiments. Users are expected to provide their own local data paths and organize the required files according to the input requirements of each script.

---

## Usage

### Train the Dance-Scene Detector

The following script fine-tunes the dance-scene detection model. The detector is trained on **1-second clips**.

```bash
python detect_dance_scenes/train_dance_detector.py \
  --video-dir <VIDEO_DIR> \
  --labels-file <LABELS_FILE> \
  --output-dir <OUTPUT_DIR> \
  --epochs <NUM_EPOCHS> \
  --mode full
```

### Apply Dance-Scene Detection

The following script applies a trained detector to identify dance-related scenes in video recordings.

```bash
python -m detect_dance_scenes/main.py \
  --video-dir <VIDEO_DIR> \
  --model-path <MODEL_PATH> \
  --output-dir <OUTPUT_DIR>
```

This stage processes the input videos, detects scene boundaries, performs clip-level inference, and stores the detected dance-scene intervals in output files.

---

### Skeleton-Based Processing

This stage includes:

1. trimming videos to their detected dance scenes,
2. applying **ByteTrack** for multi-person tracking,
3. selecting the **primary dancer**,
4. applying **AlphaPose** for pose estimation, and
5. storing keypoints together with metadata in `.json` format.

```bash
python -m extract_skeletons/main.py
```

---

### Create Skeleton Embeddings

This step creates skeleton embeddings for the training, validation, and test splits after selecting **T = 32 skeletons** from each clip.

The selection process is based on a quality-aware pipeline including:

- normalization,
- interpolation,
- joint confidence,
- bone-length consistency,
- left-right symmetry,
- temporal jitter penalties,
- skeleton similarity.

```bash
python -m skeletons/main.py cr_embeddings --set train --device cuda
python -m skeletons/main.py cr_embeddings --set val --device cuda
python -m skeletons/main.py cr_embeddings --set test --device cuda
```

---

### Train the STGCN-like Skeleton Model

The skeleton-based model used in this work is a lightweight GCN architecture that includes:

- ST-GCN-like blocks with **64 channels**,
- multi-scale temporal convolutions with kernel sizes **9** and **3**,
- residual connections,
- global average pooling,
- a linear multilabel classification head.

In contrast to deeper adaptive variants, the graph adjacency remains fixed throughout training, which keeps the model lightweight and helps reduce overfitting when noisy skeleton inputs are used.

```bash
python -m skeletons/main.py train --model_name STGCN --device cuda
```

### Evaluate the STGCN-like Skeleton Model

```bash
python -m skeletons/main.py eval --model_name STGCN --device cuda
```

---

### Extract Video Embeddings

In this setup, video embeddings are extracted using one of five pretrained video models.

```bash
python -m video/extract_video_embeddings/extract_embeddings.py --dataset "lyra" --audio_model_name "ast" --seed {42, 123, 1337, 2024, 9999} --model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --device {"cpu", "cuda"}
```

---

### Train the Video Model

In this setup, the video model is trained using frozen embeddings extracted in the previous step.

```bash
python -m video/train.py --dataset "lyra" --time_window "8.00" --subset {"True", "False"} --embs "frozen" --seed {42, 123, 1337, 2024, 9999} --model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --device {"cpu", "cuda"}
```

---

### Evaluate the Video Model

```bash
python -m video/eval.py --dataset "lyra" --time_window "8.00" --subset {"True", "False"} --embs "frozen" --seed {int} --model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --device {"cpu", "cuda"}
```

---

### Late Fusion

Late fusion is applied by aggregating the output probabilities of each modality after extracting modality-specific predictions.

```bash
python -m video/late_fusion.py --modalities {'a,v', 'a,s', 'v,s', 'a,v,s'} --fusion {"weighted", "mean", "sum"} --weights {"equal", "f1_macro"} --dataset "lyra" --time_window "8.00" {--subset} --seed {int} --video_model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --skeleton_model_name "STGCN"
```

---

### Train a Multimodal Model

The repository supports multiple multimodal fusion settings. In the current implementation:

- **simple transformer** performs early fusion,
- **gated fusion** learns clip-level modality weighting,
- **cross-attention fusion** is inspired by MulT-style cross-modal interaction.

```bash
python -m mutimodal/transfomrer.py --dataset "lyra" --time_window "8.00" {--subset} --seed {int} --standardize --model_name {"seq_transformer_avs_masked", "seq_transformer_as_masked", "seq_transformer_vs_masked", "seq_transformer_av"} --transformer {"simple, "gated", "cros_attention"} --device {"cpu", "cuda"}
```

### Evaluate a Multimodal Model

```bash
python -m mutimodal/transfomrer.py --dataset "lyra" --time_window "8.00" {--subset} --seed {int} --standardize --model_name {"seq_transformer_avs_masked", "seq_transformer_as_masked", "seq_transformer_vs_masked", "seq_transformer_av"} --transformer {"simple, "gated", "cros_attention"} --device {"cpu", "cuda"} --eval_only
```

---

## Key Findings

The experimental findings of the associated study can be summarized as follows:

- **Audio remains the strongest unimodal modality** for automatic tagging on Lyra.
- **Video provides useful complementary information**, especially in multimodal settings.
- **Skeletons are weak in isolation**, but still capture meaningful embodied cues.
- **Skeleton-based motion becomes useful under multimodal fusion**, especially for semantically embodied tag categories such as **genre** and **geographic style**.
- The best trimodal systems outperform the strongest audio-only baseline by +4.1 macro ROC-AUC, showing that **pose-aware multimodal fusion can improve automatic tagging performance**.

---

## Outputs

Depending on the selected script, the repository may produce:

- extracted skeleton or pose representations,
- detected dance-scene intervals,
- skeleton embedding files,
- video embedding files,
- trained model checkpoints,
- prediction files,
- evaluation reports.

---

## Reproducibility

The codebase is designed to support reproducible experimentation while allowing users to define their own environment-specific settings.

Users can specify their own:

- input paths,
- output paths,
- model paths,
- label files,

without modifying hard-coded values in the source code.

Placeholders such as the following should be replaced with user-specific values:

- `<VIDEO_DIR>`
- `<LABELS_FILE>`
- `<OUTPUT_DIR>`
- `<MODEL_PATH>`
- `<NUM_EPOCHS>`

---

## Notes and Limitations

- Some scripts require pretrained models or preprocessed intermediate data.
- The exact structure of the input data may vary depending on the experiment.
- Skeleton extraction is performed on **unconstrained broadcast material**, so pose detections can be noisy, incomplete, or missing.
- The current pipeline selects a **single primary dancer**, which improves robustness but may under-represent ensemble choreography in multi-person dance scenes.
- It is recommended to inspect each module individually before running the full pipeline.

---

## Academic Context

This repository was developed in the context of academic research on **multimodal analysis** and **automatic tagging of Greek traditional music performances**.

It is intended to support experimental work on:

- dance-related visual cues,
- pose-derived motion representations,
- skeleton-based sequence modeling,
- unimodal learning pipelines,
- multimodal fusion strategies for semantic music-performance tagging.

More broadly, the project contributes to the study of **culturally grounded MIR**, where musical meaning is distributed across sound, image, and embodied performance.

---

## Citation

If you use this repository in academic work, please cite the corresponding thesis and/or paper once available.

---

## Author

**Alexandros Alexiou**
