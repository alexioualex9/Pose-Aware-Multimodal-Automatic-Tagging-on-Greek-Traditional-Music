# Pose-Aware Multimodal Automatic Tagging on Greek Traditional Music

This repository contains the code developed for the thesis project **“Pose-Aware Multimodal Automatic Tagging on Greek Traditional Music”**.

The goal of this work is to investigate how pose-related visual information and additional modalities can be exploited for the automatic tagging and analysis of **Greek traditional music performances**.  
The repository includes code for **skeleton extraction**, **dance-scene detection**, and **unimodal processing pipelines**, which can be used either independently or as parts of a larger multimodal framework.

---

## Overview

Greek traditional music performances often combine multiple sources of information, including movement, posture, visual context, and musical content.  
This project focuses on building computational tools that can process such information and support automatic semantic analysis and tagging.

The current repository includes:

- skeleton extraction utilities,
- dance-scene detection scripts,
- unimodal processing modules.

---

## Repository Structure

```text
.
├── extract_skeletons/
├── detect_dance_scenes/
├── Unimodals/
│   └── skeletons/
└── README.md
```

### Folder Description

- **extract_skeletons/**  
  Code related to pose and skeleton extraction from video material.

- **detect_dance_scenes/**  
  Code for training and applying models that detect dance-related scenes in video recordings.

- **Unimodals/**  
  Code related to unimodal experiments and processing pipelines.

- **Unimodals/skeletons/**  
  Skeleton-based unimodal processing components.

---

## Methodological Outline

The overall workflow represented in this repository can be summarized as follows:

1. **Video preprocessing**
2. **Skeleton / pose extraction**
3. **Dance-scene detection**
4. **Feature extraction**
5. **Unimodal processing**
6. **Multimodal fusion**
7. **Automatic tagging / classification**

Each component can be studied separately or integrated into a broader end-to-end system.

---

## Data

This repository does not necessarily include the raw datasets used in the experiments.  
Users are expected to provide their own data paths and organize the required files according to the input requirements of each script.

Depending on the experiment, the code may use:

- video recordings,
- annotation or label files,
- trained checkpoints,
- metadata associated with Greek traditional music performances.

---

## Usage

### Train dance-scene detector

The following script can be used to fine-tune the dance-scene detection model.

```bash
python detect_dance_scenes/train_dance_detector.py \
  --video-dir <VIDEO_DIR> \
  --labels-file <LABELS_FILE> \
  --output-dir <OUTPUT_DIR> \
  --epochs <NUM_EPOCHS> \
  --mode full
```


### Apply dance-scene detection

The following script applies a trained model in order to detect dance-related scenes in video recordings.

```bash
python -m detect_dance_scenes/main.py \
  --video-dir <VIDEO_DIR> \
  --model-path <MODEL_PATH> \
  --output-dir <OUTPUT_DIR>
```

This step processes the input videos, detects scene boundaries, applies clip-level inference, and stores the detected dance-scene intervals in output files.

### Skeleton-based processing

This step includes trimming videos in their dance scenes, then apply ByteTrack in order to multi-track each person, select primary dancer, apply AlphaPose in order to get skeleton of dancer and finally store keypoints with metadata in .json file.

```bash
python -m extract_skeletons/main.py
```

### Create skeleton embeddings from .json file for train, validation and test sets

```bash
python -m skeletons/main.py cr_embeddings --set train --device cuda
python -m skeletons/main.py cr_embeddings --set val --device cuda
python -m skeletons/main.py cr_embeddings --set test --device cuda
```


### Train STGCN-like model for autotagging task

```bash
python -m skeletons/main.py train --model_name STGCN --device cuda
```

### Evaluate STGCN-like model for autotagging task

```bash
python -m skeletons/main.py eval --model_name STGCN --device cuda
```


---

## Outputs

Depending on the script, the repository may produce:

- extracted skeleton or pose representations,
- detected dance-scene intervals,
- trained model checkpoints,
- prediction files,
- evaluation reports.

---

## Reproducibility

The scripts are designed so that users can define their own:

- input paths,
- output paths,
- model paths,
- label files,

without editing hard-coded paths inside the source code.

Users should replace placeholders such as:

- `<VIDEO_DIR>`
- `<LABELS_FILE>`
- `<OUTPUT_DIR>`
- `<MODEL_PATH>`
- `<NUM_EPOCHS>`

with their own local paths and experiment settings.

---

## Notes

- Some scripts may require pre-trained models or preprocessed data.
- The exact structure of the input data may vary depending on the experiment.
- It is recommended to inspect each module separately before running the complete pipeline.

---

## Academic Context

This repository was developed in the context of academic research on **multimodal analysis** and **automatic tagging of Greek traditional music performances**.

It is intended to support experimental work on dance-related visual cues, skeleton-based representations, and modality-specific or multimodal learning approaches.

---

## Author

**Alexandros Alexiou**
