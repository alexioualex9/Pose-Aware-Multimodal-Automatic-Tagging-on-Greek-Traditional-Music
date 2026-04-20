# Pose-Aware Multimodal Automatic Tagging on Greek Traditional Music

This repository contains the code developed for the thesis project **“Pose-Aware Multimodal Automatic Tagging on Greek Traditional Music”**.

The goal of this work is to investigate how pose-related visual information and additional modalities can be exploited for the automatic tagging and analysis of **Greek traditional music performances**.  
The repository includes code for **skeleton extraction**, **dance-scene detection**, **unimodal processing pipelines** and **multimodal fusion** of all combinations of audio, video and skeleton modalities with four different multimodal models in order to do automatic tagging in top-N=28 labels of Lyra Dataset.

---

## Overview

Greek traditional music performances often combine multiple sources of information, including movement, posture, visual context, and musical content.  
This project focuses on building computational tools that can process such information and support automatic semantic analysis and tagging.
Skeletons are a missing modality, extracted from in-the-wild videos, so they are noisy and a missing-data mechanism that uses learnable missing tokens is being used in this project.
Audio analysis is borrowed from this initial project done on Lyra Dataset: https://github.com/pxaris/ccml/tree/main.

The current repository includes:

- dance-scene detection scripts,
- skeleton extraction utilities,
- unimodal processing modules,
- multimdoal fusion of audio, video and skeletons.

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

- **extract_skeletons/**  
  Code related to pose and skeleton extraction from video material.

- **detect_dance_scenes/**  
  Code for training and applying models that detect dance-related scenes in video recordings.

- **skeletons/**  
  Skeleton-based unimodal processing components.

---

## Methodological Outline

The overall workflow represented in this repository can be summarized as follows:

1. **Dance-scene detection**
2. **Skeleton / pose extraction**
3. **Video preprocessing**
4. **Feature extraction**
5. **Unimodal processing**
6. **Multimodal fusion**
7. **Automatic tagging / classification**

---

## Data

This repository does not necessarily include the raw datasets used in the experiments.  
Users are expected to provide their own data paths and organize the required files according to the input requirements of each 
script.
In all experiments of this thesis Lyra Dataset was used.

Depending on the experiment, the code may use:

- video recordings,
- annotation or label files,
- trained checkpoints,
- metadata associated with Lyra Dataset.
  
---

## Usage

### Train dance-scene detector

The following script can be used to fine-tune the dance-scene detection model. In order to do so, code uses clips of 1 sec.

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

This step processes the input videos, detects scene boundaries, applies clip-level inference, and stores the detected dance-scene 
intervals in output files.

### Skeleton-based processing

This step includes trimming videos in their dance scenes, then apply ByteTrack in order to multi-track each person, select 
primary dancer, apply AlphaPose in order to get skeleton of dancer and finally store keypoints with metadata in .json file.

```bash
python -m extract_skeletons/main.py
```

### Create skeleton embeddings

This section contains creating skeleton embeddings for training, validation and test sets after selecting T=32 skeletons from 
each clip. The selection of these T skeletons is based-on a pipeline including normalization, interpolation, joint confidence, 
bone-length consistency, left–right symmetry, temporal jitter penalties and skeleton similarity.

```bash
python -m skeletons/main.py cr_embeddings --set train --device cuda
python -m skeletons/main.py cr_embeddings --set val --device cuda
python -m skeletons/main.py cr_embeddings --set test --device cuda
```


### Train STGCN-like model for autotagging task

The STGCN-like model that was used is a lightweight GCN model that includes ST-GCN-like blocks with 64 channels, multi-scale 
temporal convolutions (kernel sizes 9 and 3), residual connections, and global average pooling followed by a linear 
classification head. In contrast to deeper adaptive variants, the adjacency remains fixed throughout, which keeps the model 
lightweight and reduces the risk of overfitting in case of using noisy skeletons.

```bash
python -m skeletons/main.py train --model_name STGCN --device cuda
```

### Evaluate STGCN-like model for autotagging task

```bash
python -m skeletons/main.py eval --model_name STGCN --device cuda
```

### Extract Video Embeddings

In this setup, video embeddings are being extracted using one of five video pre-trained models. 

```bash
python -m video/extract_video_embeddings/extract_embeddings.py --dataset "lyra" --audio_model_name "ast" --seed {42, 123, 1337, 2024, 9999} --model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --device {"cpu", "cuda"}
```

### Train Video Model

In this setup, video model is being trained using frozen embeddings that have been extracted in previous step. 

```bash
python -m video/train.py --dataset "lyra" --time_window "8.00" --subset {"True", "False"} --embs "frozen" --seed {42, 123, 1337, 2024, 9999} --model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --device {"cpu", "cuda"}
```

### Evaluate Video Model

Evaluation of video model.

```bash
python -m video/eval.py --dataset "lyra" --time_window "8.00" --subset {"True", "False"} --embs "frozen" --seed {int} --model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --device {"cpu", "cuda"}
```

### Late Fusion

Apply late fusion model by taking the average of each modality's outcome, after extracting each modality's probs.

```bash
python -m video/late_fusion.py --modalities {'a,v', 'a,s', 'v,s', 'a,v,s'} --fusion {"weighted", "mean", "sum"} --weights {"equal", "f1_macro"} --dataset "lyra" --time_window "8.00" {--subset} --seed {int} --video_model_name {"slowfast50", "timesformer", "vitb16", "resnet50", "videomae"} --skeleton_model_name "STGCN"
```

### Train Multimodal Model

Train multimodal model using one of four possible models. Simple transformer applies early fusion, while Gated applies a gate mechanism in order to give weights in each modality. Cross attention model is based-on MulT-style model.

```bash
python -m mutimodal/transfomrer.py --dataset "lyra" --time_window "8.00" {--subset} --seed {int} --standardize --model_name {"seq_transformer_avs_masked", "seq_transformer_as_masked", "seq_transformer_vs_masked", "seq_transformer_av"} --transformer {"simple, "gated", "cros_attention"} --device {"cpu", "cuda"}
```

### Evaluate Multimodal Model

Train multimodal model using one of four possible models. Simple transformer applies early fusion, while Gated applies a gate mechanism in order to give weights in each modality. Cross attention model is based-on MulT-style model.

```bash
python -m mutimodal/transfomrer.py --dataset "lyra" --time_window "8.00" {--subset} --seed {int} --standardize --model_name {"seq_transformer_avs_masked", "seq_transformer_as_masked", "seq_transformer_vs_masked", "seq_transformer_av"} --transformer {"simple, "gated", "cros_attention"} --device {"cpu", "cuda"} --eval_only
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
