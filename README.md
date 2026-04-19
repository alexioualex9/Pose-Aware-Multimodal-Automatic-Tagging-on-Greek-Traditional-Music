# Pose-Aware Multimodal Automatic Tagging on Greek Traditional Music

This repository contains the code developed for the thesis project **“Pose-Aware Multimodal Automatic Tagging on Greek Traditional Music”**.  
The main objective of this work is to study how pose-related visual information and additional modalities can be exploited for the automatic tagging and analysis of Greek traditional music performances.

## Overview

The repository includes modules for:

- skeleton extraction from video data,
- dance scene detection,
- unimodal processing pipelines,
- multimodal modeling components for automatic tagging.

The project is organized in a modular way so that each stage of the pipeline can be developed and evaluated independently.

## Project Goal

The broader goal of this work is to support automatic semantic analysis of Greek traditional music performances by combining movement-related cues with additional modalities.  
Special emphasis is placed on the identification of dance-related content and on the extraction of structured representations that can later be used for tagging or classification tasks.

## Repository Structure

```text
.
├── extract_skeletons/
├── detect_dance_scenes/
├── Unimodals/
└── README.md
