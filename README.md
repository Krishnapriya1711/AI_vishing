# Real-Time Detection of AI-Cloned Voices in Vishing Calls

## Overview

This project focuses on detecting AI-generated / spoofed speech in
realistic voice-phishing (vishing) call conditions.

The system investigates whether AI-cloned voices can be detected when
speech is:

- Very short in duration
- Compressed by telephone communication channels
- Band-limited to telephone speech frequencies
- Encoded using telephone codecs
- Presented as a continuous streaming signal

The project also investigates generalization to Indian English and Hindi.

---

## Research Objective

Traditional audio deepfake detection systems often operate on relatively
clean and longer speech recordings.

This project focuses on a more realistic vishing scenario:

> Can AI-generated speech still be detected from short, compressed
> telephone speech?

The main research variables are:

1. Speech duration
2. Telephone bandwidth degradation
3. Codec compression
4. Sliding-window aggregation
5. Detection latency
6. Cross-domain and language generalization

---

## Dataset

The primary development dataset is:

**ASVspoof 2019 Logical Access (LA)**

The dataset contains:

- Bona fide human speech
- Spoofed / synthetic speech

The original ASVspoof train, development and evaluation splits are
preserved to prevent data leakage.

### Dataset statistics

| Split | Total | Bona fide | Spoof |
|------|------:|----------:|------:|
| Train | 25,380 | 2,580 | 22,800 |
| Development | 24,844 | 2,548 | 22,296 |
| Evaluation | 71,237 | 7,355 | 63,882 |
| Total | 121,461 | 12,483 | 108,978 |

The ASVspoof 2021 DF evaluation dataset is also retained as a separate
held-out dataset for later evaluation.

---

## Project Pipeline

```text
ASVspoof 2019 LA
       |
       v
Dataset Metadata
       |
       v
Original Train / Dev / Eval Split
       |
       v
Short-Window Metadata
       |
       v
+-----------------------------+
| Dynamic Audio Preprocessing |
+-----------------------------+
       |
       +------ Clean (16 kHz)
       |
       +------ Telephone (8 kHz)
       |
       +------ Telephone + G.711 A-law (8 kHz)
       |
       v
Short Speech Windows
0.5 / 0.75 / 1 / 1.5 / 2 / 3 sec
       |
       v
Feature Extraction
       |
       +------ MFCC
       +------ Mel Spectrogram
       +------ Pitch / Prosody
       |
       v
Baseline Models
       |
       +------ Logistic Regression
       +------ Random Forest
       +------ SVM
       |
       v
Lightweight CNN
       |
       v
Short-Duration Experiments
       |
       v
Phone / Codec Robustness
       |
       v
Sliding-Window Detection
       |
       v
Indian English / Hindi Evaluation
       |
       v
Real-Time Detection Demo