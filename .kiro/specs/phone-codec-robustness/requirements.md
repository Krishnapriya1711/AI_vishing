# Requirements Document

## Introduction

This specification covers the AI Voice Vishing Detection pipeline for phone/codec robustness. The project detects AI-generated and spoofed speech in realistic voice-phishing (vishing) call conditions using the ASVspoof 2019 LA dataset. The existing codebase provides telephone bandpass filtering (300-3400 Hz, 8 kHz resampling), G.711 A-law codec simulation, dynamic audio loading across clean/telephone/telephone_codec conditions, and metadata files for 121,461 audio files across three conditions.

The work is divided into two phases. Phase 1 replicates a team member's pipeline: building a balanced evaluation dataset, extracting Mel spectrogram features, defining and evaluating a lightweight CNN model, and analyzing threshold behavior across telephone conditions. Phase 2 implements a second team member's tasks: verifying the clean baseline, investigating domain shift, training a robustness/adaptation model with proper data splits, comparing three channel conditions, tuning operating thresholds on validation data, and producing publication-quality analysis.

A key finding from Phase 1 is that the clean-trained Mel CNN does not transfer well to simulated telephone conditions, achieving approximately 50% accuracy with poor operating-point performance despite some ranking information (ROC-AUC ~0.67-0.69). Phase 2 addresses this through model adaptation and rigorous comparison.

## Glossary

- **ASVspoof_2019_LA**: The Logical Access subset of the ASVspoof 2019 dataset containing 121,461 audio files split into train (25,380), dev (24,844), and eval (71,237) partitions.
- **Bonafide**: A genuine human speech recording in the ASVspoof dataset, used as the positive class label.
- **Spoof**: A synthetic or manipulated speech recording in the ASVspoof dataset, used as the negative class label.
- **Phone_Simulator**: The existing module that applies a 6th-order Butterworth bandpass filter (300-3400 Hz) and downsamples audio to 8 kHz to simulate telephone channel conditions.
- **Codec_Simulator**: The existing module that applies G.711 A-law encode/decode to telephone-conditioned audio, simulating digital telephony codec degradation.
- **Audio_Loader**: The existing dynamic audio loader that loads raw FLAC files and optionally applies telephone filtering and codec simulation to produce clean, telephone, or telephone_codec audio.
- **Phone_Conditions_Metadata**: The CSV file (phone_conditions_metadata.csv) containing 364,383 rows that map each of the 121,461 audio files to three conditions (clean, telephone, telephone_codec) with sample rate and codec annotations.
- **Mel_Spectrogram**: A time-frequency representation of audio computed via short-time Fourier transform with a Mel-frequency filter bank, producing a 64x101x1 tensor suitable for CNN input.
- **Mel_CNN**: A lightweight convolutional neural network that takes 64x101x1 Mel spectrogram input and outputs a binary classification (bonafide vs spoof).
- **Phone_Robustness_Eval_Dataset**: A balanced set of 1,000 audio files (500 bonafide + 500 spoof) selected from the ASVspoof 2019 LA eval split, each rendered under telephone and telephone+codec conditions.
- **Domain_Shift**: The statistical distribution difference between clean audio features and telephone-conditioned audio features that causes a clean-trained model to degrade on telephone inputs.
- **Robustness_Model**: A Mel CNN trained or fine-tuned on telephone/codec-conditioned audio to maintain detection performance across channel degradations.
- **Operating_Threshold**: The decision boundary probability value above which a prediction is classified as spoof; the default is 0.5 but validation-derived thresholds may improve performance.
- **Clean_Baseline**: The performance metrics of the Mel CNN evaluated on clean-condition audio, serving as the reference for measuring degradation under telephone conditions.
- **Speaker_Disjoint_Split**: A data partitioning strategy where no speaker appears in more than one of the train, validation, or test sets, preventing data leakage.
- **G711_A_Law**: A standard companding algorithm for digital telephony that quantizes 16-bit PCM audio to 8-bit codes, introducing quantization distortion.

## Requirements

### Requirement 1: Build Balanced Phone Robustness Evaluation Dataset

**User Story:** As a researcher, I want a balanced evaluation dataset of telephone and telephone+codec audio, so that I can assess model performance under realistic vishing call conditions without class imbalance bias.

#### Acceptance Criteria

1. WHEN the build script is executed, THE Build_Phone_Robustness_Dataset SHALL select exactly 500 bonafide and 500 spoof files from the ASVspoof_2019_LA eval split
2. WHEN selected files are processed, THE Build_Phone_Robustness_Dataset SHALL generate telephone-condition audio (300-3400 Hz bandpass, 8 kHz resampling) for all 1,000 files into data/processed/phone_eval
3. WHEN selected files are processed, THE Build_Phone_Robustness_Dataset SHALL generate telephone+G711_A_Law audio for all 1,000 files into data/processed/phone_codec_eval
4. WHEN dataset generation is complete, THE Build_Phone_Robustness_Dataset SHALL write a metadata CSV (data/metadata/phone_robustness_eval_metadata.csv) containing file name, condition, label, split, speaker_id, and output path for every generated file
5. IF an audio file is too short for the required duration, THEN THE Build_Phone_Robustness_Dataset SHALL skip that file and log the skip

### Requirement 2: Extract Mel Spectrogram Features

**User Story:** As a machine learning engineer, I want Mel spectrogram feature extraction, so that I can convert raw audio into the 64x101x1 tensor representation expected by the CNN model.

#### Acceptance Criteria

1. WHEN audio is loaded for feature extraction, THE Mel_Spectrogram_Extractor SHALL accept any audio array and sample rate as input
2. WHEN Mel spectrogram extraction is performed, THE Mel_Spectrogram_Extractor SHALL produce a tensor with shape 64x101x1 (64 Mel bins, 101 time frames, 1 channel)
3. WHEN Mel spectrograms are computed, THE Mel_Spectrogram_Extractor SHALL downmix multi-channel audio to mono automatically before feature extraction
4. WHEN Mel spectrograms are computed, THE Mel_Spectrogram_Extractor SHALL apply consistent parameters across clean, telephone, and telephone_codec conditions so that all conditions produce identically shaped output
5. IF the audio is shorter than required for a full 101-frame spectrogram, THEN THE Mel_Spectrogram_Extractor SHALL pad the audio to produce the expected output shape

### Requirement 3: Build Mel Spectrogram Dataset from Metadata

**User Story:** As a machine learning engineer, I want to build a Mel spectrogram dataset from audio files using a metadata CSV, so that I can batch-process features for training and evaluation.

#### Acceptance Criteria

1. WHEN a metadata CSV is provided, THE Mel_Dataset_Builder SHALL read the CSV and process each listed audio file into a 64x101x1 Mel spectrogram
2. WHEN building the dataset, THE Mel_Dataset_Builder SHALL associate each Mel spectrogram with its label (bonafide or spoof) from the metadata
3. WHEN building the dataset, THE Mel_Dataset_Builder SHALL support filtering by condition (clean, telephone, telephone_codec) and by split (train, dev, eval)

### Requirement 4: Define Mel CNN Model Architecture

**User Story:** As a machine learning engineer, I want a lightweight CNN model definition for 64x101x1 input, so that I can train and evaluate a binary classifier for bonafide vs spoof detection.

#### Acceptance Criteria

1. WHEN the model is instantiated, THE Mel_CNN SHALL accept input of shape (64, 101, 1)
2. WHEN the model is instantiated, THE Mel_CNN SHALL produce a single sigmoid output representing spoof probability
3. WHEN the model is compiled, THE Mel_CNN SHALL use binary cross-entropy loss and an optimizer suitable for binary classification
4. WHEN the model architecture is defined, THE Mel_CNN SHALL be lightweight enough to train on a single GPU or CPU

### Requirement 5: Evaluate Phone Robustness of Clean-Trained Model

**User Story:** As a researcher, I want to evaluate the clean-trained Mel CNN on telephone and telephone+codec conditions, so that I can quantify the performance degradation caused by channel conditions.

#### Acceptance Criteria

1. WHEN evaluation is run, THE Phone_Robustness_Evaluator SHALL load models/mel_cnn_best.keras with input shape (64, 101, 1)
2. WHEN evaluation is run, THE Phone_Robustness_Evaluator SHALL load the balanced 1,000-sample metadata (phone_robustness_eval_metadata.csv)
3. WHEN predictions are generated, THE Phone_Robustness_Evaluator SHALL extract Mel spectrogram inputs for telephone and telephone_codec conditions and produce spoof probabilities
4. IF telephone extraction fails while telephone_codec extraction succeeds, THEN THE Phone_Robustness_Evaluator SHALL proceed with telephone_codec results only and flag the telephone condition as missing in the output
5. WHEN metrics are computed, THE Phone_Robustness_Evaluator SHALL calculate accuracy, precision, recall, F1 score, and ROC-AUC for both telephone and telephone_codec conditions
6. WHEN metrics are computed, THE Phone_Robustness_Evaluator SHALL generate confusion matrices and classification reports for both conditions
7. WHEN evaluation is complete, THE Phone_Robustness_Evaluator SHALL write results to data/results/phone_robustness_summary.csv and data/results/phone_robustness_detailed.csv

### Requirement 6: Analyze Decision Thresholds

**User Story:** As a researcher, I want to analyze alternative decision thresholds across 2,000 predictions, so that I can understand how threshold selection affects detection performance under telephone conditions.

#### Acceptance Criteria

1. WHEN threshold analysis is run, THE Threshold_Analyzer SHALL collect 2,000 predictions across telephone and telephone_codec conditions
2. WHEN thresholds are evaluated, THE Threshold_Analyzer SHALL test a range of threshold values and compute accuracy, precision, recall, and F1 for each
3. WHEN analysis is complete, THE Threshold_Analyzer SHALL write results to data/results/phone_threshold_analysis.csv

### Requirement 7: Verify and Preserve Clean Baseline

**User Story:** As a researcher, I want to evaluate the Mel CNN on clean audio and record baseline metrics, so that I have a reference point for measuring telephone condition degradation.

#### Acceptance Criteria

1. WHEN clean baseline evaluation is run, THE Clean_Baseline_Evaluator SHALL extract Mel spectrograms from clean-condition audio and run the Mel CNN to produce spoof probabilities
2. WHEN clean baseline metrics are computed, THE Clean_Baseline_Evaluator SHALL calculate accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix
3. WHEN clean baseline results are stored, THE Clean_Baseline_Evaluator SHALL persist results in a structured format (CSV) that enables direct comparison with telephone and telephone_codec results

### Requirement 8: Verify Mel Pipeline Consistency Across Conditions

**User Story:** As a researcher, I want to confirm that telephone and codec audio are converted to the same 64x101x1 representation, so that I can rule out feature extraction inconsistency as a source of performance degradation.

#### Acceptance Criteria

1. WHEN Mel spectrograms are extracted from telephone audio, THE Pipeline_Verifier SHALL confirm the output shape is (64, 101, 1)
2. WHEN Mel spectrograms are extracted from telephone_codec audio, THE Pipeline_Verifier SHALL confirm the output shape is (64, 101, 1)
3. WHEN Mel spectrograms are extracted from clean audio, THE Pipeline_Verifier SHALL confirm the output shape is (64, 101, 1)
4. WHEN the pipeline is verified, THE Pipeline_Verifier SHALL always generate a report that confirms consistency or reports any shape or value range discrepancies, and set a pipeline_verified flag to true when no discrepancy is found

### Requirement 9: Investigate Domain Shift Between Clean and Telephone Conditions

**User Story:** As a researcher, I want to compare Mel spectrogram statistics between clean and telephone audio, so that I can understand why the clean-trained CNN fails on telephone conditions.

#### Acceptance Criteria

1. WHEN domain shift analysis is run, THE Domain_Shift_Analyzer SHALL compute per-bin Mel spectrogram statistics (mean, standard deviation) for clean, telephone, and telephone_codec conditions
2. WHEN domain shift analysis is run, THE Domain_Shift_Analyzer SHALL compute a quantitative measure of distribution difference between clean and telephone conditions
3. WHEN domain shift analysis is complete, THE Domain_Shift_Analyzer SHALL generate visual comparison plots (e.g., Mel spectrogram examples, statistic distributions) saved to data/results/

### Requirement 10: Train Robustness Adaptation Model

**User Story:** As a researcher, I want to train or fine-tune a CNN on telephone/codec data, so that the model maintains detection performance under realistic vishing call channel conditions.

#### Acceptance Criteria

1. WHEN the robustness training script is executed, THE Robustness_Trainer SHALL create models/train_mel_cnn_robust.py that trains a Mel CNN on telephone and telephone_codec conditioned audio
2. WHEN the model is trained, THE Robustness_Trainer SHALL save the trained model to models/mel_cnn_robust_best.keras
3. WHEN the model is trained, THE Robustness_Trainer SHALL log training metrics (loss, accuracy) per epoch to enable convergence verification

### Requirement 11: Prevent Data Leakage in Robustness Training

**User Story:** As a researcher, I want speaker-disjoint or file-disjoint train/validation/test splits, so that model performance estimates are not inflated by data leakage.

#### Acceptance Criteria

1. WHEN data splits are created, THE Robustness_Trainer SHALL partition data such that no speaker appears in more than one of the train, validation, or test sets
2. WHEN data splits are created, THE Robustness_Trainer SHALL keep the final robustness test set separate from training and validation data
3. WHEN data splits are created, THE Robustness_Trainer SHALL document the split strategy and report the number of unique speakers and files in each partition

### Requirement 12: Compare Three Channel Conditions

**User Story:** As a researcher, I want a side-by-side comparison of clean baseline, telephone-only, and telephone+G711_A_Law performance, so that I can quantify degradation and improvement after robustness training.

#### Acceptance Criteria

1. WHEN the three-condition comparison is run, THE Condition_Comparator SHALL report accuracy, precision, recall, F1, ROC-AUC, and confusion matrices for clean, telephone, and telephone_codec conditions using the robustness-trained model
2. WHEN the three-condition comparison is run, THE Condition_Comparator SHALL include the clean-trained model results for the same conditions as a baseline comparison
3. WHEN the comparison is complete, THE Condition_Comparator SHALL write results to a structured CSV in data/results/

### Requirement 13: Tune Operating Threshold on Validation Set

**User Story:** As a researcher, I want to select decision thresholds using a validation set rather than the final test set, so that threshold selection does not overfit to test data.

#### Acceptance Criteria

1. WHEN threshold tuning is performed, THE Threshold_Tuner SHALL select the optimal threshold using validation set predictions only
2. WHEN the optimal threshold is selected, THE Threshold_Tuner SHALL report both the default 0.5 threshold results and the validation-derived threshold results on the test set
3. WHEN threshold tuning is complete, THE Threshold_Tuner SHALL write threshold comparison results to a structured CSV in data/results/

### Requirement 14: Expand Codec and Channel Robustness

**User Story:** As a researcher, I want to add additional channel degradation conditions (noise, bandwidth variation, alternative codec/bit-rate), so that I can conduct an ablation study across multiple channel impairments.

#### Acceptance Criteria

1. WHEN additional conditions are implemented, THE Channel_Expander SHALL apply at least one additional degradation type (additive noise, bandwidth variation, or alternative codec/bit-rate) to audio files
2. WHEN additional conditions are generated, THE Channel_Expander SHALL keep each condition separately labeled in the metadata for ablation analysis
3. WHEN additional conditions are evaluated, THE Channel_Expander SHALL report performance metrics for each condition separately

### Requirement 15: Generate Publication-Quality Analysis

**User Story:** As a researcher, I want publication-quality tables and plots showing performance degradation and improvement, so that I can present findings in a paper.

#### Acceptance Criteria

1. WHEN publication analysis is generated, THE Publication_Analyzer SHALL create comparison tables showing clean baseline, telephone, and telephone_codec metrics for both clean-trained and robustness-trained models
2. WHEN publication analysis is generated, THE Publication_Analyzer SHALL produce ROC curve plots for all conditions
3. WHEN publication analysis is generated, THE Publication_Analyzer SHALL produce confusion matrix visualizations for all conditions
4. WHEN no additional channel conditions have been evaluated, THE Publication_Analyzer SHALL skip ablation table generation entirely
5. WHEN publication analysis is complete, THE Publication_Analyzer SHALL save all plots and tables to data/results/

### Requirement 16: Add TensorFlow/Keras Dependencies

**User Story:** As a developer, I want TensorFlow/Keras added to requirements.txt, so that the CNN model training and evaluation scripts have the necessary dependencies available.

#### Acceptance Criteria

1. WHEN requirements are updated, THE Requirements_Updater SHALL add tensorflow (or tensorflow-cpu) to requirements.txt
2. WHEN requirements are updated, THE Requirements_Updater SHALL specify a pinned or compatible version range for tensorflow
