# Banking Vishing Application Dataset Specification

## 1. Objective

The banking-vishing dataset is an application-specific validation dataset for evaluating AI-generated voice detection in simulated fraudulent banking calls.

The main AI-voice detector is developed using ASVspoof datasets. BANKING77 is used to provide realistic banking-domain language and transaction-related scenarios.

The banking-vishing dataset is therefore an application validation layer and is not intended to replace ASVspoof for detector training.

## 2. Banking Scenarios

The current dataset contains 18 banking-vishing scenarios:

1. Suspicious card transaction
2. Suspicious cash withdrawal
3. Compromised card
4. Failed bank transfer
5. Pending bank transfer
6. Transfer not received
7. Identity verification
8. Identity verification problem
9. Source of funds verification
10. Failed account top-up
11. Unknown direct debit
12. Unexpected account charge
13. Card not working
14. Lost or stolen card
15. PIN blocked
16. Refund missing
17. Refund request
18. Beneficiary problem

These scenarios were derived from relevant BANKING77 banking-intent categories.

## 3. Speech Classes

The application dataset will contain two speech classes:

### BONAFIDE

Speech produced by a real human speaker.

### SPOOF

Speech generated using an AI-based TTS, voice-conversion, or voice-cloning system.

## 4. Audio Format

The preferred source format is:

* WAV
* Mono
* 16 kHz
* PCM
* Original audio retained without modification

Telephone conditions are generated dynamically during preprocessing rather than permanently replacing the original recordings.

## 5. Telephone Conditions

Three evaluation conditions are planned:

### Clean

Original 16-kHz speech.

### Telephone

* 300–3400 Hz bandpass
* Resampling to 8 kHz
* Peak normalization

### Telephone + G.711 A-law

* Telephone bandpass
* 8-kHz sampling
* G.711 A-law encoding and decoding

## 6. Short-Time Windows

The following windows are supported:

* 0.5 s
* 0.75 s
* 1.0 s
* 1.5 s
* 2.0 s
* 3.0 s

Windows are generated dynamically from the source audio to avoid unnecessary duplication of audio files.

## 7. Metadata

The final metadata should contain:

* file
* label
* speaker_id
* scenario_id
* category
* language
* sample_rate
* duration
* source

Additional experimental fields may include:

* condition
* window_sec
* start_sec

## 8. Data Leakage Prevention

The same speaker should not appear across development and final evaluation sets where avoidable.

Multiple human speakers and multiple synthetic voices should be used so that the detector cannot simply learn speaker identity.

Scenario identity should also not determine the class.

The same banking scenario should contain both bona-fide and spoof speech where possible.

## 9. Role in the Project

ASVspoof 2019 LA is used for primary model development.

ASVspoof 2021 DF is retained as an external/held-out deepfake evaluation dataset.

The banking-vishing dataset is used to evaluate whether the developed detector can operate in a realistic banking-vishing context.

BANKING77 supplies banking-domain language and intent information but is not treated as an audio dataset.

## 10. Member 1 Responsibility

Member 1 is responsible for:

* Dataset preparation
* Dataset organization
* Audio property analysis
* Dataset metadata
* Short-window metadata
* Telephone simulation
* Codec simulation
* Banking-vishing scenario preparation
* Application-dataset metadata
* Data leakage checks
* Providing standardized input data to the model-development stage