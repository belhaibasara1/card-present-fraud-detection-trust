# Trustworthy Machine Learning-Based Fraud Detection for Card-Present Payment Systems

MSc dissertation project evaluating the trustworthiness of machine learning-based fraud
detectors for card-present payment systems, combining a STRIDE-derived attack taxonomy,
three classifiers (supervised and unsupervised), and cross-model SHAP explainability.

## Academic Context

- **Author:** Sara Belhaiba
- **Supervisor:** Prof Christian Wagner
- **Programme:** MSc Cyber-Physical Systems (Cyber Security pathway)
- **Institution:** University of Nottingham, 2025-2026

## Project Structure

```
src/pipeline.py                   Full pipeline: preprocessing, class imbalance
                                     comparison, model training (RF, XGBoost,
                                     Isolation Forest), threshold sensitivity,
                                     SHAP explainability, STRIDE-based attack
                                     simulation, and the traffic light framework
README.md                           This file
data/                                Dataset folder (not included, see below)
```

This is the single pipeline referenced throughout the dissertation (Chapter 4 and
Appendix C); all results reported in Chapter 5 and Appendix A were produced by running
this file exactly as submitted.

## Dataset

This project uses the [Credit Card Fraud Detection dataset (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
from Kaggle: approximately 285,000 anonymised, PCA-transformed card-present
transactions with a documented fraud rate of roughly 0.17%. The dataset is not
included in this submission due to its size. To run the code:

1. Download the dataset from the Kaggle link above.
2. Extract the archive.
3. Place `creditcard.csv` in a `data/` folder at the root of this project.

This dataset is used directly within its native domain, card-present payment fraud,
rather than as a structural proxy for a different context. It was selected because it
provides ground-truth fraud labels and shares the structural properties central to this
evaluation: extreme class imbalance, a purely transactional data structure, and
statistically rare anomaly patterns. The dataset's limitations, most notably the absence
of a card or account identifier, are discussed in full in Section 1.3 and Section 3.5 of
the dissertation.

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the pipeline:

```
python src/pipeline.py
```

The script prints progress to the console section by section (data preparation, class
imbalance comparison, model training, cross-validation, threshold sensitivity, SHAP
analysis, attack simulation, and the traffic light framework) and writes results to
`results/`, trained models to `models/`, and figures to `figures/` (all created
automatically on first run).

## Current Status

All eight objectives set out in Chapter 1 have been addressed:

- Literature review: complete (Chapter 2)
- STRIDE threat model and five-type attack taxonomy: complete (Chapter 3)
- Data preparation and feature engineering: complete (Section 4.2, 4.4)
- Class imbalance handling comparison: complete (Section 4.3)
- Random Forest, XGBoost, and Isolation Forest training: complete (Section 4.5)
- SHAP explainability analysis (training-time and post-hoc): complete (Section 4.7)
- Attack simulation and per-attack-type robustness evaluation: complete (Section 4.6)
- Traffic light trustworthiness framework: complete, with the automated
  tier-plus-agreement refinement proposed in Section 6.3 identified as future work

## Known Limitations

- **Identity linkage:** the dataset contains no card or account identifier, which
  means three of the five simulated attack types (card cloning, velocity attacks,
  account sharing) cannot be represented at the behavioural level this project's
  taxonomy defines them by. This is a structural constraint of the dataset, stated in
  advance in Section 3.5 and confirmed empirically in Section 5.7, not a weakness in
  the classifiers.
- **Data leakage in preprocessing:** `StandardScaler` is fitted on the full dataset
  before the train/test split, rather than on the training partition alone. This is a
  genuine methodological error, disclosed and corrected in Appendix C.1, where the
  corrected code is also provided. Its likely practical impact on the headline results
  in Chapter 5 is small (it affects only 2 of 31 features), but has not been formally
  re-quantified; re-running the comparison under the corrected ordering is listed as
  the first item of future work (Section 7.4).
- **Batch-relative normalisation:** Isolation Forest's anomaly scores are normalised
  using the min/max of the test set as a whole, which is adequate for the offline
  comparison in Chapter 5 but would need to be fixed to a reference range before use
  in single-transaction, real-time inference (Appendix C.2).
- **Inference latency not measured:** Section 6.5 discusses the computational cost of
  running three classifiers plus SHAP per transaction, but wall-clock latency was not
  benchmarked. A timing script is provided in Appendix C.4 for this purpose.

