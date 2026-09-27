# Business Entity Resolution Pipeline

This repository contains the end-to-end pipeline for cross-source business entity resolution across multilingual datasets (US, India, and France). The pipeline handles candidate blocking via dense multilingual sentence embeddings, candidate-filtered phonetic normalization, pairwise lexical and structural feature engineering, and classification using an adversarial-trained XGBoost model optimized for the $F_{0.5}$ metric.

---

## Architecture Overview

1. **Multilingual Dense Blocking (FAISS)**
   * Encodes entity names using `paraphrase-multilingual-MiniLM-L12-v2`.
   * Indexes target sources ($S_2, S_3$) partitioned by country using `faiss.IndexFlatIP`.
   * Retrieves top-$k$ ($k=50$) nearest candidate matches for each $S_1$ entity.
   * Outputs `output/candidate_pairs.tsv`.

2. **Targeted Phonetic Normalization**
   * Transliterates native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali) into the International Phonetic Alphabet (IPA) via Epitran.
   * Normalizes English and Latin texts using neural grapheme-to-phoneme (G2P-en).
   * Restricts transliteration strictly to the retrieved candidate subset to bypass large-scale database catalog bottlenecks.

3. **Pairwise Feature Engineering**
   * Lexical similarity: Levenshtein distance, Jaccard token overlap, RapidFuzz `token_sort_ratio`, and `partial_ratio`.
   * Phonetic similarity: Levenshtein distance and RapidFuzz `token_sort_ratio` on IPA representations.
   * Structural matching: Postal ZIP code matching and regex street-building number conflict detection.
   * Acronym resolution: First-character initialism verification.

4. **Classification & Decision Calibration**
   * Gradient boosted decision trees (`XGBClassifier`) trained with two-pass hard negative mining.
   * Threshold sweeping tailored directly to the macro $F_{0.5}$ metric to maximize precision over false positives.
   * Outputs `output/matching_results.tsv`.

---

## Directory Structure

```text
.
├── code/
│   └── business_entity_resolution/
│       ├── models/
│       │   └── xgb_model.pkl
│       ├── src/
│       │   ├── blocking_diagnostics.py
│       │   ├── extra_features.py
│       │   ├── faiss_blocking.py
│       │   ├── feature_extraction.py
│       │   ├── main.py
│       │   ├── phonetic_normalization.py
│       │   ├── pipeline_cache.py
│       │   └── train_model.py
│       ├── README.md
│       └── requirements.txt
├── dataset/
│   ├── test/
│   │   ├── test_source1.tsv
│   │   ├── test_source2.tsv
│   │   └── test_source3.tsv
│   └── train/
│       ├── train_ground_truth.tsv
│       ├── train_source1.tsv
│       ├── train_source2.tsv
│       └── train_source3.tsv
├── Documentation_template.md
└── output/
    ├── candidate_pairs.tsv
    └── matching_results.tsv
```

---

## Installation & Setup

### Environment Requirements
* Python 3.10+
* CUDA-capable GPU (NVIDIA T4, L4, or A100) or multi-core CPU

### Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r code/business_entity_resolution/requirements.txt
```

---

## Reproducing End-to-End Execution

### 1. Training & Threshold Calibration
To rebuild the training dataset, train the XGBoost classifier with hard negative mining, and sweep the validation set for the optimal macro $F_{0.5}$ decision boundary:

```bash
python code/business_entity_resolution/src/train_model.py
```

The resulting model weights and calibrated decision threshold are serialized directly into:
`code/business_entity_resolution/models/xgb_model.pkl`

### 2. Test Set Inference & Submission Generation
To run the full inference pipeline on the test dataset:

```bash
python code/business_entity_resolution/src/main.py
```

This script executes four sequential stages:
1. Loads test splits from `dataset/test/`.
2. Generates candidate pairs per country block via FAISS dense vector retrieval ($k=50$) and writes `output/candidate_pairs.tsv`.
3. Normalizes candidate names through the Epitran/G2P phonetic cache.
4. Computes 14 pairwise features, predicts match probabilities, applies the optimal threshold, and writes `output/matching_results.tsv`.

---

## Verification of Deliverables

Confirm generated files match submission formatting standards:

```bash
head -n 5 output/candidate_pairs.tsv
head -n 5 output/matching_results.tsv
wc -l output/candidate_pairs.tsv
wc -l output/matching_results.tsv
```

Both files are tab-delimited (`\t`) with headers `source1_entity_id` and `matched_entity_ids`. Targets are serialized as comma-separated lists, with unmatched query records retained as empty strings.
