# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** Meehir Prabhakar
**Submission Date:** September 27, 2026

---

## 1. Executive Summary
A high-performance, 4-stage pipeline utilizing multilingual Sentence Transformers for robust dense blocking, paired with lazy phonetic normalization. The classification relies on an adversarial-mined XGBoost model using 14 lexical, phonetic, and structural features, explicitly optimized for the macro $F_{0.5}$ metric.

---

## 2. Methodology

### 2.1 Problem Analysis
We identified significant transliteration noise and unseen linguistic distributions (e.g., French data in the test set) which break traditional character-level TF-IDF. Furthermore, raw edit distances on addresses fail to distinguish sister companies sharing identical street names but different building numbers.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier
**Core Innovation:** Replacing heuristic blocking with domain-agnostic `paraphrase-multilingual-MiniLM-L12-v2` dense retrieval, followed by targeted post-retrieval G2P/Epitran phonetic normalization to bypass massive dataset bottlenecks.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Multilingual dense embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) via FAISS Inner Product search, partitioned by country.
- **Candidate pairs generated:** Top k=50 neighbors per S1 query.
- **How you ensured true matches were not lost:** By mapping raw names into a semantic multilingual space, the embedding captures cross-lingual anagrams and un-transliterated similarities that strict phonetic keys drop.

---

## 4. Matching Model

**Features used:**
- Name features: Levenshtein, Jaccard, Token Sort, Partial Ratio, Acronym matching, IPA Phonetic Token Sort, and IPA Partial Ratio.
- Address features: Address Token Sort, Address Levenshtein, regex-extracted street number conflict detection, and ZIP code exact matching.
- Other: Absolute name length differences.

**Model type:** XGBoost Classifier (Two-pass adversarial training).
**Threshold selection method:** Automated threshold sweeping explicitly maximizing Macro $F_{0.5}$ on the validation set.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9190
- **Common false positives (wrong merges):** Franchise branches with identical names and unlisted/missing building numbers resolving to the same coordinate block.
- **Common false negatives (missed matches):** Complete rebrands or heavy abbreviations not captured by initialism logic.

---

## 6. Conclusion
By decoupling semantic dense blocking from heavy phonetic feature engineering, we achieved high recall on unseen multilingual test data while retaining strict execution efficiency. The adversarial XGBoost configuration and strict 0.80 decision boundary effectively minimize false positives to dominate the $F_{0.5}$ penalty curve.

---

## Appendix

### A. Code Artefacts
The complete, runnable code ships in the submission zip under `code/business_entity_resolution/` with all source in `src/`. Reproducibility is split into two entry points: `train_model.py` rebuilds the dataset, executes adversarial training, and sweeps for the optimal threshold. `main.py` orchestrates the test inference, executing FAISS blocking, targeted phonetic normalization, and XGBoost inference to generate `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

### B. Additional Results
Threshold tuning loop demonstrated a smooth $F_{0.5}$ curve peaking at 0.80, proving standard 0.50 thresholds are suboptimal for this specific competition metric.
