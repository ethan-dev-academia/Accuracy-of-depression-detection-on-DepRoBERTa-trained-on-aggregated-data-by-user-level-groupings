# Depression Detection Research — Chronological Project Roadmap

**Paper:** *Accuracy of depression detection on DepRoBERTa tuned on aggregated data by user-level groupings*  
**Authors:** E. Yip, R. Lafnitzegger | January 9, 2026

This document maps the research flow (paper sections) to the project layout so you can navigate and extend the work alongside your original exploration.

---

## Phase 1: Data Collection (Summer 2025)

**Paper:** *Materials and Methods — Dataset Collection and User-level Aggregation*

- Source: Reddit Mental Health Dataset (RMHD)
- Steps: Extract depressed/anxiety/suicidal users → filter English → query Reddit API for historical posts → build user-level sequences
- Output: AGG_PACKET JSON files (`F:\DATA STORAGE\AGG_PACKET`)

| Location | Purpose |
|----------|---------|
| `01_data_collection/` | Reddit API toolkit, user analyzer, validation scripts |

**Key scripts:** `reddit_ml_toolkit.py`, `reddit_user_analyzer.py`, `check_processed_files.py`

---

## Phase 2: Model Environment & Exploration (Oct 2025)

**Paper:** *Materials and Methods — Model Architecture and Tuning Procedure*

- Set up DepRoBERTa environment
- Load and explore pretrained `rafalposwiata/deproberta-large-depression`

| Location | Purpose |
|----------|---------|
| `02_model_exploration/` | DepRoBERTa setup, pretrained model loading |
| `02_model_exploration/loadingPretrainedModel.ipynb` | Notebook for model loading and exploration |
| `02_model_exploration/iterations/` | Archived iterations (e.g., "wrong version" run) |
| `DepRoBERTa-env/` | Virtual environment (stays at root) |

---

## Phase 3: Labeling & Dataset Preparation (Nov 2025)

**Paper:** *Materials and Methods — Text Preprocessing and Packet Aggregation*

- RMHD labels → match to AGG_PACKET users → validate → create train/val/test splits (80/10/10)
- Packet aggregation: 3–5 posts per user, chronological, 512-token truncation

| Location | Purpose |
|----------|---------|
| `03_pipeline/labeling/` | `extract_rmh_labels`, `match_labels_to_agg_packet`, `validate_labeled_data`, `prepare_training_dataset` |
| `03_pipeline/dataset/` | `build_final_training_set` |
| `03_pipeline/data_exploration/` | RMHD inspection, label analysis |
| `03_pipeline/labeling/labeling_config.json` | Paths and label mappings |

**Outputs:** `train.json`, `val.json`, `test.json` (34,172 examples)

---

## Phase 4: Training (Nov 2025)

**Paper:** *Materials and Methods — Model Architecture and Tuning Procedure*

- Base: DepRoBERTa
- Fine-tuning: 3 epochs, batch 4, lr 2e-5, CrossEntropyLoss, early stopping

| Location | Purpose |
|----------|---------|
| `modelB_training.py` (root) | Main interactive training script |
| `03_pipeline/training/` | `train_final_model`, `train_with_early_stopping`, etc. |

**Outputs:** `saved_models/`, `ModelB_final/`

---

## Phase 5: Evaluation & Comparison (Nov 2025)

**Paper:** *Materials and Methods — Model Evaluation and Statistical Analysis*  
**Paper:** *Results — Model Performance, Comparison, McNemar's Test*

- Evaluate on 3,419 aggregated test packets
- Compare fine-tuned vs. default DepRoBERTa
- Evaluate on 10,000 individual messages
- McNemar’s chi-squared test (p &lt; 0.001)

| Location | Purpose |
|----------|---------|
| `03_pipeline/evaluation/` | `compare_models`, `compare_models_single_messages`, `statistical_testing` |
| `03_pipeline/evaluation/model_comparison_results/` | Aggregate-level metrics, confusion matrices |
| `03_pipeline/evaluation/single_message_comparison_results/` | Individual-message metrics |
| `03_pipeline/analysis/` | Training results, pipeline explanation |

---

## Phase 6: Outputs & Results

**Paper:** *Results — Tables 1–9, Figures 1–4*

| Location | Purpose |
|----------|---------|
| `04_outputs/models/` | ModelB_final, saved_models (alternate copy) |
| `04_outputs/results/` | Model comparison reports (in `03_pipeline/evaluation/`) |
| `saved_models/` (root) | Primary model dir for scripts — stays at root for compatibility |

---

## Quick Reference: Paper → Code

| Paper section | Code location |
|---------------|---------------|
| Dataset Collection | `01_data_collection/` |
| Packet Aggregation | `03_pipeline/labeling/prepare_training_dataset.py` |
| Model Architecture | `modelB_training.py`, `DepRoBERTa-env` |
| Training Procedure | `modelB_training.py`, `03_pipeline/training/` |
| Test Evaluation | `03_pipeline/evaluation/compare_models.py` |
| Individual-Message Test | `03_pipeline/evaluation/compare_models_single_messages.py` |
| McNemar’s Test | `03_pipeline/evaluation/statistical_testing.py` |

---

## Run order (pipeline)

1. `01_data_collection/` → produce AGG_PACKET
2. `03_pipeline/labeling/run_labeling_pipeline.py` → labeled train/val/test
3. `modelB_training.py --train --auto` → train model
4. `03_pipeline/evaluation/compare_models.py` → comparison on aggregated test
5. `03_pipeline/evaluation/compare_models_single_messages.py` → single-message comparison

---

*Use this roadmap with your original files; paths in scripts remain as documented in `labeling_config.json` and training args.*
