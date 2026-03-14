# Depression Detection on DepRoBERTa (User-Level Aggregates)

*Accuracy of depression detection on DepRoBERTa tuned on aggregated data by user-level groupings*  
E. Yip, R. Lafnitzegger | Jan 2026

---

## Structure (Chronological)

| Phase | Folder | Purpose |
|-------|--------|---------|
| 1 | `01_data_collection/` | Reddit API, user data → AGG_PACKET |
| 2 | `02_model_exploration/` | DepRoBERTa setup, pretrained model loading |
| 3–5 | `03_pipeline/` | Labeling, training, evaluation |
| 6 | `04_outputs/` | Model checkpoints, results |

**See `PROJECT_ROADMAP.md`** for the full paper→code map and run order.

---

## Quick start

1. **Data collection** (Machine A): `01_data_collection/`
2. **Labeling**: `03_pipeline/labeling/run_labeling_pipeline.py`
3. **Training**: `python modelB_training.py --train --auto`
4. **Evaluation**: `03_pipeline/evaluation/compare_models.py`

Data paths: `F:\DATA STORAGE\AGG_PACKET`, `F:\DATA STORAGE\RMH Dataset`  
Config: `03_pipeline/labeling/labeling_config.json`
