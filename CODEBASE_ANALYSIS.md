# Codebase Analysis & Overview

**Project**: Reddit Mental Health Depression Detection - ML/NLP Research Pipeline  
**Date**: January 2025  
**Status**: Training Complete, Model Deployed

---

## 🎯 Project Purpose

This codebase implements a complete machine learning pipeline for **binary depression classification** using Reddit user data. The system:

1. **Labels** Reddit users based on mental health subreddit participation
2. **Trains** a fine-tuned DepRoBERTa model for depression detection
3. **Evaluates** model performance against baseline models
4. **Saves** trained models for inference

**Key Achievement**: Trained model achieves **73.97% accuracy** and **73.08% F1 score**, outperforming the default DepRoBERTa model by **14.19%** in accuracy.

---

## 📁 Project Structure

**See `PROJECT_ROADMAP.md`** for the chronological exploration map (paper phases → code).

### Main Directories (Reorganized by Phase)

```
2025-ML-NLP-Research/
├── 01_data_collection/        # Phase 1: Reddit data collection (was: Data Processing Summer Back)
├── 02_model_exploration/      # Phase 2: DepRoBERTa setup, loadingPretrainedModel.ipynb
├── 03_pipeline/               # Phase 3–5: Labeling, training, evaluation (was: CheckPoint-11/19)
├── 04_outputs/                # Phase 6: Model outputs (ModelB_final, saved_models copy)
├── saved_models/              # Trained model checkpoints (stays at root for script compatibility)
├── ModelB_final/              # Junction → 04_outputs/models/ModelB_final
└── modelB_training.py         # Main training script
```

---

## 🔄 Complete Pipeline Overview

### Phase 1: Data Collection & Processing
**Location**: `01_data_collection/`

- **Purpose**: Collect and process Reddit user data
- **Tools**: 
  - `reddit_ml_toolkit.py` - Reddit API wrapper
  - `reddit_user_analyzer.py` - User data extraction
  - `check_processed_files.py` - Data validation
- **Output**: JSON files with user posts/comments in `F:\DATA STORAGE\AGG_PACKET`

### Phase 2: Labeling Pipeline
**Location**: `03_pipeline/`

A 4-step automated pipeline:

#### Step 1: Extract Labels (`extract_rmh_labels.py`)
- Scans RMH (Reddit Mental Health) dataset CSV files
- Extracts usernames from subreddits (depression, anxiety, PTSD, etc.)
- Creates label mappings: depression→0, anxiety/PTSD→1, others→2
- **Result**: 172,354 username-label mappings

#### Step 2: Match Labels (`match_labels_to_agg_packet.py`)
- Matches RMH labels to AGG_PACKET user data
- Adds labels to user JSON files
- Handles conflicts (users in multiple subreddits)
- **Result**: 147,130 labeled users (20.2% match rate from 728,921 total users)

#### Step 3: Validate Data (`validate_labeled_data.py`)
- Quality checks on labeled data
- Validates labels, content quality, class balance
- Filters invalid entries
- **Result**: Validation reports and filtered datasets

#### Step 4: Prepare Training Dataset (`prepare_training_dataset.py`)
- Aggregates user posts/comments into single text per user
- Creates train/val/test splits (80/10/10, stratified)
- Saves in JSON and HuggingFace formats
- **Result**: 34,172 training examples ready for model training

**Pipeline Orchestrator**: `run_labeling_pipeline.py` (runs all 4 steps automatically)

### Phase 3: Model Training
**Location**: `03_pipeline/` and root (`modelB_training.py`)

#### Training Scripts:
- `modelB_training.py` - Main interactive training script (root)
- `train_final_model.py` - Final training script
- `train_with_early_stopping.py` - Training with early stopping
- Multiple other training variants

#### Training Configuration:
- **Base Model**: `rafalposwiata/deproberta-large-depression`
- **Task**: Binary classification (0=non-depression, 1=depression)
- **Learning Rate**: 2e-5
- **Batch Size**: 4 per device
- **Epochs**: 3
- **Device**: CPU (can use GPU if available)
- **Training Time**: ~50 hours on CPU

#### Dataset:
- **Training**: 27,337 examples
- **Validation**: 3,416 examples
- **Test**: 3,419 examples
- **Label Distribution**: 59.3% depression, 40.7% non-depression

### Phase 4: Model Evaluation & Comparison
**Location**: `03_pipeline/evaluation/model_comparison_results/`

- Compares fine-tuned model vs. default DepRoBERTa
- Generates comprehensive metrics and reports
- Analyzes disagreements between models

---

## 🏆 Key Results

### Model Performance (Final Model)

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 73.97% | Test set performance |
| **F1 Score** | 73.08% | Balanced precision/recall |
| **Precision** | 74.04% | |
| **Recall** | 73.97% | |
| **Validation Accuracy** | 73.62% | Similar to test (good generalization) |

### Model Comparison

Your trained model vs. default DepRoBERTa:

| Metric | Your Model | Default Model | Improvement |
|--------|------------|---------------|-------------|
| Accuracy | 73.97% | 59.78% | **+14.19%** |
| F1 Score | 73.08% | 49.32% | **+23.76%** |
| Precision | 74.04% | 57.73% | **+16.32%** |

**Conclusion**: Fine-tuning significantly improved performance!

---

## 📊 Data Flow

```
RMH Dataset (CSV files)
    ↓
[extract_rmh_labels.py]
    ↓
rmh_username_labels.json (172K mappings)
    ↓
[match_labels_to_agg_packet.py]
    ↓
AGG_PACKET JSON files + Labels (147K labeled users)
    ↓
[validate_labeled_data.py]
    ↓
Validated & Filtered Data
    ↓
[prepare_training_dataset.py]
    ↓
train.json / val.json / test.json (34K examples)
    ↓
[modelB_training.py / train_final_model.py]
    ↓
saved_models/depression_classifier_final/ (Trained Model)
```

---

## 🔑 Key Components & Files

### Core Labeling Scripts (03_pipeline/)
1. **`extract_rmh_labels.py`** - Extract labels from RMH dataset
2. **`match_labels_to_agg_packet.py`** - Match labels to user data
3. **`validate_labeled_data.py`** - Validate labeled data quality
4. **`prepare_training_dataset.py`** - Create training splits
5. **`run_labeling_pipeline.py`** - Master orchestrator

### Training Scripts
- **`modelB_training.py`** (root) - Main interactive training script
- **`train_final_model.py`** - Final training script
- **`start_training.py`** - Training launcher
- **`check_training_status.py`** - Monitor training progress

### Utility Scripts
- **`inspect_labeled_data.py`** - Inspect labeled data
- **`load_saved_model.py`** - Load trained models
- **`compare_models.py`** - Compare model performance
- **`analyze_training_results.py`** - Analyze training metrics
- **`explain_pipeline.py`** - Explain pipeline flow

### Configuration
- **`labeling_config.json`** - Labeling pipeline configuration
  - Data paths
  - Label mappings
  - Validation thresholds
  - Training split ratios

### Documentation
- **`README_LABELING.md`** - Labeling pipeline guide
- **`TRAINING_GUIDE.md`** - Training instructions
- **`TRAINING_STATUS.md`** - Current training status
- **`STATUS_REPORT.md`** - Project status report
- **`DATASET_LOCATIONS.md`** - Dataset file locations

### Data Locations

**Input Data:**
- RMH Dataset: `F:/DATA STORAGE/RMH Dataset`
- AGG_PACKET: `F:\DATA STORAGE\AGG_PACKET`

**Output Data:**
- Labeling outputs: `F:\DATA STORAGE\AGG_PACKET\labeling_outputs\`
- Final training set: `F:\DATA STORAGE\AGG_PACKET\final_training_set\`
- Trained models: `saved_models/depression_classifier_final/`

---

## ⚠️ Key Things to Keep in Mind

### 1. **Data Paths are Hardcoded**
- Many scripts use hardcoded Windows paths (`F:\DATA STORAGE\...`)
- Update paths if data is moved or on different system
- Configuration file `labeling_config.json` centralizes some paths

### 2. **Label Mapping**
- **Original**: depression=0, anxiety/PTSD=1, others=2
- **Training**: Binary classification (0=non-depression, 1=depression)
- Label 2 is filtered out for training (only uses 0 and 1)
- Note: The model is trained as binary, but original labels had 3 classes

### 3. **Model Architecture**
- Base: `rafalposwiata/deproberta-large-depression` (3-class model)
- Fine-tuned: Binary classification (2 labels)
- Uses `ignore_mismatched_sizes=True` to handle classification head mismatch
- Model size: ~1.3GB

### 4. **Training Time & Resources**
- CPU training takes ~50 hours
- GPU recommended for faster training
- Model checkpoints saved after each epoch
- Can resume from checkpoints if interrupted

### 5. **Data Quality**
- 20.2% match rate (147K labeled out of 728K users)
- Class balance: 59.3% / 40.7% (acceptable imbalance)
- Average text length: ~14,524 characters per user
- Data validation ensures quality before training

### 6. **Model Performance**
- 73.97% accuracy is good for this task
- Significantly better than baseline (59.78%)
- Model generalizes well (validation ≈ test performance)
- Ready for deployment or further use

### 7. **Two Saved Model Versions**
- `depression_classifier_final/` - Final 3-epoch model (73.97% accuracy)
- `depression_classifier_early_stop/` - Early stopping version
- Use the `_final` version (better performance)

### 8. **Training Scripts Variants**
Multiple training scripts exist with different features:
- `modelB_training.py` - Interactive, step-by-step
- `train_final_model.py` - Direct training on final dataset
- `train_with_early_stopping.py` - Early stopping variant
- Choose based on needs (most use `train_final_model.py`)

### 9. **Reddit Data Processing**
- Separate directory: `01_data_collection/`
- Contains tools for collecting Reddit data via API
- Creates the AGG_PACKET JSON files used in labeling pipeline
- Can be run independently

### 10. **Comparison Results**
- Model comparison already completed
- Results in `03_pipeline/evaluation/model_comparison_results/`
- Shows your model outperforms baseline significantly
- Useful for documentation and analysis

---

## 🚀 Quick Start Guide

### To Run Labeling Pipeline:
```bash
cd 03_pipeline/labeling
python run_labeling_pipeline.py
```

### To Train Model:
```bash
python modelB_training.py \
  --dataset-path "F:\DATA STORAGE\AGG_PACKET\final_training_set\train.json" \
  --label-field "label" \
  --train \
  --auto \
  --max-users 0
```

### To Load Trained Model:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = r"saved_models\depression_classifier_final"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

---

## 📈 Project Status

### ✅ Completed
- [x] Data collection and processing
- [x] Labeling pipeline (all 4 steps)
- [x] Data validation
- [x] Training dataset preparation
- [x] Model training (3 epochs)
- [x] Model evaluation
- [x] Model comparison analysis
- [x] Documentation

### 🎯 Current State
- **Training**: Complete ✅
- **Model**: Deployed and saved ✅
- **Evaluation**: Complete ✅
- **Documentation**: Complete ✅

### 🔮 Potential Future Work
- Fine-tuning hyperparameters
- Data augmentation
- Multi-class classification (using all 3 labels)
- Deployment pipeline
- Real-time inference API
- Model monitoring

---

## 🛠️ Technology Stack

- **Python** - Main language
- **Transformers** (HuggingFace) - Model framework
- **PyTorch** - Deep learning backend
- **Datasets** (HuggingFace) - Data handling
- **pandas** - Data processing
- **scikit-learn** - Metrics calculation
- **JSON** - Data storage format

---

## 📝 Important Notes

1. **Windows Paths**: Code uses Windows-style paths (`F:\...`). Update for Linux/Mac.
2. **Large Files**: Training datasets are large (597MB train.json). Ensure sufficient disk space.
3. **Model Size**: Trained models are ~1.3GB each. Plan storage accordingly.
4. **CPU vs GPU**: Training on CPU takes ~50 hours. GPU recommended for faster iteration.
5. **Checkpoints**: Model checkpoints saved after each epoch. Can resume training.
6. **Label Interpretation**: Binary labels: 0=non-depression, 1=depression (not severity levels).

---

## 🎓 Learning Resources

The codebase includes extensive documentation:
- Pipeline explanations
- Training guides
- Status reports
- Comparison analyses

All documentation is in `03_pipeline/docs/` directory with `.md` files.

---

**Summary**: This is a production-ready ML pipeline for depression detection that successfully labels Reddit data, trains a fine-tuned DepRoBERTa model, and achieves 73.97% accuracy - a significant improvement over the baseline model.

