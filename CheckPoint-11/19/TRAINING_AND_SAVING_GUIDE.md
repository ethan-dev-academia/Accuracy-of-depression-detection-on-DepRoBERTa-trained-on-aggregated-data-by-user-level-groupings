# Training and Saving Model Guide

## ✅ Your Data is Ready!

Your final training corpus is ready in:
```
F:\DATA STORAGE\AGG_PACKET\final_training_set\
```

**Label Mapping (Corrected):**
- **Label 1** = Depression
- **Label 0** = Non-depression

---

## 🚀 Training Options

### Option 1: Use Helper Script (Recommended)

```bash
cd "F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\CheckPoint-11\19"
python train_and_save_model.py
```

This will:
- Train the model on your final training set
- Save the model to: `saved_models/depression_classifier/`
- Allow you to reuse the model multiple times

### Option 2: Direct Training Command

```bash
cd "F:\PROCESSING ALGORITHM\2025-ML-NLP-Research"

python modelB_training.py \
  --dataset-path "F:\DATA STORAGE\AGG_PACKET\labeling_outputs\all_labeled_users.json" \
  --label-field "label" \
  --train \
  --auto \
  --max-users 0 \
  --output-dir "./saved_models/depression_classifier"
```

---

## 💾 Model Saving

The training script automatically saves:
- **Best model checkpoint** (based on validation F1 score)
- **Final model** (after all epochs)
- **Tokenizer** (for text preprocessing)
- **Config** (model configuration)

**Save Location:**
```
F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier\
```

**What Gets Saved:**
- `config.json` - Model configuration
- `pytorch_model.bin` or `model.safetensors` - Model weights
- `tokenizer_config.json` - Tokenizer settings
- `vocab.json` - Vocabulary
- `training_args.bin` - Training arguments
- Checkpoint folders (if multiple epochs)

---

## 🔄 Using Saved Model Multiple Times

### Load and Use Saved Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

# Load saved model
model_dir = Path(r"F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier")

model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

# Use for predictions
text = "Your text here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

### Or Use the Helper Script

```bash
python load_saved_model.py
```

---

## 📋 Training Process

1. **Load Pretrained Model**: `rafalposwiata/deproberta-large-depression`
2. **Load Your Data**: From `final_training_set/`
3. **Tokenize**: Convert text to model inputs
4. **Train**: Fine-tune on your labeled data (3 epochs)
5. **Evaluate**: Check performance on validation set
6. **Save**: Best model saved automatically

---

## ⏱️ Training Time

- **CPU**: Several hours (6-12+ hours for 34K examples)
- **GPU**: Much faster (1-3 hours)
- **Checkpoints**: Saved after each epoch (can resume if interrupted)

---

## 📁 Model Files Structure

After training, you'll have:
```
saved_models/depression_classifier/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── training_args.bin
└── checkpoint-*/ (if multiple checkpoints saved)
```

---

## 🔄 Resume Training

If training is interrupted, you can resume from a checkpoint:
```bash
python modelB_training.py \
  --dataset-path "..." \
  --label-field "label" \
  --train \
  --output-dir "./saved_models/depression_classifier" \
  --resume_from_checkpoint "./saved_models/depression_classifier/checkpoint-2"
```

---

## ✅ Ready to Train?

**Quick Start:**
```bash
cd "F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\CheckPoint-11\19"
python train_and_save_model.py
```

The model will be saved locally and you can use it as many times as you want!

