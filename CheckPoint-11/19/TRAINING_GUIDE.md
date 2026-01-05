# ML Model Training Guide

## ✅ Your Data is Ready!

You have successfully labeled your data and created training datasets:
- **34,172 training examples** ready
- **Train/Val/Test splits** created
- **Labels**: 0 (depression) and 1 (anxiety/ptsd)

## 📁 Training Data Location

```
F:\DATA STORAGE\AGG_PACKET\labeling_outputs\
├── train.json      (27,337 examples)
├── val.json        (3,416 examples)
├── test.json       (3,419 examples)
└── training_dataset.json (full dataset)
```

## 🚀 How to Train Your Model

### Option 1: Use modelB_training.py (Recommended)

This script is designed to work with your labeled data.

#### Step 1: Point to your training data

The script can load from:
- A single JSON file (like `all_labeled_users.json`)
- A directory with JSON files
- Your prepared training files

#### Step 2: Run training

```bash
cd "F:\PROCESSING ALGORITHM\2025-ML-NLP-Research"

python modelB_training.py \
  --dataset-path "F:\DATA STORAGE\AGG_PACKET\labeling_outputs\train.json" \
  --label-field "label" \
  --train \
  --auto \
  --max-users 0
```

**OR** use the consolidated file:

```bash
python modelB_training.py \
  --dataset-path "F:\DATA STORAGE\AGG_PACKET\labeling_outputs\all_labeled_users.json" \
  --label-field "label" \
  --train \
  --auto \
  --max-users 0
```

### Option 2: Use HuggingFace Dataset Format

Your data is also available in HuggingFace format:

```python
from datasets import load_from_disk

# Load your prepared datasets
train_dataset = load_from_disk(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs\train_hf")
val_dataset = load_from_disk(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs\val_hf")
test_dataset = load_from_disk(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs\test_hf")
```

## 📋 Training Parameters

The `modelB_training.py` script uses:
- **Model**: `rafalposwiata/deproberta-large-depression` (pretrained)
- **Learning rate**: 2e-5
- **Batch size**: 4 per device
- **Epochs**: 3
- **Evaluation**: After each epoch
- **Metric**: F1 score

## 🔧 Required Arguments

- `--dataset-path`: Path to your training data
- `--label-field`: Set to `"label"` (the field name in your data)
- `--train`: Enable training (required for training)
- `--auto`: Run without prompts
- `--max-users 0`: Use all data (0 = no limit)

## 📊 What Happens During Training

1. **Load Model**: Loads pretrained `deproberta-large-depression`
2. **Load Data**: Loads your labeled training data
3. **Tokenize**: Converts text to tokens
4. **Train**: Fine-tunes on your labeled data
5. **Evaluate**: Tests on validation set
6. **Save**: Saves fine-tuned model

## 💻 Example Training Command

```bash
python modelB_training.py \
  --dataset-path "F:\DATA STORAGE\AGG_PACKET\labeling_outputs\all_labeled_users.json" \
  --label-field "label" \
  --train \
  --auto \
  --max-users 0 \
  --output-dir "./ModelB_final"
```

## ⚠️ Important Notes

1. **GPU Recommended**: Training is much faster on GPU
   - Check: `torch.cuda.is_available()` should be `True`

2. **Memory**: Large models need significant RAM/VRAM
   - If you get OOM errors, reduce `--max-users` or batch size

3. **Time**: Training 34K examples may take several hours
   - Monitor progress in the logs

4. **Data Format**: Your data has the correct format:
   ```json
   {
     "text": "...",
     "label": 0,
     "user_id": "..."
   }
   ```

## 🎯 Quick Start (Simplest)

```bash
# Navigate to project root
cd "F:\PROCESSING ALGORITHM\2025-ML-NLP-Research"

# Run training
python modelB_training.py \
  --dataset-path "F:\DATA STORAGE\AGG_PACKET\labeling_outputs\all_labeled_users.json" \
  --label-field "label" \
  --train \
  --auto
```

## 📈 Expected Results

After training, you should see:
- Training loss decreasing
- Validation accuracy and F1 score
- Model saved to `--output-dir` (default: `./ModelB_final`)

## 🔍 Troubleshooting

**Problem**: "No labels found"
- **Solution**: Make sure `--label-field "label"` is set

**Problem**: "CUDA out of memory"
- **Solution**: Reduce `--max-users` or use smaller batch size

**Problem**: "File not found"
- **Solution**: Check the path to your training data

## 📝 Next Steps After Training

1. **Evaluate**: Check validation/test metrics
2. **Inference**: Use the saved model for predictions
3. **Fine-tune**: Adjust hyperparameters if needed

