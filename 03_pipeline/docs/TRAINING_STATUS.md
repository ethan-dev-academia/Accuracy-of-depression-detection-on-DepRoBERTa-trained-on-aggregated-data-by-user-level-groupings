# Training Status & Guide

## Current Status

✅ **Training script is running in the background!**

The model is being trained on your final dataset with:
- **27,337 training examples**
- **3,416 validation examples**  
- **3,419 test examples**
- **Label mapping**: 0 = non-depression, 1 = depression

## Training Configuration

- **Model**: `rafalposwiata/deproberta-large-depression` (fine-tuned for binary classification)
- **Learning rate**: 2e-5
- **Batch size**: 4 per device
- **Epochs**: 3
- **Device**: CPU (will take several hours)

## Model Output Location

The trained model will be saved to:
```
F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier_final\
```

## How to Monitor Training

### Option 1: Check Status Script
Run this to see current progress:
```bash
python check_training_status.py
```

### Option 2: Check Logs Directly
Training logs are saved in:
```
saved_models/depression_classifier_final/logs/
```

### Option 3: Check for Checkpoints
After each epoch, a checkpoint is saved:
```
saved_models/depression_classifier_final/checkpoint-1/
saved_models/depression_classifier_final/checkpoint-2/
saved_models/depression_classifier_final/checkpoint-3/
```

## What Gets Saved

After training completes, you'll have:

1. **Final Model** (`pytorch_model.bin` or `model.safetensors`)
2. **Tokenizer** (tokenizer files)
3. **Config** (`config.json`)
4. **Training Info** (`training_info.json`) - includes test results
5. **Checkpoints** (one per epoch)

## Using the Saved Model

Once training is complete, you can load and use the model:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = r"F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier_final"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Use for inference
text = "I've been feeling really down lately..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

## Training Time Estimate

- **CPU**: 6-12 hours (depending on your CPU)
- **GPU**: 1-3 hours (if you have CUDA)

## If Training Stops

If training is interrupted:
1. Checkpoints are saved after each epoch
2. You can resume from the latest checkpoint
3. Or use the latest checkpoint as your model

## Next Steps

1. **Wait for training to complete** (check status with `check_training_status.py`)
2. **Review test results** in `training_info.json`
3. **Load and test the model** using the code above
4. **Use the model for inference** on new data

---

**Note**: The training script runs automatically and saves everything locally. You can train multiple times or resume from checkpoints as needed!

