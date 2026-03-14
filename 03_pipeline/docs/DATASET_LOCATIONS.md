# Dataset Locations

## Test, Validation, and Training Sets

All your evaluation datasets are located in:

```
F:\DATA STORAGE\AGG_PACKET\final_training_set\
```

## File Details

### 1. Test Set (for final evaluation)
- **Location**: `F:\DATA STORAGE\AGG_PACKET\final_training_set\test.json`
- **Size**: 74.35 MB
- **Purpose**: Final evaluation on unseen data
- **Used for**: Testing model performance after training

### 2. Validation Set (for monitoring during training)
- **Location**: `F:\DATA STORAGE\AGG_PACKET\final_training_set\val.json`
- **Size**: 76.82 MB
- **Purpose**: Monitor training progress and prevent overfitting
- **Used for**: Validation during training epochs

### 3. Training Set (for model training)
- **Location**: `F:\DATA STORAGE\AGG_PACKET\final_training_set\train.json`
- **Size**: 597.68 MB
- **Purpose**: Train the model
- **Used for**: Learning patterns and updating weights

## Quick Access

### In Python:
```python
from pathlib import Path

# Test set
test_file = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set\test.json")

# Validation set
val_file = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set\val.json")

# Training set
train_file = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set\train.json")
```

### Loading the test set:
```python
import json
from pathlib import Path

test_file = Path(r"F:\DATA STORAGE\AGG_PACKET\final_training_set\test.json")
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Test set contains {len(test_data)} examples")
```

## Data Format

Each file contains a JSON array of examples, where each example has:
- `text`: The aggregated text content (posts + comments)
- `label`: The binary label (0 = non-depression, 1 = depression)

Example:
```json
[
  {
    "text": "I've been feeling really down lately...",
    "label": 1
  },
  {
    "text": "Had a great day today!",
    "label": 0
  }
]
```

## Dataset Statistics

Based on your training results:
- **Training examples**: 27,337
- **Validation examples**: 3,416
- **Test examples**: 3,419
- **Total**: ~34,172 examples

## Related Files

### Model Location:
```
F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\saved_models\depression_classifier_final\
```

### Training Scripts:
```
F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\CheckPoint-11\19\train_final_model.py
```

## Notes

- All files are in JSON format
- Files are UTF-8 encoded
- Labels are binary: 0 = non-depression, 1 = depression
- Test set should only be used for final evaluation (not during training)




