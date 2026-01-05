"""
Load and use a saved model for inference.

This script shows how to load your saved model and use it for predictions.
"""

from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

print("="*80)
print("LOAD SAVED MODEL")
print("="*80)

# Model location
model_dir = Path(__file__).parent.parent.parent / "saved_models" / "depression_classifier"

# Check if model exists
if not model_dir.exists():
    print(f"\n[ERROR] Model not found at: {model_dir}")
    print("\nPlease train the model first using:")
    print("  python train_and_save_model.py")
    print("\nOr specify the correct model path.")
    exit(1)

print(f"\nLoading model from: {model_dir}")

try:
    # Load model and tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()  # Set to evaluation mode
    
    print("[OK] Model loaded successfully!")
    
    # Show model info
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    print(f"Model type: {type(model).__name__}")
    print(f"Number of labels: {model.config.num_labels}")
    print(f"Label mapping: {model.config.id2label}")
    
    # Example inference
    print("\n" + "="*80)
    print("EXAMPLE INFERENCE")
    print("="*80)
    
    example_text = "I've been feeling really down lately. Nothing seems to help and I can't find motivation to do anything."
    
    print(f"\nExample text: {example_text}")
    
    # Tokenize
    inputs = tokenizer(example_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_label].item()
    
    label_name = model.config.id2label.get(str(predicted_label), f"label_{predicted_label}")
    print(f"\nPrediction:")
    print(f"  Label: {predicted_label} ({label_name})")
    print(f"  Confidence: {confidence*100:.1f}%")
    print(f"\nAll probabilities:")
    for i, prob in enumerate(predictions[0]):
        label = model.config.id2label.get(str(i), f"label_{i}")
        print(f"  {label}: {prob*100:.1f}%")
    
    print("\n" + "="*80)
    print("MODEL READY FOR USE")
    print("="*80)
    print("\nYou can now use this model for predictions!")
    print("\nTo use in your code:")
    print(f"  model = AutoModelForSequenceClassification.from_pretrained(r'{model_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained(r'{model_dir}')")
    
except Exception as e:
    print(f"\n[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()

