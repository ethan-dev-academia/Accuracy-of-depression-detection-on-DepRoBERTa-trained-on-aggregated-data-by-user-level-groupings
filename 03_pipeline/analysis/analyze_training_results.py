"""
Analyze and compare training results.
"""
print("="*80)
print("TRAINING RESULTS ANALYSIS")
print("="*80)

# Previous results (3 epochs)
previous = {
    "epochs": 3,
    "test_accuracy": 0.7397,
    "test_f1": 0.7308,
    "test_loss": 0.5961,
}

# Current results (4 epochs)
current = {
    "epochs": 4,
    "test_accuracy": 0.5929,
    "test_f1": 0.4413,
    "test_loss": 0.6791,
    "train_loss": 0.6857,
    "train_time_hours": 242966.4592 / 3600,  # Convert seconds to hours
}

print("\nPREVIOUS TRAINING (3 epochs):")
print(f"  Test Accuracy: {previous['test_accuracy']:.2%}")
print(f"  Test F1 Score: {previous['test_f1']:.2%}")
print(f"  Test Loss: {previous['test_loss']:.4f}")

print("\nCURRENT TRAINING (4 epochs):")
print(f"  Test Accuracy: {current['test_accuracy']:.2%}")
print(f"  Test F1 Score: {current['test_f1']:.2%}")
print(f"  Test Loss: {current['test_loss']:.4f}")
print(f"  Train Loss: {current['train_loss']:.4f}")
print(f"  Training Time: {current['train_time_hours']:.1f} hours")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

accuracy_change = current['test_accuracy'] - previous['test_accuracy']
f1_change = current['test_f1'] - previous['test_f1']
loss_change = current['test_loss'] - previous['test_loss']

print(f"\nAccuracy: {accuracy_change:+.2%} ({'WORSE' if accuracy_change < 0 else 'BETTER'})")
print(f"F1 Score: {f1_change:+.2%} ({'WORSE' if f1_change < 0 else 'BETTER'})")
print(f"Test Loss: {loss_change:+.4f} ({'WORSE' if loss_change > 0 else 'BETTER'})")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if accuracy_change < -0.10:  # More than 10% drop
    print("\n[WARNING] Significant performance degradation detected!")
    print("\nPossible causes:")
    print("  1. OVERFITTING: Model memorized training data, performs poorly on test")
    print("  2. Training too long: 4 epochs may be too many for this dataset")
    print("  3. Learning rate too high: Model may have overshot optimal weights")
    print("  4. Different training conditions: Different data split or seed?")
    
    print("\nEvidence of overfitting:")
    if current['train_loss'] < current['test_loss']:
        print(f"  - Train loss ({current['train_loss']:.4f}) < Test loss ({current['test_loss']:.4f})")
        print("  - This suggests model is fitting training data too closely")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n1. USE THE PREVIOUS MODEL (3 epochs)")
    print("   - Much better performance (73.97% vs 59.29%)")
    print("   - Better F1 score (73.08% vs 44.13%)")
    print("   - Lower test loss (0.5961 vs 0.6791)")
    
    print("\n2. If you want to improve further:")
    print("   - Use early stopping (stops before overfitting)")
    print("   - Try lower learning rate (1e-5 instead of 2e-5)")
    print("   - Use the 3-epoch model as starting point")
    print("   - Train with early stopping (max 5-6 epochs)")
    
    print("\n3. Training time concern:")
    print(f"   - 4 epochs took {current['train_time_hours']:.1f} hours")
    print("   - This is very long! Consider:")
    print("     * Using GPU if available")
    print("     * Reducing batch size if memory allows")
    print("     * Using early stopping to avoid unnecessary epochs")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nThe 3-epoch model is MUCH BETTER than the 4-epoch model.")
print("Recommendation: Use the previous 3-epoch model and don't train longer.")
print("If you want to improve, use early stopping with the 3-epoch model as starting point.")




