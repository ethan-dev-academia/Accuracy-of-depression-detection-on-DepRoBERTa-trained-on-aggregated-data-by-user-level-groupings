# Model Comparison Report

## Your Trained Model vs Default DepRoBERTa

**Test Set**: 3,419 examples

## Summary Metrics

| Metric | Your Model | Default Model (Binary) | Improvement |
|--------|------------|------------------------|-------------|
| Accuracy | 0.7397 (73.97%) | 0.5978 (59.78%) | +0.1419 (+14.19%) |
| F1 Score | 0.7308 (73.08%) | 0.4932 (49.32%) | +0.2376 (+23.76%) |
| Precision | 0.7404 (74.04%) | 0.5773 (57.73%) | +0.1632 |
| Recall | 0.7397 (73.97%) | 0.5978 (59.78%) | +0.1419 |

## Default Model 3-Class Distribution

- **not depression**: 201 (5.9%)
- **moderate**: 785 (23.0%)
- **severe**: 2,433 (71.2%)

## Disagreement Analysis

- **Total disagreements**: 1003 (29.3%)
- **Agreement rate**: 70.7%

## Conclusion

✅ **Your trained model performs BETTER than the default DepRoBERTa model** on this test set.

Your fine-tuning improved accuracy by 14.19% and F1 score by 23.76%.

## Files Generated

- `per_example_predictions.csv` - Full test set with all predictions
- `comparison_results_summary.json` - Overall metrics
- `confusion_matrices.json` - Confusion matrices for both models
- `disagreement_analysis.csv` - Cases where models disagree
- `class_distribution_analysis.json` - 3-class distribution from default model
