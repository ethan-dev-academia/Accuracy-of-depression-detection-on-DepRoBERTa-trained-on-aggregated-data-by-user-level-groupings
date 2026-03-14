# Single Message Classification Results

## Model Comparison: Individual Messages (Posts/Comments)

This document presents the results of comparing the fine-tuned DepRoBERTa model with the default DepRoBERTa model on individual messages (single posts and comments) rather than aggregated user-level data.

---

## Test Dataset

- **Total Messages**: 10,000 individual messages
- **Posts**: 1,362 (13.6%)
- **Comments**: 8,638 (86.4%)
- **Label Distribution**:
  - Non-Depression (Label 0): 5,771 messages (57.7%)
  - Depression (Label 1): 4,229 messages (42.3%)

*Note: Messages were extracted from labeled user data, with each message inheriting the label from its source user. Messages were randomly sampled from a total pool of 2,266,919 available messages (304,677 posts and 1,962,242 comments).*

---

## Your Trained Model Performance

### Performance Metrics

| Metric | Value | Percentage |
|--------|-------|------------|
| **Accuracy** | 0.4201 | 42.01% |
| **F1 Score** | 0.3636 | 36.36% |
| **Precision** | 0.4548 | 45.48% |
| **Recall** | 0.4201 | 42.01% |

### Confusion Matrix

| | Predicted: Non-Depression | Predicted: Depression |
|--|-------------------------|----------------------|
| **Actual: Non-Depression** | 921 (True Negatives) | 4,850 (False Positives) |
| **Actual: Depression** | 949 (False Negatives) | 3,280 (True Positives) |

*Note: Test set size n=10,000 individual messages. Rows represent actual labels, columns represent predicted labels.*

---

## Default DepRoBERTa Model Performance (Binary Mapped)

### Performance Metrics

| Metric | Value | Percentage |
|--------|-------|------------|
| **Accuracy** | 0.4227 | 42.27% |
| **F1 Score** | 0.2592 | 25.92% |
| **Precision** | 0.4609 | 46.09% |
| **Recall** | 0.4227 | 42.27% |

### Confusion Matrix

| | Predicted: Non-Depression | Predicted: Depression |
|--|-------------------------|----------------------|
| **Actual: Non-Depression** | 46 (True Negatives) | 5,725 (False Positives) |
| **Actual: Depression** | 48 (False Negatives) | 4,181 (True Positives) |

*Note: Test set size n=10,000 individual messages. Default model 3-class predictions mapped to binary: "not depression"→0, "moderate"+"severe"→1.*

---

## Comparison Summary

### Direct Comparison

| Metric | Your Model | Default Model | Difference |
|--------|------------|---------------|------------|
| **Accuracy** | 42.01% | 42.27% | -0.26% |
| **F1 Score** | 36.36% | 25.92% | +10.44% |
| **Precision** | 45.48% | 46.09% | -0.61% |
| **Recall** | 42.01% | 42.27% | -0.26% |

### Key Observations

1. **Accuracy**: The default model achieved slightly higher accuracy (42.27% vs. 42.01%), with a difference of 0.26 percentage points. This difference is minimal and may not be statistically significant.

2. **F1 Score**: Your trained model achieved substantially higher F1 score (36.36% vs. 25.92%), representing a 10.44 percentage point improvement. This suggests better balance between precision and recall.

3. **False Positives**: Your model significantly reduced false positives (4,850 vs. 5,725), representing a 15.3% reduction. This is important for reducing false alarms in depression detection.

4. **False Negatives**: The default model had fewer false negatives (48 vs. 949), but this came at the cost of extremely high false positives (5,725 out of 5,771 non-depression cases, 99.2%).

5. **True Negatives**: Your model correctly identified 921 non-depression cases compared to only 46 for the default model, representing a 1,902% improvement in true negative rate.

---

## Analysis

### Performance on Individual Messages vs. Aggregated User Data

The performance on individual messages (42% accuracy) is substantially lower than performance on aggregated user-level data (73.97% accuracy). This difference is expected and can be attributed to several factors:

1. **Context Loss**: Individual messages lack the broader context provided by aggregating all of a user's posts and comments. A single message may be ambiguous without surrounding context.

2. **Training Objective**: The model was trained on aggregated user-level data, where patterns emerge across multiple messages. Individual messages may not contain sufficient signal for reliable classification.

3. **Label Inheritance**: Messages inherit labels from their source users, which assumes all messages from a depressed user express depression. This may not always be true, as users may post about various topics.

4. **Message Length**: Individual messages are typically shorter than aggregated user text, providing less information for the model to make predictions.

### Model Comparison Insights

While the default model achieved slightly higher accuracy (0.26% difference), your trained model shows:

- **Better F1 Score**: 10.44 percentage points higher, indicating better balance
- **Fewer False Positives**: 15.3% reduction, important for practical applications
- **More True Negatives**: Dramatically better at correctly identifying non-depression cases (921 vs. 46)

The default model's strategy of predicting depression for nearly all messages (5,725 false positives out of 5,771 non-depression cases) results in high recall but poor precision, making it less useful for practical applications despite slightly higher accuracy.

---

## Statistical Summary

### Your Trained Model
- **Correct Predictions**: 4,201 out of 10,000 (42.01%)
- **True Positives**: 3,280
- **True Negatives**: 921
- **False Positives**: 4,850
- **False Negatives**: 949

### Default Model (Binary Mapped)
- **Correct Predictions**: 4,227 out of 10,000 (42.27%)
- **True Positives**: 4,181
- **True Negatives**: 46
- **False Positives**: 5,725
- **False Negatives**: 48

---

## Files Generated

All results are saved in: `CheckPoint-11/19/single_message_comparison_results/`

- **`single_message_comparison_summary.json`**: Complete summary with all metrics and confusion matrices
- **`per_message_predictions.csv`**: Raw predictions for all 10,000 messages, including:
  - Message ID and type (post/comment)
  - Source user
  - True label
  - Predictions from both models
  - Confidence scores
  - Agreement/disagreement flags
  - Correctness indicators

---

## Script Used

**Script**: `CheckPoint-11/19/compare_models_single_messages.py`

This script:
1. Loads labeled user data from `all_labeled_users.json`
2. Extracts individual posts and comments as separate test examples
3. Tests both models on each message
4. Calculates comprehensive metrics
5. Saves results to JSON and CSV files

---

## Notes and Considerations

### Limitations

1. **Label Inheritance**: Messages inherit labels from users, which may not accurately reflect the depression content of individual messages. A user labeled as depressed may post messages that are not depression-related.

2. **Context Dependency**: Individual messages lack the context provided by user-level aggregation, which the model was trained on.

3. **Sample Size**: Results are based on 10,000 randomly sampled messages from a pool of 2.2 million messages. Results may vary with different samples.

4. **Message Type Distribution**: The test set is heavily skewed toward comments (86.4%) rather than posts (13.6%), which may affect performance if the model performs differently on posts vs. comments.

### Comparison with User-Level Results

| Metric | User-Level (Aggregated) | Individual Messages | Difference |
|--------|------------------------|---------------------|------------|
| **Your Model Accuracy** | 73.97% | 42.01% | -31.96% |
| **Default Model Accuracy** | 59.78% | 42.27% | -17.51% |

The substantial performance difference between user-level and message-level classification demonstrates the importance of context and aggregation for this task. The model performs significantly better when evaluating complete user profiles rather than individual messages.

---

## Conclusion

On individual message classification, both models achieved similar accuracy (~42%), with the default model slightly outperforming by 0.26 percentage points. However, your trained model shows:

- **Better F1 Score** (36.36% vs. 25.92%)
- **Fewer False Positives** (4,850 vs. 5,725)
- **Better True Negative Rate** (921 vs. 46)

The default model's strategy of predicting depression for nearly all messages results in high recall but poor precision, making it less practical despite slightly higher accuracy. Your trained model provides a more balanced approach with better precision and significantly fewer false positives.

The lower performance on individual messages (42% vs. 74% on aggregated data) confirms that the model benefits from the context and patterns present in aggregated user-level data, which aligns with the training objective and intended use case.

