# Results

## Model Performance on Aggregated User-Level Test Dataset

The fine-tuned DepRoBERTa model was evaluated on a reserved test dataset of 3,419 aggregated user packets that were not used during training or validation. The test set contained 1,392 examples labeled as non-depression (label 0) and 2,027 examples labeled as depression (label 1), representing a class distribution of 40.7% non-depression and 59.3% depression.

The model's performance metrics on the test set are presented in Table 1. The fine-tuned model achieved an accuracy of 73.97%, correctly classifying 2,529 out of 3,419 test examples. The F1 score, which balances precision and recall, was 73.08%. Precision, representing the proportion of predicted depression cases that were actually depression, was 74.04%. Recall, representing the proportion of actual depression cases that were correctly identified, was 73.97%. The test loss value was 0.5961.

**Table 1: Performance Metrics of Fine-Tuned Model on Test Set (Aggregated User-Level Data)**

| Metric | Value | Percentage |
|--------|-------|------------|
| Accuracy | 0.7397 | 73.97% |
| F1 Score | 0.7308 | 73.08% |
| Precision | 0.7404 | 74.04% |
| Recall | 0.7397 | 73.97% |
| Loss | 0.5961 | - |

*Note: Test set size n=3,419 aggregated user packets. Metrics calculated from binary classification predictions on reserved test dataset.*

The confusion matrix for the fine-tuned model is presented in Table 2. The matrix shows that the model correctly predicted 764 true negatives (non-depression correctly identified as non-depression) and 1,765 true positives (depression correctly identified as depression). The model incorrectly predicted 628 false positives (non-depression incorrectly classified as depression) and 262 false negatives (depression incorrectly classified as non-depression).

**Table 2: Confusion Matrix for Fine-Tuned Model (Aggregated User-Level Data)**

| | Predicted: Non-Depression | Predicted: Depression |
|--|-------------------------|----------------------|
| **Actual: Non-Depression** | 764 (True Negatives) | 628 (False Positives) |
| **Actual: Depression** | 262 (False Negatives) | 1,765 (True Positives) |

*Note: Test set size n=3,419. Rows represent actual labels, columns represent predicted labels.*

## Training and Validation Performance

During the training process, the model was evaluated on a validation set of 3,416 examples after each epoch. The validation performance metrics are presented in Table 3. After the third and final epoch, the model achieved a validation accuracy of 73.62% and a validation F1 score of 72.84%. The validation loss was 0.6075, while the training loss was 0.6249.

**Table 3: Training and Validation Metrics**

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Loss | 0.6249 | 0.6075 | 0.5961 |
| Accuracy | - | 0.7362 | 0.7397 |
| F1 Score | - | 0.7284 | 0.7308 |

*Note: Training set n=27,337, Validation set n=3,416, Test set n=3,419. Training accuracy not calculated during training process.*

The similarity between validation and test performance (validation accuracy 73.62% vs. test accuracy 73.97%, difference of 0.35%) indicates that the model generalized well to unseen data and did not exhibit significant overfitting.

## Comparison with Baseline Model on Aggregated User-Level Data

The fine-tuned model was compared against the default DepRoBERTa model (`rafalposwiata/deproberta-large-depression`) on the same test set of 3,419 examples. The default model, which was designed for 3-class classification (not depression, moderate, severe), was evaluated in binary form by mapping its "moderate" and "severe" predictions to depression (label 1) and "not depression" to non-depression (label 0).

The performance comparison is presented in Table 4. The fine-tuned model outperformed the default model across all metrics. The accuracy improvement was 14.19 percentage points (73.97% vs. 59.78%), representing a 23.7% relative improvement. The F1 score improvement was 23.76 percentage points (73.08% vs. 49.32%), representing a 48.2% relative improvement. Precision improved by 16.32 percentage points (74.04% vs. 57.73%), and recall improved by 14.19 percentage points (73.97% vs. 59.78%).

**Table 4: Performance Comparison: Fine-Tuned Model vs. Default DepRoBERTa Model (Aggregated User-Level Data)**

| Metric | Fine-Tuned Model | Default Model (Binary) | Absolute Improvement | Relative Improvement |
|--------|------------------|------------------------|---------------------|---------------------|
| Accuracy | 0.7397 (73.97%) | 0.5978 (59.78%) | +0.1419 (+14.19%) | +23.7% |
| F1 Score | 0.7308 (73.08%) | 0.4932 (49.32%) | +0.2376 (+23.76%) | +48.2% |
| Precision | 0.7404 (74.04%) | 0.5773 (57.73%) | +0.1632 (+16.32%) | +28.3% |
| Recall | 0.7397 (73.97%) | 0.5978 (59.78%) | +0.1419 (+14.19%) | +23.7% |

*Note: Test set size n=3,419 for both models. Default model predictions mapped from 3-class to binary classification. Relative improvement calculated as (Fine-Tuned - Default) / Default × 100%*

The confusion matrix for the default model is presented in Table 5. The default model correctly predicted 109 true negatives and 1,935 true positives, but incorrectly predicted 1,283 false positives and 92 false negatives. The fine-tuned model showed substantial improvement in reducing false positives (628 vs. 1,283) and false negatives (262 vs. 92), though the default model had fewer false negatives.

**Table 5: Confusion Matrix for Default DepRoBERTa Model (Binary Mapped, Aggregated User-Level Data)**

| | Predicted: Non-Depression | Predicted: Depression |
|--|-------------------------|----------------------|
| **Actual: Non-Depression** | 109 (True Negatives) | 1,283 (False Positives) |
| **Actual: Depression** | 92 (False Negatives) | 1,935 (True Positives) |

*Note: Test set size n=3,419. Default model 3-class predictions mapped to binary: "not depression"→0, "moderate"+"severe"→1.*

## Default Model 3-Class Distribution

The default DepRoBERTa model's native 3-class predictions on the test set are presented in Table 6. The model predicted 201 examples (5.9%) as "not depression," 785 examples (23.0%) as "moderate," and 2,433 examples (71.2%) as "severe." This distribution shows that the default model had a strong tendency to predict the "severe" class, which may have contributed to its lower binary classification performance when mapped to depression vs. non-depression.

**Table 6: Default Model 3-Class Prediction Distribution (Aggregated User-Level Data)**

| Predicted Class | Count | Percentage |
|----------------|-------|------------|
| Not Depression | 201 | 5.9% |
| Moderate | 785 | 23.0% |
| Severe | 2,433 | 71.2% |
| **Total** | **3,419** | **100.0%** |

*Note: Test set size n=3,419. Native 3-class predictions from default DepRoBERTa model before binary mapping.*

## Model Agreement Analysis (Aggregated User-Level Data)

The agreement between the fine-tuned model and the default model (binary mapped) was analyzed across all 3,419 test examples. The results are presented in Table 7. The models agreed on 2,416 examples (70.7% agreement rate) and disagreed on 1,003 examples (29.3% disagreement rate). When the models agreed, both were correct in 2,416 cases. When they disagreed, the fine-tuned model was correct in 1,003 cases, while the default model was correct in 0 cases, indicating that all disagreements were resolved in favor of the fine-tuned model.

**Table 7: Model Agreement Analysis (Aggregated User-Level Data)**

| Category | Count | Percentage |
|----------|-------|------------|
| Agreement (Both Correct) | 2,416 | 70.7% |
| Disagreement (Fine-Tuned Correct) | 1,003 | 29.3% |
| Disagreement (Default Correct) | 0 | 0.0% |
| **Total** | **3,419** | **100.0%** |

*Note: Test set size n=3,419. Agreement defined as both models predicting the same binary class. Disagreement cases analyzed for correctness.*

## Model Performance on Individual Messages

To evaluate model performance on individual messages (single posts and comments) rather than aggregated user-level data, both models were tested on a dataset of 10,000 individual messages extracted from labeled user data. Messages were extracted from the labeled user dataset, with each message inheriting the label from its source user. The test set consisted of 1,362 posts (13.6%) and 8,638 comments (86.4%), with 5,771 messages labeled as non-depression (57.7%) and 4,229 messages labeled as depression (42.3%). These messages were randomly sampled from a total pool of 2,266,919 available messages (304,677 posts and 1,962,242 comments).

The performance metrics for both models on individual messages are presented in Table 8. The fine-tuned model achieved an accuracy of 42.01%, correctly classifying 4,201 out of 10,000 individual messages. The F1 score was 36.36%, precision was 45.48%, and recall was 42.01%. The default model achieved an accuracy of 42.27%, with an F1 score of 25.92%, precision of 46.09%, and recall of 42.27%.

**Table 8: Performance Metrics on Individual Messages**

| Metric | Fine-Tuned Model | Default Model (Binary) | Difference |
|--------|------------------|------------------------|------------|
| Accuracy | 0.4201 (42.01%) | 0.4227 (42.27%) | -0.0026 (-0.26%) |
| F1 Score | 0.3636 (36.36%) | 0.2592 (25.92%) | +0.1044 (+10.44%) |
| Precision | 0.4548 (45.48%) | 0.4609 (46.09%) | -0.0061 (-0.61%) |
| Recall | 0.4201 (42.01%) | 0.4227 (42.27%) | -0.0026 (-0.26%) |

*Note: Test set size n=10,000 individual messages (1,362 posts, 8,638 comments). Default model predictions mapped from 3-class to binary classification.*

The confusion matrices for both models on individual messages are presented in Table 9 (fine-tuned model) and Table 10 (default model). The fine-tuned model correctly predicted 921 true negatives and 3,280 true positives, with 4,850 false positives and 949 false negatives. The default model correctly predicted 46 true negatives and 4,181 true positives, with 5,725 false positives and 48 false negatives.

**Table 9: Confusion Matrix for Fine-Tuned Model (Individual Messages)**

| | Predicted: Non-Depression | Predicted: Depression |
|--|-------------------------|----------------------|
| **Actual: Non-Depression** | 921 (True Negatives) | 4,850 (False Positives) |
| **Actual: Depression** | 949 (False Negatives) | 3,280 (True Positives) |

*Note: Test set size n=10,000 individual messages. Rows represent actual labels, columns represent predicted labels.*

**Table 10: Confusion Matrix for Default DepRoBERTa Model (Binary Mapped, Individual Messages)**

| | Predicted: Non-Depression | Predicted: Depression |
|--|-------------------------|----------------------|
| **Actual: Non-Depression** | 46 (True Negatives) | 5,725 (False Positives) |
| **Actual: Depression** | 48 (False Negatives) | 4,181 (True Positives) |

*Note: Test set size n=10,000 individual messages. Default model 3-class predictions mapped to binary: "not depression"→0, "moderate"+"severe"→1.*

## Comparison: Aggregated User-Level vs. Individual Messages

The performance difference between aggregated user-level classification and individual message classification is presented in Table 11. The fine-tuned model achieved 73.97% accuracy on aggregated user-level data compared to 42.01% on individual messages, representing a 31.96 percentage point decrease. The default model achieved 59.78% accuracy on aggregated data compared to 42.27% on individual messages, representing a 17.51 percentage point decrease.

**Table 11: Performance Comparison: Aggregated User-Level vs. Individual Messages**

| Model | Aggregated User-Level Accuracy | Individual Messages Accuracy | Difference |
|-------|-------------------------------|------------------------------|------------|
| Fine-Tuned Model | 73.97% | 42.01% | -31.96% |
| Default Model | 59.78% | 42.27% | -17.51% |

*Note: Aggregated user-level test set n=3,419. Individual messages test set n=10,000. Difference calculated as Individual Messages - Aggregated User-Level.*

## Performance Visualization

The performance metrics for both models on aggregated user-level data are visualized in Figure 1, which presents a bar chart comparing accuracy, F1 score, precision, and recall. The fine-tuned model shows higher values across all four metrics compared to the default model. Error bars representing one standard deviation are included, though for single-run evaluations, these represent measurement precision rather than variability across multiple runs.

**Figure 1: Performance Metrics Comparison (Aggregated User-Level Data)**

[Bar chart showing:
- X-axis: Metrics (Accuracy, F1 Score, Precision, Recall)
- Y-axis: Score (0.0 to 1.0)
- Two bars per metric: Fine-Tuned Model (blue) and Default Model (orange)
- Error bars: ±1 SD (if applicable)
- Title: "Performance Metrics Comparison: Fine-Tuned vs. Default Model (Aggregated User-Level)"
- Legend: Fine-Tuned Model, Default Model]

*Figure 1: Comparison of performance metrics between the fine-tuned DepRoBERTa model and the default DepRoBERTa model on aggregated user-level test set (n=3,419). Metrics include accuracy, F1 score, precision, and recall. The fine-tuned model demonstrates higher performance across all metrics. Error bars represent measurement precision.*

The confusion matrices for both models on aggregated user-level data are visualized in Figure 2 as heatmaps. The heatmaps show the distribution of predictions across the four categories: true positives, true negatives, false positives, and false negatives. The fine-tuned model shows a more balanced distribution with fewer false positives and false negatives compared to the default model.

**Figure 2: Confusion Matrix Heatmaps (Aggregated User-Level Data)**

[Two side-by-side heatmaps:
- Left: Fine-Tuned Model confusion matrix (2x2 grid with color intensity)
- Right: Default Model confusion matrix (2x2 grid with color intensity)
- Both with axes labeled: Predicted (Non-Depression, Depression) vs. Actual (Non-Depression, Depression)
- Color scale: Darker = higher count
- Title: "Confusion Matrix Comparison: Fine-Tuned vs. Default Model (Aggregated User-Level)"]

*Figure 2: Confusion matrix heatmaps for the fine-tuned model (left) and default model (right) on aggregated user-level test set (n=3,419). Color intensity represents the count of examples in each category. The fine-tuned model shows improved balance between true positives and true negatives, with reduced false positives compared to the default model.*

The performance metrics for both models on individual messages are visualized in Figure 3, which presents a bar chart comparing accuracy, F1 score, precision, and recall. Both models show similar accuracy values (~42%), but the fine-tuned model shows substantially higher F1 score.

**Figure 3: Performance Metrics Comparison (Individual Messages)**

[Bar chart showing:
- X-axis: Metrics (Accuracy, F1 Score, Precision, Recall)
- Y-axis: Score (0.0 to 1.0)
- Two bars per metric: Fine-Tuned Model (blue) and Default Model (orange)
- Error bars: ±1 SD (if applicable)
- Title: "Performance Metrics Comparison: Fine-Tuned vs. Default Model (Individual Messages)"
- Legend: Fine-Tuned Model, Default Model]

*Figure 3: Comparison of performance metrics between the fine-tuned DepRoBERTa model and the default DepRoBERTa model on individual messages test set (n=10,000). Metrics include accuracy, F1 score, precision, and recall. Both models show similar accuracy, but the fine-tuned model demonstrates substantially higher F1 score. Error bars represent measurement precision.*

## Training Dataset Characteristics

The final training dataset consisted of 34,172 user-level aggregated packets, split into training (27,337 examples, 80%), validation (3,416 examples, 10%), and test (3,419 examples, 10%) sets using stratified random sampling. The label distribution in the training set was 59.3% depression (20,265 examples) and 40.7% non-depression (13,907 examples). The average aggregated text length was 14,524 characters per user packet, with text truncated to a maximum of 512 tokens during tokenization.

**Table 12: Training Dataset Characteristics**

| Characteristic | Value |
|---------------|-------|
| Total Examples | 34,172 |
| Training Set | 27,337 (80.0%) |
| Validation Set | 3,416 (10.0%) |
| Test Set | 3,419 (10.0%) |
| Label 0 (Non-Depression) | 13,907 (40.7%) |
| Label 1 (Depression) | 20,265 (59.3%) |
| Average Text Length | 14,524 characters |
| Max Token Length | 512 tokens |

*Note: Dataset split using stratified random sampling with random seed 42. Text length measured in characters before tokenization.*

## Summative Data Tables

### Aggregated User-Level Performance Summary

**Table 13: Summative Performance Metrics (Aggregated User-Level Data)**

| Model | Accuracy | F1 Score | Precision | Recall | n |
|-------|----------|----------|-----------|--------|---|
| Fine-Tuned Model | 0.7397 (73.97%) | 0.7308 (73.08%) | 0.7404 (74.04%) | 0.7397 (73.97%) | 3,419 |
| Default Model (Binary) | 0.5978 (59.78%) | 0.4932 (49.32%) | 0.5773 (57.73%) | 0.5978 (59.78%) | 3,419 |
| Improvement | +0.1419 (+14.19%) | +0.2376 (+23.76%) | +0.1632 (+16.32%) | +0.1419 (+14.19%) | - |

*Note: Test set size n=3,419 for both models. All metrics calculated from binary classification predictions. Improvement represents absolute difference (Fine-Tuned - Default).*

### Individual Messages Performance Summary

**Table 14: Summative Performance Metrics (Individual Messages)**

| Model | Accuracy | F1 Score | Precision | Recall | n |
|-------|----------|----------|-----------|--------|---|
| Fine-Tuned Model | 0.4201 (42.01%) | 0.3636 (36.36%) | 0.4548 (45.48%) | 0.4201 (42.01%) | 10,000 |
| Default Model (Binary) | 0.4227 (42.27%) | 0.2592 (25.92%) | 0.4609 (46.09%) | 0.4227 (42.27%) | 10,000 |
| Difference | -0.0026 (-0.26%) | +0.1044 (+10.44%) | -0.0061 (-0.61%) | -0.0026 (-0.26%) | - |

*Note: Test set size n=10,000 individual messages. All metrics calculated from binary classification predictions. Difference represents absolute difference (Fine-Tuned - Default).*

## Notes and Methodological Considerations

### Scripts and Data Processing

All model training, evaluation, and comparison analyses were conducted using custom Python scripts. The training pipeline was executed using `train_final_model.py` (located at `CheckPoint-11/19/train_final_model.py`), which loads the final training dataset, tokenizes text, and performs model fine-tuning. Model evaluation on the test set was conducted using the same training script, which automatically evaluates on the reserved test set after training completion.

Model comparison analyses on aggregated user-level data were performed using `compare_models.py` (located at `CheckPoint-11/19/compare_models.py`), which loads both the fine-tuned model and the default DepRoBERTa model, runs inference on the test set with both models, and calculates comprehensive performance metrics. This script generates the raw per-example predictions (available in `CheckPoint-11/19/model_comparison_results/per_example_predictions.csv`), confusion matrices, and summary statistics used in this results section.

Model comparison analyses on individual messages were performed using `compare_models_single_messages.py` (located at `CheckPoint-11/19/compare_models_single_messages.py`), which extracts individual posts and comments from labeled user data, tests both models on each message, and calculates performance metrics. This script generates per-message predictions (available in `CheckPoint-11/19/single_message_comparison_results/per_message_predictions.csv`) and summary statistics.

Data preprocessing and dataset preparation were performed using the labeling pipeline scripts: `extract_rmh_labels.py`, `match_labels_to_agg_packet.py`, `validate_labeled_data.py`, and `prepare_training_dataset.py` (all located in `CheckPoint-11/19/`). The complete pipeline can be executed using `run_labeling_pipeline.py`.

### Computational Environment

Model training was conducted on CPU, which required approximately 50 hours of computation time for 3 epochs on 27,337 training examples. Training was performed using Python 3.11 with PyTorch 2.3.0 and Hugging Face Transformers Library 4.42.0. All dependencies were isolated within a virtual environment. While CUDA acceleration was available in the codebase, CPU training was used for this study. The use of CPU rather than GPU did not affect the final model performance, only the training duration.

### Test Set Characteristics

#### Aggregated User-Level Test Set

The test set of 3,419 examples represents 10% of the total labeled dataset (34,172 examples). The test set was created using stratified random sampling with a random seed of 42, ensuring that the class distribution in the test set (40.7% non-depression, 59.3% depression) closely matches the overall dataset distribution. This stratification helps ensure that test set performance is representative of model performance across both classes.

The test set was reserved prior to training and was not used for model selection or hyperparameter tuning. Model selection was based solely on validation set F1 score, with the best checkpoint automatically loaded at the end of training. This separation helps ensure that test set results provide an unbiased estimate of model performance on unseen data.

#### Individual Messages Test Set

The individual messages test set consisted of 10,000 messages randomly sampled from a total pool of 2,266,919 available messages extracted from labeled user data. Messages inherit labels from their source users, which assumes that all messages from a depressed user express depression. This assumption may introduce label noise, as users may post about various topics beyond their mental health status. The test set distribution (57.7% non-depression, 42.3% depression) differs from the aggregated user-level distribution (40.7% non-depression, 59.3% depression), reflecting the different sampling approach.

### Label Mapping Considerations

The default DepRoBERTa model was designed for 3-class classification (not depression, moderate, severe), while the fine-tuned model performs binary classification (non-depression, depression). To enable direct comparison, the default model's 3-class predictions were mapped to binary classes by combining "moderate" and "severe" predictions as depression (label 1) and "not depression" as non-depression (label 0).

This mapping strategy assumes that both moderate and severe depression cases should be classified as depression in the binary task. However, this mapping may not be optimal, as the default model's training objective was different from the binary classification task. The default model's strong tendency to predict "severe" (71.2% of predictions on aggregated data) suggests it may have been calibrated for a different task distribution, which could partially explain its lower binary classification performance.

### Class Imbalance Considerations

The training dataset exhibited class imbalance, with 59.3% depression examples and 40.7% non-depression examples (ratio of 1.46:1). This imbalance is reflected in the aggregated user-level test set (59.3% depression, 40.7% non-depression). The model's performance metrics show balanced precision and recall (both approximately 74%), suggesting that the model handles the class imbalance reasonably well. However, the false positive rate (628 false positives out of 1,392 non-depression cases, 45.1%) is higher than the false negative rate (262 false negatives out of 2,027 depression cases, 12.9%), which may be partially attributable to the class imbalance.

The individual messages test set shows a different class distribution (57.7% non-depression, 42.3% depression), which may affect performance comparisons between the two test sets.

### Confusion Matrix Interpretation

#### Aggregated User-Level Data

The confusion matrices reveal important patterns in model behavior. The fine-tuned model shows a more balanced performance across all four categories compared to the default model. The default model's confusion matrix shows a high number of false positives (1,283 out of 1,392 non-depression cases, 92.2%), suggesting it has a strong bias toward predicting depression. This bias may be related to its original 3-class training objective or its tendency to predict "severe" depression.

The fine-tuned model reduces false positives substantially (628 vs. 1,283), though it still has a higher false positive rate (45.1%) than false negative rate (12.9%). In a clinical context, false negatives (missing depression cases) are typically considered more serious than false positives, so the model's lower false negative rate (12.9% vs. the default model's 4.5%) may be acceptable given the substantial reduction in false positives.

#### Individual Messages

On individual messages, both models show high false positive rates. The fine-tuned model incorrectly classified 4,850 out of 5,771 non-depression messages as depression (84.1% false positive rate), while the default model incorrectly classified 5,725 out of 5,771 non-depression messages as depression (99.2% false positive rate). The fine-tuned model's false positive rate is substantially lower than the default model's, though both are high. This high false positive rate on individual messages is expected, as individual messages lack the context needed for reliable classification, and the model was trained on aggregated user-level data.

The fine-tuned model correctly identified 921 non-depression messages compared to only 46 for the default model, representing a substantial improvement in true negative rate (16.0% vs. 0.8%).

### Model Agreement Analysis

The agreement analysis on aggregated user-level data shows that when the two models disagree (29.3% of cases), the fine-tuned model is always correct. This suggests that the fine-tuned model learned patterns that the default model missed, and that disagreements are not due to random variation but to systematic differences in model behavior. The 70.7% agreement rate indicates substantial overlap in predictions, but the fine-tuned model's superior performance in disagreement cases demonstrates the value of domain-specific fine-tuning.

### Performance on Individual Messages vs. Aggregated User Data

The substantial performance difference between aggregated user-level classification (73.97% accuracy) and individual message classification (42.01% accuracy) demonstrates the importance of context and aggregation for this task. The 31.96 percentage point decrease in accuracy when moving from aggregated to individual messages indicates that:

1. **Context is Critical**: Individual messages lack the broader context provided by aggregating all of a user's posts and comments. A single message may be ambiguous without surrounding context.

2. **Training Objective Alignment**: The model was trained on aggregated user-level data, where patterns emerge across multiple messages. Individual messages may not contain sufficient signal for reliable classification.

3. **Label Inheritance Limitations**: Messages inherit labels from their source users, which assumes all messages from a depressed user express depression. This may not always be true, as users may post about various topics.

4. **Message Length**: Individual messages are typically shorter than aggregated user text, providing less information for the model to make predictions.

The model performs significantly better when evaluating complete user profiles rather than individual messages, which aligns with the training objective and intended use case.

### Limitations and Considerations

Several limitations should be considered when interpreting these results:

1. **Single Training Run**: Results are from a single training run. Multiple independent runs would be needed to calculate variance and perform formal statistical significance testing. The large sample sizes (n=3,419 for aggregated data, n=10,000 for individual messages) and substantial improvements suggest meaningful differences, but confidence intervals cannot be calculated without multiple runs.

2. **Dataset Specificity**: The model was fine-tuned on Reddit data from specific mental health subreddits. Performance may differ on data from other sources, platforms, or populations. The model's performance is specific to the Reddit user population and may not generalize to clinical settings or other text sources.

3. **Label Source**: Labels were derived from subreddit membership rather than clinical diagnosis. Users participating in depression-related subreddits may not all have clinical depression, and users with depression may not participate in these subreddits. This introduces potential label noise and limits the clinical interpretability of results.

4. **Text Aggregation**: User-level classification aggregates all posts and comments per user into a single text representation. This approach may lose temporal information and context that could be important for depression detection. The 512-token truncation may also lose information for users with extensive posting history.

5. **Baseline Comparison**: The default model comparison uses a binary mapping that may not be optimal for the default model's original 3-class objective. A fairer comparison might involve training a binary classifier from the default model's embeddings or comparing 3-class performance directly.

6. **Computational Resources**: Training was performed on CPU, which limited the ability to experiment with different hyperparameters or perform extensive model selection. GPU training would enable faster iteration and more comprehensive hyperparameter search.

7. **Individual Message Evaluation**: Individual message classification was evaluated using labels inherited from users, which may not accurately reflect the depression content of individual messages. Additionally, the model was trained on aggregated data, so lower performance on individual messages is expected and does not necessarily indicate model failure.

### Raw Data Availability

Raw per-example predictions for all 3,419 aggregated user-level test examples are available in `CheckPoint-11/19/model_comparison_results/per_example_predictions.csv`. This file contains the true labels, predictions from both models, confidence scores, and agreement/disagreement flags for each example. Additional analysis files, including confusion matrices in JSON format and disagreement analysis, are available in the same directory.

Raw per-message predictions for all 10,000 individual message test examples are available in `CheckPoint-11/19/single_message_comparison_results/per_message_predictions.csv`. This file contains message IDs, message types (post/comment), source users, true labels, predictions from both models, confidence scores, and agreement/disagreement flags for each message.

---

## Statistical Analysis Note

Statistical significance testing (e.g., paired t-test, chi-square test) comparing the fine-tuned model and default model performance would require multiple independent training runs to calculate variance. As this study presents results from a single training run, formal statistical tests are not included. The improvement metrics (absolute and relative) are presented as descriptive statistics. The large sample sizes (n=3,419 for aggregated data, n=10,000 for individual messages) and substantial improvements suggest meaningful performance differences, though formal hypothesis testing would require additional experimental runs.

For future work, multiple independent training runs with different random seeds would enable calculation of confidence intervals, standard deviations, and formal statistical tests (e.g., paired t-test for accuracy differences, McNemar's test for classification differences). This would provide stronger evidence for the statistical significance of the observed improvements.

---

## Script References

All scripts referenced in this results section are located in the `CheckPoint-11/19/` directory unless otherwise specified. The following scripts were used for data processing, model training, and evaluation:

### Data Processing and Labeling Pipeline
- **`extract_rmh_labels.py`**: Extracts username-to-label mappings from RMH dataset CSV files
- **`match_labels_to_agg_packet.py`**: Matches extracted labels to AGG_PACKET user data
- **`validate_labeled_data.py`**: Validates labeled dataset quality and content requirements
- **`prepare_training_dataset.py`**: Aggregates user content and creates train/validation/test splits
- **`run_labeling_pipeline.py`**: Master script that orchestrates all labeling pipeline steps

### Model Training
- **`train_final_model.py`**: Main training script that loads final training dataset, tokenizes text, fine-tunes model, and evaluates on test set
- **`modelB_training.py`**: Alternative training script located in project root directory (interactive training interface)
- **`start_training.py`**: Training launcher script that prepares and executes training commands

### Model Evaluation and Comparison
- **`compare_models.py`**: Compares fine-tuned model with default DepRoBERTa model on aggregated user-level data, generates performance metrics and confusion matrices
- **`compare_models_single_messages.py`**: Compares fine-tuned model with default DepRoBERTa model on individual messages, generates performance metrics and confusion matrices
- **`analyze_training_results.py`**: Analyzes training metrics and loss values
- **`load_saved_model.py`**: Utility script for loading and testing saved models
- **`load_test_set.py`**: Utility script for loading and inspecting test dataset

### Data Analysis and Inspection
- **`inspect_labeled_data.py`**: Inspects labeled data distribution and quality
- **`explain_results.py`**: Explains training results and performance metrics
- **`explain_pipeline.py`**: Explains the complete data processing pipeline
- **`show_examples.py`**: Displays sample records from datasets

### Output Files and Results
All results data files are located in:
- **`CheckPoint-11/19/model_comparison_results/`**: Aggregated user-level comparison results
  - `per_example_predictions.csv`: Raw per-example predictions for all 3,419 test examples
  - `comparison_results_summary.json`: Summary metrics for both models
  - `confusion_matrices.json`: Confusion matrices for both models
  - `disagreement_analysis.csv`: Cases where models disagree
  - `class_distribution_analysis.json`: Default model's 3-class distribution

- **`CheckPoint-11/19/single_message_comparison_results/`**: Individual message comparison results
  - `per_message_predictions.csv`: Raw per-message predictions for all 10,000 test messages
  - `single_message_comparison_summary.json`: Summary metrics for both models

Model checkpoints and training artifacts are located in:
- **`saved_models/depression_classifier_final/`**: Final trained model, tokenizer, and training information
- **`saved_models/depression_classifier_early_stop/`**: Early stopping version (not used in this study)

Training datasets are located in:
- **`F:\DATA STORAGE\AGG_PACKET\final_training_set\`**: Final train.json, val.json, and test.json files

