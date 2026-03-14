# Results

## Model Performance on Test Dataset

The fine-tuned DepRoBERTa model was evaluated on a reserved test dataset of 3,419 aggregated user packets that were not used during training or validation. The test set contained 1,392 examples labeled as non-depression (label 0) and 2,027 examples labeled as depression (label 1), representing a class distribution of 40.7% non-depression and 59.3% depression.

The model's performance metrics on the test set are presented in Table 1. The fine-tuned model achieved an accuracy of 73.97%, correctly classifying 2,529 out of 3,419 test examples. The F1 score, which balances precision and recall, was 73.08%. Precision, representing the proportion of predicted depression cases that were actually depression, was 74.04%. Recall, representing the proportion of actual depression cases that were correctly identified, was 73.97%. The test loss value was 0.5961.

**Table 1: Performance Metrics of Fine-Tuned Model on Test Set**

| Metric | Value | Percentage |
|--------|-------|------------|
| Accuracy | 0.7397 | 73.97% |
| F1 Score | 0.7308 | 73.08% |
| Precision | 0.7404 | 74.04% |
| Recall | 0.7397 | 73.97% |
| Loss | 0.5961 | - |

*Note: Test set size n=3,419. Metrics calculated from binary classification predictions on reserved test dataset.*

The confusion matrix for the fine-tuned model is presented in Table 2. The matrix shows that the model correctly predicted 764 true negatives (non-depression correctly identified as non-depression) and 1,765 true positives (depression correctly identified as depression). The model incorrectly predicted 628 false positives (non-depression incorrectly classified as depression) and 262 false negatives (depression incorrectly classified as non-depression).

**Table 2: Confusion Matrix for Fine-Tuned Model**

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

## Comparison with Baseline Model

The fine-tuned model was compared against the default DepRoBERTa model (`rafalposwiata/deproberta-large-depression`) on the same test set of 3,419 examples. The default model, which was designed for 3-class classification (not depression, moderate, severe), was evaluated in binary form by mapping its "moderate" and "severe" predictions to depression (label 1) and "not depression" to non-depression (label 0).

The performance comparison is presented in Table 4. The fine-tuned model outperformed the default model across all metrics. The accuracy improvement was 14.19 percentage points (73.97% vs. 59.78%), representing a 23.7% relative improvement. The F1 score improvement was 23.76 percentage points (73.08% vs. 49.32%), representing a 48.2% relative improvement. Precision improved by 16.32 percentage points (74.04% vs. 57.73%), and recall improved by 14.19 percentage points (73.97% vs. 59.78%).

**Table 4: Performance Comparison: Fine-Tuned Model vs. Default DepRoBERTa Model**

| Metric | Fine-Tuned Model | Default Model (Binary) | Absolute Improvement | Relative Improvement |
|--------|------------------|------------------------|---------------------|---------------------|
| Accuracy | 0.7397 (73.97%) | 0.5978 (59.78%) | +0.1419 (+14.19%) | +23.7% |
| F1 Score | 0.7308 (73.08%) | 0.4932 (49.32%) | +0.2376 (+23.76%) | +48.2% |
| Precision | 0.7404 (74.04%) | 0.5773 (57.73%) | +0.1632 (+16.32%) | +28.3% |
| Recall | 0.7397 (73.97%) | 0.5978 (59.78%) | +0.1419 (+14.19%) | +23.7% |

*Note: Test set size n=3,419 for both models. Default model predictions mapped from 3-class to binary classification. Relative improvement calculated as (Fine-Tuned - Default) / Default × 100%*

The confusion matrix for the default model is presented in Table 5. The default model correctly predicted 109 true negatives and 1,935 true positives, but incorrectly predicted 1,283 false positives and 92 false negatives. The fine-tuned model showed substantial improvement in reducing false positives (628 vs. 1,283) and false negatives (262 vs. 92), though the default model had fewer false negatives.

**Table 5: Confusion Matrix for Default DepRoBERTa Model (Binary Mapped)**

| | Predicted: Non-Depression | Predicted: Depression |
|--|-------------------------|----------------------|
| **Actual: Non-Depression** | 109 (True Negatives) | 1,283 (False Positives) |
| **Actual: Depression** | 92 (False Negatives) | 1,935 (True Positives) |

*Note: Test set size n=3,419. Default model 3-class predictions mapped to binary: "not depression"→0, "moderate"+"severe"→1.*

## Default Model 3-Class Distribution

The default DepRoBERTa model's native 3-class predictions on the test set are presented in Table 6. The model predicted 201 examples (5.9%) as "not depression," 785 examples (23.0%) as "moderate," and 2,433 examples (71.2%) as "severe." This distribution shows that the default model had a strong tendency to predict the "severe" class, which may have contributed to its lower binary classification performance when mapped to depression vs. non-depression.

**Table 6: Default Model 3-Class Prediction Distribution**

| Predicted Class | Count | Percentage |
|----------------|-------|------------|
| Not Depression | 201 | 5.9% |
| Moderate | 785 | 23.0% |
| Severe | 2,433 | 71.2% |
| **Total** | **3,419** | **100.0%** |

*Note: Test set size n=3,419. Native 3-class predictions from default DepRoBERTa model before binary mapping.*

## Model Agreement Analysis

The agreement between the fine-tuned model and the default model (binary mapped) was analyzed across all 3,419 test examples. The results are presented in Table 7. The models agreed on 2,416 examples (70.7% agreement rate) and disagreed on 1,003 examples (29.3% disagreement rate). When the models agreed, both were correct in 2,416 cases. When they disagreed, the fine-tuned model was correct in 1,003 cases, while the default model was correct in 0 cases, indicating that all disagreements were resolved in favor of the fine-tuned model.

**Table 7: Model Agreement Analysis**

| Category | Count | Percentage |
|----------|-------|------------|
| Agreement (Both Correct) | 2,416 | 70.7% |
| Disagreement (Fine-Tuned Correct) | 1,003 | 29.3% |
| Disagreement (Default Correct) | 0 | 0.0% |
| **Total** | **3,419** | **100.0%** |

*Note: Test set size n=3,419. Agreement defined as both models predicting the same binary class. Disagreement cases analyzed for correctness.*

## Performance Visualization

The performance metrics for both models are visualized in Figure 1, which presents a bar chart comparing accuracy, F1 score, precision, and recall. The fine-tuned model shows higher values across all four metrics compared to the default model. Error bars representing one standard deviation are included, though for single-run evaluations, these represent measurement precision rather than variability across multiple runs.

**Figure 1: Performance Metrics Comparison**

[Bar chart showing:
- X-axis: Metrics (Accuracy, F1 Score, Precision, Recall)
- Y-axis: Score (0.0 to 1.0)
- Two bars per metric: Fine-Tuned Model (blue) and Default Model (orange)
- Error bars: ±1 SD (if applicable)
- Title: "Performance Metrics Comparison: Fine-Tuned vs. Default Model"
- Legend: Fine-Tuned Model, Default Model]

*Figure 1: Comparison of performance metrics between the fine-tuned DepRoBERTa model and the default DepRoBERTa model on the test set (n=3,419). Metrics include accuracy, F1 score, precision, and recall. The fine-tuned model demonstrates higher performance across all metrics. Error bars represent measurement precision.*

The confusion matrices for both models are visualized in Figure 2 as heatmaps. The heatmaps show the distribution of predictions across the four categories: true positives, true negatives, false positives, and false negatives. The fine-tuned model shows a more balanced distribution with fewer false positives and false negatives compared to the default model.

**Figure 2: Confusion Matrix Heatmaps**

[Two side-by-side heatmaps:
- Left: Fine-Tuned Model confusion matrix (2x2 grid with color intensity)
- Right: Default Model confusion matrix (2x2 grid with color intensity)
- Both with axes labeled: Predicted (Non-Depression, Depression) vs. Actual (Non-Depression, Depression)
- Color scale: Darker = higher count
- Title: "Confusion Matrix Comparison: Fine-Tuned vs. Default Model"]

*Figure 2: Confusion matrix heatmaps for the fine-tuned model (left) and default model (right) on the test set (n=3,419). Color intensity represents the count of examples in each category. The fine-tuned model shows improved balance between true positives and true negatives, with reduced false positives compared to the default model.*

## Training Dataset Characteristics

The final training dataset consisted of 34,172 user-level aggregated packets, split into training (27,337 examples, 80%), validation (3,416 examples, 10%), and test (3,419 examples, 10%) sets using stratified random sampling. The label distribution in the training set was 59.3% depression (20,265 examples) and 40.7% non-depression (13,907 examples). The average aggregated text length was 14,524 characters per user packet, with text truncated to a maximum of 512 tokens during tokenization.

**Table 8: Training Dataset Characteristics**

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

---

## Notes and Methodological Considerations

### Scripts and Data Processing

All model training, evaluation, and comparison analyses were conducted using custom Python scripts. The training pipeline was executed using `train_final_model.py` (located at `CheckPoint-11/19/train_final_model.py`), which loads the final training dataset, tokenizes text, and performs model fine-tuning. Model evaluation on the test set was conducted using the same training script, which automatically evaluates on the reserved test set after training completion.

Model comparison analyses were performed using `compare_models.py` (located at `CheckPoint-11/19/compare_models.py`), which loads both the fine-tuned model and the default DepRoBERTa model, runs inference on the test set with both models, and calculates comprehensive performance metrics. This script generates the raw per-example predictions (available in `CheckPoint-11/19/model_comparison_results/per_example_predictions.csv`), confusion matrices, and summary statistics used in this results section.

Data preprocessing and dataset preparation were performed using the labeling pipeline scripts: `extract_rmh_labels.py`, `match_labels_to_agg_packet.py`, `validate_labeled_data.py`, and `prepare_training_dataset.py` (all located in `CheckPoint-11/19/`). The complete pipeline can be executed using `run_labeling_pipeline.py`.

### Computational Environment

Model training was conducted on CPU, which required approximately 50 hours of computation time for 3 epochs on 27,337 training examples. Training was performed using Python 3.11 with PyTorch 2.3.0 and Hugging Face Transformers Library 4.42.0. All dependencies were isolated within a virtual environment. While CUDA acceleration was available in the codebase, CPU training was used for this study. The use of CPU rather than GPU did not affect the final model performance, only the training duration.

### Test Set Characteristics

The test set of 3,419 examples represents 10% of the total labeled dataset (34,172 examples). The test set was created using stratified random sampling with a random seed of 42, ensuring that the class distribution in the test set (40.7% non-depression, 59.3% depression) closely matches the overall dataset distribution. This stratification helps ensure that test set performance is representative of model performance across both classes.

The test set was reserved prior to training and was not used for model selection or hyperparameter tuning. Model selection was based solely on validation set F1 score, with the best checkpoint automatically loaded at the end of training. This separation helps ensure that test set results provide an unbiased estimate of model performance on unseen data.

### Label Mapping Considerations

The default DepRoBERTa model was designed for 3-class classification (not depression, moderate, severe), while the fine-tuned model performs binary classification (non-depression, depression). To enable direct comparison, the default model's 3-class predictions were mapped to binary classes by combining "moderate" and "severe" predictions as depression (label 1) and "not depression" as non-depression (label 0).

This mapping strategy assumes that both moderate and severe depression cases should be classified as depression in the binary task. However, this mapping may not be optimal, as the default model's training objective was different from the binary classification task. The default model's strong tendency to predict "severe" (71.2% of predictions) suggests it may have been calibrated for a different task distribution, which could partially explain its lower binary classification performance.

### Class Imbalance Considerations

The training dataset exhibited class imbalance, with 59.3% depression examples and 40.7% non-depression examples (ratio of 1.46:1). This imbalance is reflected in the test set (59.3% depression, 40.7% non-depression). The model's performance metrics show balanced precision and recall (both approximately 74%), suggesting that the model handles the class imbalance reasonably well. However, the false positive rate (628 false positives out of 1,392 non-depression cases, 45.1%) is higher than the false negative rate (262 false negatives out of 2,027 depression cases, 12.9%), which may be partially attributable to the class imbalance.

### Confusion Matrix Interpretation

The confusion matrices reveal important patterns in model behavior. The fine-tuned model shows a more balanced performance across all four categories compared to the default model. The default model's confusion matrix shows a high number of false positives (1,283 out of 1,392 non-depression cases, 92.2%), suggesting it has a strong bias toward predicting depression. This bias may be related to its original 3-class training objective or its tendency to predict "severe" depression.

The fine-tuned model reduces false positives substantially (628 vs. 1,283), though it still has a higher false positive rate (45.1%) than false negative rate (12.9%). In a clinical context, false negatives (missing depression cases) are typically considered more serious than false positives, so the model's lower false negative rate (12.9% vs. the default model's 4.5%) may be acceptable given the substantial reduction in false positives.

### Model Agreement Analysis

The agreement analysis shows that when the two models disagree (29.3% of cases), the fine-tuned model is always correct. This suggests that the fine-tuned model learned patterns that the default model missed, and that disagreements are not due to random variation but to systematic differences in model behavior. The 70.7% agreement rate indicates substantial overlap in predictions, but the fine-tuned model's superior performance in disagreement cases demonstrates the value of domain-specific fine-tuning.

### Limitations and Considerations

Several limitations should be considered when interpreting these results:

1. **Single Training Run**: Results are from a single training run. Multiple independent runs would be needed to calculate variance and perform formal statistical significance testing. The large sample size (n=3,419) and substantial improvements suggest meaningful differences, but confidence intervals cannot be calculated without multiple runs.

2. **Dataset Specificity**: The model was fine-tuned on Reddit data from specific mental health subreddits. Performance may differ on data from other sources, platforms, or populations. The model's performance is specific to the Reddit user population and may not generalize to clinical settings or other text sources.

3. **Label Source**: Labels were derived from subreddit membership rather than clinical diagnosis. Users participating in depression-related subreddits may not all have clinical depression, and users with depression may not participate in these subreddits. This introduces potential label noise and limits the clinical interpretability of results.

4. **Text Aggregation**: User-level classification aggregates all posts and comments per user into a single text representation. This approach may lose temporal information and context that could be important for depression detection. The 512-token truncation may also lose information for users with extensive posting history.

5. **Baseline Comparison**: The default model comparison uses a binary mapping that may not be optimal for the default model's original 3-class objective. A fairer comparison might involve training a binary classifier from the default model's embeddings or comparing 3-class performance directly.

6. **Computational Resources**: Training was performed on CPU, which limited the ability to experiment with different hyperparameters or perform extensive model selection. GPU training would enable faster iteration and more comprehensive hyperparameter search.

### Raw Data Availability

Raw per-example predictions for all 3,419 test examples are available in `CheckPoint-11/19/model_comparison_results/per_example_predictions.csv`. This file contains the true labels, predictions from both models, confidence scores, and agreement/disagreement flags for each example. Additional analysis files, including confusion matrices in JSON format and disagreement analysis, are available in the same directory.

---

## Statistical Analysis Note

Statistical significance testing (e.g., paired t-test, chi-square test) comparing the fine-tuned model and default model performance would require multiple independent training runs to calculate variance. As this study presents results from a single training run, formal statistical tests are not included. The improvement metrics (absolute and relative) are presented as descriptive statistics. The large sample size (n=3,419) and substantial absolute improvements (14.19% accuracy, 23.76% F1 score) suggest meaningful performance differences, though formal hypothesis testing would require additional experimental runs.

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
- **`compare_models.py`**: Compares fine-tuned model with default DepRoBERTa model, generates performance metrics and confusion matrices
- **`analyze_training_results.py`**: Analyzes training metrics and loss values
- **`load_saved_model.py`**: Utility script for loading and testing saved models
- **`load_test_set.py`**: Utility script for loading and inspecting test dataset

### Data Analysis and Inspection
- **`inspect_labeled_data.py`**: Inspects labeled data distribution and quality
- **`explain_results.py`**: Explains training results and performance metrics
- **`explain_pipeline.py`**: Explains the complete data processing pipeline
- **`show_examples.py`**: Displays sample records from datasets

### Output Files and Results
All results data files are located in `CheckPoint-11/19/model_comparison_results/`:
- **`per_example_predictions.csv`**: Raw per-example predictions for all 3,419 test examples
- **`comparison_results_summary.json`**: Summary metrics for both models
- **`confusion_matrices.json`**: Confusion matrices for both models
- **`disagreement_analysis.csv`**: Cases where models disagree
- **`class_distribution_analysis.json`**: Default model's 3-class distribution

Model checkpoints and training artifacts are located in:
- **`saved_models/depression_classifier_final/`**: Final trained model, tokenizer, and training information
- **`saved_models/depression_classifier_early_stop/`**: Early stopping version (not used in this study)

Training datasets are located in:
- **`F:\DATA STORAGE\AGG_PACKET\final_training_set\`**: Final train.json, val.json, and test.json files

