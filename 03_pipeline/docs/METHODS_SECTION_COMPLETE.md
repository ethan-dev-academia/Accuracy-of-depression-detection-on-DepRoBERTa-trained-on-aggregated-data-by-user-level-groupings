# Methods Section - Complete and Corrected

## Text Preprocessing and Packet Aggregation

All preprocessing and aggregation procedures were programmed with Python 3.11. Text normalization involved standardizing formatting to reduce external variation, removing hyperlinks, and filtering out removed or deleted content markers (e.g., "[removed]", "[deleted]"). Each user's posts and comments were aggregated chronologically based on timestamp to preserve temporal continuity. Posts were combined by concatenating title and content fields, and comments were appended sequentially. This aggregated text was combined with user metadata (see FIGURE) to form an 'aggregate packet'. 

Aggregated text was tokenized using the RoBERTa tokenizer (via Hugging Face's AutoTokenizer), and each packet was truncated to a maximum length of 512 tokens to comply with the model's input limitations. Each packet was assigned a unique identifier and stored in JSON format. The complete dataset consisted of 34,172 user-level aggregated packets.

The dataset was split into training, validation, and test sets using stratified random sampling to maintain class distribution across splits. The split was performed at a ratio of 80:10:10 (train:validation:test), with stratification by label to ensure balanced representation of depression (label 1) and non-depression (label 0) classes in each split. The random seed was set to 42 for reproducibility. This resulted in 27,337 training examples, 3,416 validation examples, and 3,419 test examples.

---

## Model Architecture and Tuning Procedure

Model tuning and inference were conducted using Python 3.11 with PyTorch version 2.3.0 and the Hugging Face Transformers Library Version 4.42.0. All dependencies (see Appendix) were isolated within a virtual environment using 'venv'. Training was performed on CPU, though the codebase supports CUDA acceleration for GPU computation when available.

The model architecture was based on the DepRoBERTa framework (`rafalposwiata/deproberta-large-depression`), a RoBERTa-based model pre-trained for depression detection. The base model was fine-tuned for binary classification by replacing the original classification head with a new head configured for two classes. Input text was tokenized using the RoBERTa tokenizer (loaded via AutoTokenizer) with a maximum sequence length of 512 tokens. The contextual embeddings produced by the final transformer layer (dimensionality of 1,024) were passed through a dropout layer (probability 0.1, as configured in the base model) and followed by a linear classification layer that produced logits for two classes (non-depression and depression). These logits were transformed into probabilities using a softmax activation function, representing the estimated probability distribution over the two classes.

Tuning was performed using a batch size of 4 samples per device per iteration. The learning rate was set to 2 × 10⁻⁵, and tuning was conducted for 3 epochs. Mixed-precision FP16 computation was disabled (fp16=False) for CPU training. The model used standard cross-entropy loss (CrossEntropyLoss) for multi-class classification, which compares predicted probability distributions against ground truth binary labels in the set {0, 1} for non-depressed and depressed, respectively. 

Model selection was performed based on validation-set F1 score, with the best checkpoint automatically loaded at the end of training (`load_best_model_at_end=True`). Checkpoints were saved after each epoch, and the model with the highest validation F1 score was retained. Evaluation was performed after each epoch on the validation set to monitor training progress and prevent overfitting.

---

## Model Evaluation and Statistical Analysis

After tuning was complete, the final model checkpoint (selected based on highest validation-set F1 score) was evaluated using the reserved test dataset of 3,419 aggregated packets set aside from earlier. Inference was executed and performance was assessed using accuracy, precision, recall, and F1 score. Given the clinical context, recall for depressed samples (label 1) is particularly important because false negatives (failing to detect depression) are more detrimental than false positives in this context.

The final model achieved the following performance metrics on the test set:
- **Accuracy**: 73.97%
- **F1 Score**: 73.08%
- **Precision**: 74.04%
- **Recall**: 73.97%

To evaluate whether model performance exceeded baseline classification, the fine-tuned model was compared against the default DepRoBERTa model (`rafalposwiata/deproberta-large-depression`) on the same test set. The default model, which outputs 3-class predictions (not depression, moderate, severe), was evaluated in two ways: (1) using its native 3-class output, and (2) mapping its predictions to binary classes by combining "moderate" and "severe" predictions as depression (label 1) and "not depression" as non-depression (label 0).

The fine-tuned model significantly outperformed the default model when mapped to binary classification:
- **Accuracy improvement**: +14.19% (73.97% vs. 59.78%)
- **F1 score improvement**: +23.76% (73.08% vs. 49.32%)
- **Precision improvement**: +16.32% (74.04% vs. 57.73%)
- **Recall improvement**: +14.19% (73.97% vs. 59.78%)

This comparison demonstrates that fine-tuning on the Reddit mental health dataset substantially improved the model's ability to perform binary depression classification compared to the pre-trained baseline.

---

## Notes for Revision

### Corrections Made:
1. **Python version**: Changed from 3.14 to 3.11 (actual version used)
2. **Max sequence length**: Consistent 512 tokens throughout (not 256)
3. **Tokenizer**: RoBERTa tokenizer via AutoTokenizer (not DebertaV3Tokenizer)
4. **Batch size**: 4 per device (not 8 with gradient accumulation to 32)
5. **FP16**: Disabled (fp16=False) for CPU training
6. **Loss function**: CrossEntropyLoss (not Binary Cross Entropy)
7. **Model selection**: Best model based on validation F1 (not early stopping with specific criteria)
8. **Test set size**: 3,419 examples
9. **Split method**: Stratified random sampling, 80:10:10 ratio
10. **Model architecture**: Classification head with 2 outputs and softmax (not single neuron with sigmoid)

### Additional Details Added:
- Specific test set performance metrics
- Comparison methodology with baseline model
- Exact improvement percentages
- Model selection strategy
- Evaluation strategy details

