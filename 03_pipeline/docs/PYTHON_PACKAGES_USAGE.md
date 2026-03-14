# Python Packages Used in Project - Names and Usage Cases

## Standard Library Packages

### `json`
**Usage**: Reading and writing JSON files containing training data, model configurations, results, and metadata
- Loading training/validation/test datasets from JSON files
- Saving model comparison results and metrics
- Storing configuration files and status reports

### `csv`
**Usage**: Writing comparison results and per-example predictions to CSV files for analysis
- Exporting model predictions for detailed analysis
- Creating disagreement analysis tables

### `os`
**Usage**: File system operations, path handling, and environment variable access
- Checking file existence and directory structure
- Path manipulation and file operations

### `sys`
**Usage**: System-specific parameters and functions
- Command-line argument handling
- Script execution control
- Exit codes and error handling

### `subprocess`
**Usage**: Running external processes and training scripts
- Executing training commands in background processes
- Managing training script execution
- Process control and monitoring

### `pathlib` / `Path`
**Usage**: Object-oriented filesystem paths (modern alternative to os.path)
- Cross-platform path handling
- File and directory operations
- Path construction and validation

### `collections` (Counter, defaultdict)
**Usage**: Specialized container datatypes
- **Counter**: Counting label distributions, class frequencies, prediction distributions
- **defaultdict**: Grouping data by keys, creating nested dictionaries for data organization

### `time`
**Usage**: Time-related functions
- Tracking training duration
- Adding delays in scripts
- Timestamp generation

### `datetime`
**Usage**: Date and time manipulation
- Creating timestamps for saved models
- Logging training start/end times
- File naming with timestamps

### `random`
**Usage**: Generating random numbers and shuffling data
- Random sampling for data splits
- Shuffling training data
- Random seed setting for reproducibility

### `math`
**Usage**: Mathematical functions
- Mathematical calculations in utility functions

### `importlib`
**Usage**: Dynamic import of modules
- Loading modules programmatically
- Dynamic script execution

### `shutil`
**Usage**: High-level file operations
- Copying files and directories
- File management operations

## Machine Learning and Deep Learning Packages

### `torch` (PyTorch)
**Usage**: Deep learning framework for model training and inference
- Loading and running transformer models
- GPU acceleration (CUDA) for training
- Tensor operations and model computations
- Model state management

### `transformers` (Hugging Face Transformers)
**Usage**: Pre-trained transformer models and NLP utilities
- **AutoModelForSequenceClassification**: Loading and using classification models (DepRoBERTa)
- **AutoTokenizer**: Tokenizing text input for transformer models
- **AutoModel**: General model loading
- **Trainer**: Training loop management and optimization
- **TrainingArguments**: Configuring training hyperparameters (learning rate, batch size, epochs)
- **EarlyStoppingCallback**: Preventing overfitting during training

### `datasets` (Hugging Face Datasets)
**Usage**: Efficient dataset handling and preprocessing
- **Dataset**: Creating datasets from lists/dictionaries
- **load_dataset**: Loading datasets from disk or Hugging Face Hub
- **DatasetDict**: Managing train/val/test splits
- **concatenate_datasets**: Combining multiple datasets
- Efficient data loading and batching for training

### `sklearn` (scikit-learn)
**Usage**: Machine learning metrics and utilities
- **accuracy_score**: Calculating classification accuracy
- **f1_score**: Calculating F1 score (harmonic mean of precision and recall)
- **precision_score**: Calculating precision (true positives / (true positives + false positives))
- **recall_score**: Calculating recall (true positives / (true positives + false negatives))
- **confusion_matrix**: Creating confusion matrices for error analysis
- **classification_report**: Generating detailed classification reports

### `numpy` (NumPy)
**Usage**: Numerical computing and array operations
- Mathematical calculations for metrics
- Array operations for data processing
- Statistical calculations (standard deviation, error bars)
- Data manipulation and transformation

## Data Analysis Packages

### `pandas` (pd)
**Usage**: Data manipulation and analysis
- Reading CSV files (Reddit Mental Health Dataset)
- Data analysis and exploration
- DataFrame operations for structured data
- Merging and joining datasets

## Visualization Packages

### `matplotlib` / `matplotlib.pyplot` (plt)
**Usage**: Creating plots and visualizations
- Generating bar charts for performance metrics comparison
- Creating confusion matrix heatmaps
- Error bar visualization
- Figure formatting and styling

### `tqdm` / `tqdm.auto`
**Usage**: Progress bars for long-running operations
- Showing training progress
- Progress indicators for data loading
- Progress bars for inference loops
- User feedback during long operations

## Summary by Category

### Core ML/NLP Stack
- **torch**: Deep learning framework
- **transformers**: Pre-trained models and training utilities
- **datasets**: Dataset management
- **sklearn**: Evaluation metrics
- **numpy**: Numerical operations

### Data Processing
- **pandas**: Structured data analysis
- **json**: Data serialization
- **csv**: Tabular data export

### Visualization
- **matplotlib**: Plotting and charts

### Utilities
- **tqdm**: Progress indicators
- **pathlib**: Path handling
- **collections**: Specialized data structures

### System
- **os**, **sys**, **subprocess**: System operations
- **time**, **datetime**: Time handling
- **random**, **math**: Mathematical utilities

## Package Versions (from requirements_notebook.txt)
- matplotlib >= 3.10.0
- numpy >= 2.0.0

## Additional Packages (likely used but not in requirements file)
- torch (PyTorch) - typically 2.3.0+ based on methods section
- transformers - typically 4.42.0+ based on methods section
- datasets - Hugging Face datasets library
- scikit-learn - for metrics
- pandas - for data analysis
- tqdm - for progress bars

