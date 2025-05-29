# DSPy-based Sensitive Data Classifier for Finnish Archival Documents

## Overview

A DSPy-based Sensitive Data Classifier for Finnish Archival Documents.
The classifier is tailored for two categories:
- **sensitive_data**: Contains personal data or special category data (as defined by the GDPR).
- **public_data**: Contains no personal data, only public or anonymized information.

## Repository Contents

- `dspy_classifier.py`:  
  Actual classifier. Supports both chain-of-thought and direct prediction modes. 
- `dspy_optimizer.py`:  
  Optimizes the classifier with MIPROv2. Saves the optimized program for later inference.
- `utils.py`:  
  Contains the definitions for classification categories and category instructions.

## Requirements

- dspy==2.6.14
- dspy_ai==2.6.2
- matplotlib==3.10.3
- numpy==2.2.6
- pandas==2.2.3
- scikit_learn==1.6.1
- tqdm==4.67.1

```bash
pip install requirements.txt

```

## Example workflow

```bash
# Run classification with direct prediction
python dspy_classifier.py --model=llama3.1:8b --dataset=./testset.csv --module=predict --temperature=0.0

# Run classification with CoT
python dspy_classifier.py --model=llama3.1:8b --dataset=./testset.csv --module=cot --temperature=0.0 

# Optimize the classifier
python dspy_optimizer.py --model=llama3.1:8b --trainset=./trainset.csv --auto=heavy --labeled_demos=5

# Run classification with optimized program
python dspy_classifier.py --model=llama3.1:8b --dataset=./testset.csv --optimized=./optimizations/llama31_8b__ddmm_hhmm
```

## Output

Classification results are saved in `./results/`:

`*_classification_report.txt`: Classification metrics (F1, accuracy, etc.)

`*_classified_testset.json`: Full predictions

`*_error-analysis.json`: Misclassified samples

`*_confusion_matrix.png`: Confusion matrix image

`*_dspy_history.txt`: DSPy execution trace

## Data Format
Both training and test sets should be in CSV format with the following columns:

`text`: The document text to classify

`label`: Either `sensitive_data` or `public_data`

```csv
text,label
"Sample archival document text",sensitive_data
```
