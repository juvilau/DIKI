import argparse
import json
import os
import time
import contextlib
from typing import Literal
from datetime import datetime

import dspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from utils import categories


class ClassificationSignature(dspy.Signature):
    """Classify historical documents from the Finnish National Archives as either 'publid_data' or 'sensitive_data'.
    Base your decision on the GDPR guidelines and historical context.  
    Use the classification category definitions to justify your decision with step-by-step reasoning.
    The document will be in Finnish.
    Always return both a justification and a final label.""" 
    text: str = dspy.InputField(prefix ="Document", desc="The document to classify.")
    categories: str = dspy.InputField(prefix ="Classification Categories", desc="Description of classification categories.")
    reasoning: str = dspy.OutputField(prefix ="Reasoning", desc="Step-by-step reasoning about whether the document fits 'public_data' or 'sensitive_data'.")
    label: Literal['public_data', 'sensitive_data'] = dspy.OutputField(prefix ="Label", desc="Final classification label.")


class ClassificationPredictSignature(dspy.Signature):
    """Classify historical documents from the Finnish National Archives as either 'publid_data' or 'sensitive_data'.
    Base your decision on the GDPR guidelines and historical context.  
    Use the classification category definitions to justify your decision with step-by-step reasoning.
    The document will be in Finnish.
    Always return both a justification and a final label.""" 
    text: str = dspy.InputField(prefix ="Document", desc="The document to classify.")
    categories: str = dspy.InputField(prefix ="Classification Categories", desc="Description of classification categories.")
    label: Literal['public_data', 'sensitive_data'] = dspy.OutputField(prefix ="Label", desc="Final classification label.")


class ChainOfThoughtClassifier(dspy.ChainOfThought):
    """A simple DSPy classifier using chain-of-thought reasoning."""
    def __init__(self):
        super().__init__(ClassificationSignature)


class PredictClassifier(dspy.Predict):
    """A simple DSPy classifier using direct prediction."""
    def __init__(self):
        super().__init__(ClassificationPredictSignature)


def parse_arguments():
    """Parses command-line arguments for the DSPy classifier."""
    parser = argparse.ArgumentParser(
        description="DSPy optmized classifier.",
        epilog="Example usage: python dspy_classifier.py --model=llama3.1:8b --dataset=./testset.csv --optimized=./optimizations/few-shot/llama31_8b__1005_1912")
    parser.add_argument('--model', type=str, required=True, help='Name of the llm')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--optimized', type=str, required=False, help='Path to the optimized DSPy program')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature of the llm')
    parser.add_argument('--module', type=str, required=False, choices=['cot', 'predict'], help="Classification module: 'cot' or 'predict'.")
    parser.add_argument('--nocache', action='store_true', help='Disable DSPy cache')

    args = parser.parse_args()

    args.cache = not args.nocache

    if not args.module:
        if not args.optimized or not args.optimized.strip():
            parser.error("--optimized is required if --module is not given")

    return args


def make_result_directory(model_name, dataset, optimization, module):
    timestamp = datetime.now().strftime('%d%m%H%M')
    model_name_clean = model_name.replace(".", "").replace(":", "")
    folder_name = f"{model_name_clean}_{timestamp}"
    result_dir = os.path.join("results", folder_name)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"--- Run Info ---\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Module: {module}\n")
        f.write(f"Optimization: {optimization}\n")
        f.write(f"Test Dataset: {dataset}\n")
        f.write(f"Run Date: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
    return result_dir, folder_name


def load_test_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not dataset_path.lower().endswith("eng.csv"):
        raise ValueError(f"Invalid dataset format: {dataset_path}. Dataset must end with 'eng.csv'")
    df = pd.read_csv(dataset_path, sep=',')
    print(f"\nDataset size: {len(df)}")
    print(f"Possible classes: {df['label'].unique()}\n")
    data = df.apply(lambda x: {'text': x['text'], 'label': x['label']}, axis=1).tolist()
    X = [item['text'] for item in data]
    y = [item['label'] for item in data]
    return X, y


def configure_lm(model_name, temperature):
    lm = dspy.LM(f'ollama_chat/{model_name}', api_base='http://localhost:11434', api_key='', temperature=temperature)
    dspy.settings.configure(lm=lm)


def load_classifier(path):
    try:
        return dspy.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} not found.")


def plot_confusion_matrix(cm, labels, save_path, model):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap='Blues')
    #ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=14)
    ax.set_yticklabels(labels, rotation=90, ha='center', fontsize=14)
    ax.tick_params(axis='y', pad=8)
    for label in ax.get_yticklabels():
        label.set_va('center')

    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    fontsize=14,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xlabel('Ennustetut luokat', fontsize=14)
    ax.set_ylabel('Todelliset luokat', fontsize=14)
    ax.set_title(f"{model}", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_dspy_history(path: str, n: int = 1):
    with open(path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            dspy.inspect_history(n=n)


def sklearn_report(X, y, classifier, labels, categories, result_dir, prefix, model):
    y_true, y_pred, output, total_time = run_predictions(X, y, classifier, categories)
    report_str, accuracy, cm = generate_reports(y_true, y_pred, labels)

    print(f"\n{report_str}")
    print(f"Total classification time: {int(total_time // 60)} min {int(total_time % 60)} sec")

    plot_confusion_matrix(cm, labels, os.path.join(result_dir, f"{prefix}_confusion_matrix.png"), model)
    save_reports(report_str, output, y_true, y_pred, cm, result_dir, prefix, model, total_time)


def run_predictions(X, y, classifier, categories):
    y_true, y_pred, output = [], [], []
    start_time = time.time()
    for i, x in tqdm(enumerate(X), total=len(X), desc="Running classifier", unit="sample"):
        try:
            result = classifier(text=x, categories=categories)
            y_pred.append(result.label)
            y_true.append(y[i])
            output.append({
                "text": x,
                "true_label": y[i],
                "predicted_label": result.label,
                "reasoning": getattr(result, 'reasoning', None)
            })
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    total_time = time.time() - start_time
    return y_true, y_pred, output, total_time


def generate_reports(y_true, y_pred, labels):
    y_true = [label.strip() for label in y_true]
    y_pred = [label.strip() for label in y_pred]
    report = classification_report(y_true, y_pred, labels=labels, zero_division=1, digits=4)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return report, acc, cm


def save_reports(report, output, y_true, y_pred, cm, result_dir, prefix, model, total_time):
    avg_time = total_time / len(output) if output else 0

    # Save classification report
    with open(os.path.join(result_dir, f"{prefix}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nTotal classification time: {total_time / 60:.2f} minutes\n")
        f.write(f"Average time per sample: {avg_time:.2f} seconds\n")

    # Save full classification output
    with open(os.path.join(result_dir, f"{prefix}_classified_testset.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Save mismatches (errors)
    mismatches = [entry for entry in output if entry.get('true_label') != entry.get('predicted_label')]
    error_analysis_path = os.path.join(result_dir, f"{prefix}_error-analysis.json")
    with open(error_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(mismatches)} incorrect predictions to {error_analysis_path}\n\n")

    # Save DSPy history
    save_dspy_history(os.path.join(result_dir, f"{prefix}_dspy_history.txt"))

    # Update README with timing info
    readme_path = os.path.join(result_dir, "README.md")
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write(f"\nTotal classification time: {total_time / 60:.2f} minutes\n")
        f.write(f"Average time per sample: {avg_time:.2f} seconds\n")


def main():
    """Main execution function that orchestrates the classification pipeline."""
    args = parse_arguments()

    result_dir, prefix = make_result_directory(args.model, args.dataset, args.optimized, args.module)
    X_test, y_test = load_test_data(args.dataset)
    configure_lm(args.model, args.temperature, args.cache)

    labels = ['public_data', 'sensitive_data']

    if args.module == 'cot':
        classifier = ChainOfThoughtClassifier()

    elif args.module == 'predict':
        classifier = PredictClassifier()

    else:
        classifier = load_classifier(args.optimized)
    
    sklearn_report(X_test, y_test, classifier, labels, categories, result_dir, prefix, args.model)

if __name__ == '__main__':
    main()
