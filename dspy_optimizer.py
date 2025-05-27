import argparse
import os
import time
from typing import Literal
from datetime import datetime

import pandas as pd
import dspy
from sklearn.model_selection import train_test_split

from utils import categories


class ClassificationSignature(dspy.Signature):
    """Classify historical documents from the Finnish National Archives as either 'publid_data' or 'sensitive_data'.
    Base your decision on the GDPR guidelines and administrative archival context.  
    Use the classification category definitions to justify your decision with step-by-step reasoning.
    The document will be in Finnish.
    Always return both a justification and a final label.""" 
    text: str = dspy.InputField(prefix ="Document", desc="The document to classify.")
    categories: str = dspy.InputField(prefix ="Classification Categories", desc="Description of classification categories.")
    reasoning: str = dspy.OutputField(prefix ="Reasoning", desc="Step-by-step reasoning about whether the document fits 'public_data' or 'sensitive_data'.")
    label: Literal['public_data', 'sensitive_data'] = dspy.OutputField(prefix ="Label", desc="Final classification label. Must be either 'public_data' or 'sensitive_data'.")


def parse_arguments():
    """Parse command-line arguments for model name, trainset path, optimization level, thread count, and mode."""
    parser = argparse.ArgumentParser(description="Run DSPy classification optimization.",
                                    epilog="Example: python dspy_optimizer.py --model=llama3.1:8b --trainset=./trainset.csv --auto=heavy --labeled_demos=6")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to use with Ollama')
    parser.add_argument('--trainset', type=str, required=True, help='Path to the training dataset CSV file')
    parser.add_argument('--auto', type=str, choices=['none', 'light', 'medium', 'heavy'], default='light', help='MIPROv2 optimization level')
    parser.add_argument('--labeled_demos', type=int, default=5, help='Number of labeled demos to include in the prompt.')
    return parser.parse_args()


def make_output_dir(model_name, trainset_path, full_df, train_data, val_data, auto_level, labeled_demos):
    """Create results directory and write metadata about the run to a README.md file."""
    timestamp = datetime.now().strftime('%d%m_%H%M')
    model_name_clean = model_name.replace(".", "").replace(":", "_")
    folder_name = f"{model_name_clean}__{timestamp}"
    out_dir = os.path.join("optimizations", folder_name)
    os.makedirs(out_dir, exist_ok=True)
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"--- Dataset Info ---\n")
        f.write(f"Dataset size: {len(full_df)}\n")
        f.write(f"Possible classes: {sorted(full_df['label'].unique())}\n")
        f.write(f"Training set size: {len(train_data)}\n")
        f.write(f"Validation set size: {len(val_data)}\n")
        f.write(f"\n\n--- Optimization run info ---\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Trainset: {trainset_path}\n")
        f.write(f"Run Date: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
        f.write(f"Optimization level: {auto_level}\n")
        f.write(f"Number of labeled demos: {labeled_demos}\n")
        return out_dir, folder_name


def configure_lm(model_name):
    """Initialize and configure the language model."""
    lm = dspy.LM(f'ollama_chat/{model_name}', api_base='http://localhost:11434', api_key='')
    dspy.settings.configure(lm=lm)


def accuracy_metric(example, prediction, trace=None):
    """Compute accuracy for a single example-prediction pair."""
    return prediction.label == example.label


def load_data(trainset_path):
    """Load training dataset and split it into training and validation sets."""
    df = pd.read_csv(trainset_path, sep=',')
    print(f"\nDataset size: {len(df)}")
    print(f"Possible classes: {df['label'].unique()}\n")
    train_data, val_data = train_test_split(df, test_size=0.4, random_state=9, stratify=df['label'])
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}\n")
    return df, train_data, val_data


def convert_to_examples(dataframe):
    """Convert a pandas DataFrame to a list of DSPy Examples."""
    return [
        dspy.Example(
            text=row["text"],
            categories=categories,
            label=row["label"]
        ).with_inputs("text", "categories")
        for _, row in dataframe.iterrows()
    ]

def create_teleprompter(metric, auto_level):
    """Initialize and return a configured MIPROv2 teleprompter for DSPy optimization."""
    return dspy.MIPROv2(
        metric=metric,
        auto=None if auto_level == 'none' else auto_level,
        num_threads=8,
        num_candidates=10
    )


def optimize_program(teleprompter, classifier, trainset, valset, labeled_demos):
    """Optimize a DSPy program using MIPROv2."""
    return teleprompter.compile(
        classifier,
        trainset=trainset,
        valset=valset,
        max_labeled_demos=labeled_demos,
        max_bootstrapped_demos=0, 
        num_trials=30,
        minibatch=True,
        minibatch_size=len(valset),      
        minibatch_full_eval_steps=3,     
        requires_permission_to_run=False
    )


def main():
    """Main function to run DSPy optimization using MIPROv2."""
    start_time = time.time()

    args = parse_arguments()
    configure_lm(args.model)

    full_df, train_data, val_data = load_data(args.trainset)
    output_dir, folder_name = make_output_dir(args.model, args.trainset, full_df, train_data, val_data, args.auto, args.labeled_demos)

    trainset = convert_to_examples(train_data)
    valset = convert_to_examples(val_data)

    cot_classifier = dspy.ChainOfThought(signature=ClassificationSignature)
    teleprompter = create_teleprompter(accuracy_metric, args.auto)
    optimized_program = optimize_program(teleprompter, cot_classifier, trainset, valset, args.labeled_demos)

    # Save program
    optimized_program.save(output_dir, save_program=True)
    print(f"Optimized classifier saved to {output_dir}")

    # Calculate and append runtime to README.md
    total_time_min = (time.time() - start_time) / 60
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n--- Runtime ---\n")
        f.write(f"Total time: {total_time_min:.2f} minutes\n")


if __name__ == '__main__':
    main()
