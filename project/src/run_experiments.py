"""
Main script: run hallucination detection experiments across models and datasets.
Usage:
  python -m src.run_experiments --models phi-2 --datasets truthfulqa --max_samples 50
  python -m src.run_experiments --all --max_samples 100
"""

import argparse
import os
from datetime import datetime

from .config import (
    DEFAULT_MODELS,
    DEFAULT_DATASETS,
    MODELS,
    DATASETS,
    RESULTS_DIR,
)
from .load_datasets import load_dataset_by_name, get_all_dataset_names
from .load_models import load_model_and_tokenizer
from .evaluate import run_evaluation, compute_accuracy, compute_hallucination_rate
from .utils import ensure_dir, save_json


def run_single_experiment(
    model_key: str,
    dataset_name: str,
    max_samples: int = 100,
    prompt_type: str = "zero_shot",
    use_8bit: bool = True,
    output_dir: str = RESULTS_DIR,
    verbose: bool = True,
) -> dict:
    """Run one model on one dataset and save results."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Model: {model_key} | Dataset: {dataset_name} | n={max_samples}")
        print("="*60)

    # Load dataset
    data = load_dataset_by_name(dataset_name, max_samples=max_samples)
    if not data:
        return {"error": f"No data loaded for {dataset_name}"}

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_key, use_8bit=use_8bit)

    # Evaluate
    results = run_evaluation(
        model, tokenizer, data,
        prompt_type=prompt_type,
        max_samples=max_samples,
        verbose=verbose,
    )

    # Add metadata
    results["model"] = model_key
    results["dataset"] = dataset_name
    results["prompt_type"] = prompt_type
    results["timestamp"] = datetime.now().isoformat()

    # Save
    out_dir = ensure_dir(output_dir)
    fname = f"{model_key}_{dataset_name}_{prompt_type}.json"
    out_path = os.path.join(str(out_dir), fname)
    save_json(
        {k: v for k, v in results.items() if k not in ("predictions", "references")},
        str(out_path),
    )
    if verbose:
        print(f"Accuracy (contain): {results['accuracy_contain']:.4f}")
        print(f"Hallucination rate: {results['hallucination_rate']:.4f}")
        print(f"Saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Experiments")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model keys")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Dataset names")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples per dataset")
    parser.add_argument("--prompt_type", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--no_8bit", action="store_true", help="Disable 8-bit quantization")
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    parser.add_argument("--all", action="store_true", help="Use all models and datasets (slower)")
    args = parser.parse_args()

    if args.all:
        model_list = list(MODELS.keys())
        dataset_list = get_all_dataset_names()
    else:
        model_list = args.models
        dataset_list = args.datasets

    all_results = []
    for model_key in model_list:
        for dataset_name in dataset_list:
            if dataset_name not in DATASETS:
                print(f"Skipping unknown dataset: {dataset_name}")
                continue
            try:
                res = run_single_experiment(
                    model_key=model_key,
                    dataset_name=dataset_name,
                    max_samples=args.max_samples,
                    prompt_type=args.prompt_type,
                    use_8bit=not args.no_8bit,
                    output_dir=args.output_dir,
                )
                all_results.append(res)
            except Exception as e:
                print(f"Error {model_key} x {dataset_name}: {e}")
                all_results.append({"model": model_key, "dataset": dataset_name, "error": str(e)})

    # Summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    save_json(
        [{"model": r.get("model"), "dataset": r.get("dataset"), "accuracy_contain": r.get("accuracy_contain"), "hallucination_rate": r.get("hallucination_rate"), "error": r.get("error")} for r in all_results],
        summary_path,
    )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
