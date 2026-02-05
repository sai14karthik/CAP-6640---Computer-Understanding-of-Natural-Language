# Hallucination Detection and Measurement in Large Language Models

CAP 6640 - Computer Understanding of Natural Language  
**Topic #3**: Benchmarking and Evaluation of NLP Systems

## Overview

This project evaluates hallucination detection and measurement across open-source LLMs on factual question-answering datasets. It compares multiple models (e.g., Phi-2, Mistral, Llama 2) on datasets such as TruthfulQA, WikiQA, and SQuAD 2.0, using accuracy and hallucination rate as main metrics.

## Setup

```bash
cd project
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, Hugging Face `transformers` and `datasets`. For 7B+ models, a GPU with ~8GB+ VRAM is recommended; 8-bit quantization is used by default to reduce memory.

## Project Structure

```
project/
├── src/
│   ├── config.py        # Datasets, models, and hyperparameters
│   ├── load_datasets.py # Load TruthfulQA, WikiQA, SQuAD v2, etc.
│   ├── load_models.py   # Load Hugging Face LLMs
│   ├── evaluate.py      # Metrics and evaluation loop
│   ├── run_experiments.py # Main experiment runner
│   └── utils.py         # Helpers
├── results/             # Output JSON results (created on first run)
├── scripts/
│   ├── test_setup.py    # Test data loading (no GPU)
│   └── analyze_results.py # Aggregate results table/CSV
├── requirements.txt
├── run.py               # Entry point
├── PROJECT_STATUS.md    # Proposal vs implementation checklist
└── README.md
```

## Usage

**Quick run (small subset, 2 models, 2 datasets):**

```bash
python run.py --max_samples 50
```

**Custom models and datasets:**

```bash
python run.py --models phi-2 mistral-7b --datasets truthfulqa wiki_qa --max_samples 100
```

**Few-shot prompting:**

```bash
python run.py --models phi-2 --datasets truthfulqa --prompt_type few_shot --max_samples 50
```

**All models and datasets (slow):**

```bash
python run.py --all --max_samples 100
```

**Options:**

- `--models`: Model keys from config (e.g. `phi-2`, `mistral-7b`, `llama2-7b`)
- `--datasets`: Dataset names (e.g. `truthfulqa`, `wiki_qa`, `squad_v2`)
- `--max_samples`: Max samples per dataset (default 50)
- `--prompt_type`: `zero_shot` or `few_shot`
- `--no_8bit`: Disable 8-bit quantization (needs more VRAM)
- `--output_dir`: Directory for result JSONs (default: `results/`)

## Output

- Per run: `results/<model>_<dataset>_<prompt_type>.json` with accuracy, precision, recall, hallucination rate, and metadata.
- Summary: `results/summary.json` with one row per model–dataset pair.
- **Aggregate table:** `python scripts/analyze_results.py` prints a summary table; `--format csv` for CSV.

## Datasets and Models (config)

- **Datasets:** TruthfulQA, WikiQA, Natural Questions, FEVER, SQuAD 2.0 (subset sizes in `src/config.py`).
- **Models:** Llama 2 7B, Phi-2, Mistral 7B, Gemma 7B, Qwen 7B, Llama 3 8B (Hugging Face IDs in `src/config.py`).

## Metrics

- **Accuracy (contain):** Fraction of answers that contain the ground-truth answer (normalized).
- **Accuracy (exact):** Exact match after normalization.
- **Precision / Recall:** For single-answer QA, both equal accuracy (correct vs incorrect).
- **Hallucination rate:** 1 − accuracy (contain).

## License

For academic use as part of CAP 6640, UCF.
