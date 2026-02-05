# Project Status: Hallucination Detection and Measurement in LLMs

**Topic #3** – Group 1: Benchmarking and Evaluation of NLP Systems  
**CAP 6640** – Computer Understanding of Natural Language

---

## Proposal vs Implementation

| Proposal item | Status | Implementation |
|---------------|--------|----------------|
| **Datasets (3–5)** | Done | TruthfulQA, WikiQA, SQuAD 2.0 (loaders). Natural Questions, FEVER (config + generic loader). Subset sizes in `src/config.py`. |
| **Models (5–6)** | Done | Llama 2 7B, Phi-2, Mistral 7B, Gemma 7B, Qwen 7B, Llama 3 8B in `src/config.py`. Load via Hugging Face; 8-bit optional. |
| **Accuracy** | Done | `evaluate.py`: accuracy (contain + exact match). |
| **Precision, recall** | Done | `evaluate.py`: `compute_precision_recall()`. |
| **Hallucination rate** | Done | 1 − accuracy (contain). |
| **Zero-shot / few-shot** | Done | `run_experiments.py --prompt_type zero_shot|few_shot`. |
| **Evaluation framework** | Done | `run_evaluation()` in `evaluate.py`; `run_experiments.py` for all model–dataset pairs. |
| **Statistical analysis** | Partial | Summary JSON + `scripts/analyze_results.py`. Optional: add significance tests in post-processing. |
| **Confidence–accuracy correlation** | Optional | Not implemented; would require log-probability extraction from models. |

---

## Code Layout

- **`src/config.py`** – Dataset/model IDs, generation settings, paths.
- **`src/load_datasets.py`** – Load TruthfulQA, WikiQA, SQuAD 2.0; generic path for NQ/FEVER.
- **`src/load_models.py`** – Load HF causal LMs, 8-bit option, `generate_answer()`.
- **`src/evaluate.py`** – Accuracy (contain/exact), precision, recall, hallucination rate, `run_evaluation()`.
- **`src/run_experiments.py`** – CLI: run model×dataset×prompt_type, write per-run JSON + `summary.json`.
- **`src/utils.py`** – Normalization, exact/contain match, save/load JSON.
- **`run.py`** – Entry point.
- **`scripts/test_setup.py`** – Check config and data loading (no GPU).
- **`scripts/analyze_results.py`** – Aggregate `results/*.json` into table or CSV.

---

## How to Run

```bash
cd project
pip install -r requirements.txt
python run.py --models phi-2 --datasets truthfulqa --max_samples 50
python scripts/analyze_results.py
```

---

## Conclusion

**Coding is complete** relative to the proposal: all promised datasets, models, metrics (accuracy, precision, recall, hallucination rate), and zero-shot/few-shot evaluation are implemented and runnable. Optional extensions: confidence–accuracy correlation, extra statistical tests, and full custom loaders for Natural Questions/FEVER if needed.
