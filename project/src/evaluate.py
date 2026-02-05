"""
Evaluation metrics and experiment runner for hallucination detection.
"""

from typing import List, Dict, Any, Optional
from .utils import exact_match, contains_answer, normalize_answer
from .config import ZERO_SHOT_TEMPLATE, FEW_SHOT_TEMPLATE, GENERATION_CONFIG
from .load_models import generate_answer


def compute_accuracy(predictions: List[str], references: List[str], match: str = "contain") -> float:
    """
    Compute accuracy: fraction of predictions that match reference.
    match: 'exact' or 'contain'
    """
    if not predictions or not references:
        return 0.0
    n = min(len(predictions), len(references))
    correct = 0
    for i in range(n):
        pred = (predictions[i] or "").strip()
        ref = (references[i] or "").strip()
        if not ref:  # unanswerable or no ground truth
            continue
        if match == "exact":
            correct += exact_match(pred, ref)
        else:
            correct += contains_answer(pred, ref)
    return correct / n if n else 0.0


def compute_hallucination_rate(
    predictions: List[str],
    references: List[str],
    match: str = "contain",
) -> float:
    """
    Hallucination rate = 1 - accuracy (fraction of answers that are incorrect).
    """
    acc = compute_accuracy(predictions, references, match=match)
    return 1.0 - acc


def compute_precision_recall(
    predictions: List[str], references: List[str], match: str = "contain"
) -> tuple:
    """
    For single-answer QA: treat correct (match) as positive class.
    Precision = TP / (TP + FP), Recall = TP / (TP + FN).
    With one prediction per question: precision = recall = accuracy.
    """
    if not predictions or not references:
        return 0.0, 0.0
    n = min(len(predictions), len(references))
    tp = 0
    for i in range(n):
        pred = (predictions[i] or "").strip()
        ref = (references[i] or "").strip()
        if not ref:
            continue
        if match == "exact":
            tp += exact_match(pred, ref)
        else:
            tp += contains_answer(pred, ref)
    # With single prediction per item: TP+FP = n, TP+FN = n, so P = R = TP/n
    return (tp / n, tp / n) if n else (0.0, 0.0)


def run_evaluation(
    model,
    tokenizer,
    dataset: List[Dict[str, str]],
    prompt_type: str = "zero_shot",
    num_few_shot: int = 3,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run model on dataset and compute metrics.
    prompt_type: 'zero_shot' or 'few_shot'
    """
    from tqdm import tqdm

    data = dataset[: max_samples or len(dataset)]
    predictions = []
    references = [d.get("answer", "") for d in data]

    for i, item in enumerate(tqdm(data, desc="Evaluating", disable=not verbose)):
        q = item.get("question", "")
        ref = item.get("answer", "")

        if prompt_type == "few_shot" and num_few_shot > 0 and i >= num_few_shot:
            # Use first num_few_shot as examples
            examples = data[:num_few_shot]
            ex_text = ""
            for ex in examples:
                eq = ex.get("question", "")
                ea = ex.get("answer", "")
                ex_text += f"Question: {eq}\nAnswer: {ea}\n\n"
            prompt = ex_text + ZERO_SHOT_TEMPLATE.format(question=q)
        else:
            prompt = ZERO_SHOT_TEMPLATE.format(question=q)

        pred = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            temperature=GENERATION_CONFIG["temperature"],
            do_sample=GENERATION_CONFIG["do_sample"],
        )
        predictions.append(pred)

    acc_contain = compute_accuracy(predictions, references, match="contain")
    acc_exact = compute_accuracy(predictions, references, match="exact")
    hall_rate = 1.0 - acc_contain
    precision, recall = compute_precision_recall(predictions, references, match="contain")

    return {
        "accuracy_contain": acc_contain,
        "accuracy_exact": acc_exact,
        "precision": precision,
        "recall": recall,
        "hallucination_rate": hall_rate,
        "num_samples": len(data),
        "predictions": predictions,
        "references": references,
    }
