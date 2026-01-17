"""
Model robustness evaluation for SFT.

This module provides functions to evaluate finetuned models by comparing
their perplexity on completions against a baseline model, and measuring
generalization via train/test performance gaps.

Key metrics:
- Completion perplexity improvement: How much better is the finetuned model?
- Generalization gap: How much worse is test vs train performance?
- Robustness ratio: Relative gap compared to baseline model
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.diversity.perplexity import (
    compute_perplexity_profile,
)
from src.diversity.utils import (
    format_delta,
    get_device,
    sample_texts,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def evaluate_model_robustness(
    texts: list[str],
    finetuned_model: str,
    baseline_model: str | None = None,
    eval_mode: str = "completion",
    sample_size: int | None = None,
    seed: int = 42,
    device: str = "auto",
    show_progress: bool = True,
    apply_chat_template: bool = True,
) -> dict[str, Any]:
    """
    Evaluate model robustness on a dataset.

    Computes perplexity for the finetuned model and optionally compares
    against a baseline model to measure improvement.

    Args:
        texts: List of formatted text samples.
        finetuned_model: Path to finetuned model.
        baseline_model: Path to baseline model (optional).
        eval_mode: "completion" for completion-only, "full" for full sample.
        sample_size: Number of samples to evaluate.
        seed: Random seed.
        device: Device to use.
        show_progress: Whether to show progress.
        apply_chat_template: If True (default), convert IRCA-formatted text to
                            the model's native chat template before computing
                            perplexity.

    Returns:
        Dictionary with evaluation results:
        {
            "finetuned": { perplexity metrics },
            "baseline": { perplexity metrics } (if baseline provided),
            "improvement": {
                "mean_perplexity_change": str,  # e.g., "-67.2%"
                ...
            },
            "summary": str,
        }
    """
    if device == "auto":
        device = get_device()

    result: dict[str, Any] = {
        "eval_mode": eval_mode,
        "sample_count": len(texts),
    }

    # Sample if needed
    if sample_size is not None and len(texts) > sample_size:
        texts = sample_texts(texts, sample_size, seed)
        logger.info(f"Sampled {sample_size} texts for evaluation")
        result["sample_count"] = sample_size

    # Evaluate finetuned model
    logger.info(f"Evaluating finetuned model: {finetuned_model}")
    result["finetuned"] = compute_perplexity_profile(
        texts,
        model_name=finetuned_model,
        device=device,
        show_progress=show_progress,
        eval_mode=eval_mode,
        use_cache=False,  # Don't cache for evaluation
        apply_chat_template=apply_chat_template,
    )

    # Evaluate baseline model if provided
    if baseline_model:
        logger.info(f"Evaluating baseline model: {baseline_model}")
        result["baseline"] = compute_perplexity_profile(
            texts,
            model_name=baseline_model,
            device=device,
            show_progress=show_progress,
            eval_mode=eval_mode,
            use_cache=False,
            apply_chat_template=apply_chat_template,
        )

        # Compute improvement metrics
        ft_ppl = result["finetuned"]["mean_perplexity"]
        base_ppl = result["baseline"]["mean_perplexity"]
        result["improvement"] = {
            "mean_perplexity_change": format_delta(base_ppl, ft_ppl),
            "absolute_change": ft_ppl - base_ppl,
            "relative_change": (ft_ppl - base_ppl) / base_ppl if base_ppl > 0 else 0,
        }

    # Generate summary
    result["summary"] = _generate_robustness_summary(result, baseline_model is not None)

    return result


def evaluate_generalization(
    train_texts: list[str],
    test_texts: list[str],
    finetuned_model: str,
    baseline_model: str | None = None,
    eval_mode: str = "completion",
    sample_size: int | None = None,
    seed: int = 42,
    device: str = "auto",
    show_progress: bool = True,
    apply_chat_template: bool = True,
) -> dict[str, Any]:
    """
    Evaluate generalization by comparing train vs test performance.

    This is the primary function for detecting overfitting. It computes
    perplexity on both train and test sets, and measures the gap.

    Args:
        train_texts: Training set texts.
        test_texts: Held-out test set texts.
        finetuned_model: Path to finetuned model.
        baseline_model: Path to baseline model (optional).
        eval_mode: "completion" for completion-only, "full" for full sample.
        sample_size: Number of samples per set.
        seed: Random seed.
        device: Device to use.
        show_progress: Whether to show progress.
        apply_chat_template: If True (default), convert IRCA-formatted text to
                            the model's native chat template before computing
                            perplexity.

    Returns:
        Dictionary with generalization metrics:
        {
            "finetuned": {
                "train": { perplexity metrics },
                "test": { perplexity metrics },
                "gap": {
                    "absolute": float,  # test_ppl - train_ppl
                    "relative": str,    # e.g., "+28.5%"
                }
            },
            "baseline": { ... } (if provided),
            "comparison": {
                "train_improvement": str,  # finetuned vs baseline on train
                "test_improvement": str,   # finetuned vs baseline on test
                "gap_ratio": float,        # finetuned_gap / baseline_gap
            },
            "summary": str,
        }
    """
    if device == "auto":
        device = get_device()

    result: dict[str, Any] = {
        "eval_mode": eval_mode,
        "train_count": len(train_texts),
        "test_count": len(test_texts),
    }

    # Sample if needed
    if sample_size is not None:
        if len(train_texts) > sample_size:
            train_texts = sample_texts(train_texts, sample_size, seed)
            result["train_count"] = sample_size
        if len(test_texts) > sample_size:
            test_texts = sample_texts(test_texts, sample_size, seed)
            result["test_count"] = sample_size

    # Evaluate finetuned model on both sets
    logger.info("Evaluating finetuned model on train set...")
    ft_train = compute_perplexity_profile(
        train_texts,
        model_name=finetuned_model,
        device=device,
        show_progress=show_progress,
        eval_mode=eval_mode,
        use_cache=False,
        apply_chat_template=apply_chat_template,
    )

    logger.info("Evaluating finetuned model on test set...")
    ft_test = compute_perplexity_profile(
        test_texts,
        model_name=finetuned_model,
        device=device,
        show_progress=show_progress,
        eval_mode=eval_mode,
        use_cache=False,
        apply_chat_template=apply_chat_template,
    )

    ft_gap_abs = ft_test["mean_perplexity"] - ft_train["mean_perplexity"]
    ft_gap_rel = ft_gap_abs / ft_train["mean_perplexity"] if ft_train["mean_perplexity"] > 0 else 0

    result["finetuned"] = {
        "train": ft_train,
        "test": ft_test,
        "gap": {
            "absolute": ft_gap_abs,
            "relative": format_delta(ft_train["mean_perplexity"], ft_test["mean_perplexity"]),
        },
    }

    # Evaluate baseline model if provided
    if baseline_model:
        logger.info("Evaluating baseline model on train set...")
        base_train = compute_perplexity_profile(
            train_texts,
            model_name=baseline_model,
            device=device,
            show_progress=show_progress,
            eval_mode=eval_mode,
            use_cache=False,
            apply_chat_template=apply_chat_template,
        )

        logger.info("Evaluating baseline model on test set...")
        base_test = compute_perplexity_profile(
            test_texts,
            model_name=baseline_model,
            device=device,
            show_progress=show_progress,
            eval_mode=eval_mode,
            use_cache=False,
            apply_chat_template=apply_chat_template,
        )

        base_gap_abs = base_test["mean_perplexity"] - base_train["mean_perplexity"]
        base_gap_rel = base_gap_abs / base_train["mean_perplexity"] if base_train["mean_perplexity"] > 0 else 0

        result["baseline"] = {
            "train": base_train,
            "test": base_test,
            "gap": {
                "absolute": base_gap_abs,
                "relative": format_delta(base_train["mean_perplexity"], base_test["mean_perplexity"]),
            },
        }

        # Compute comparison metrics
        result["comparison"] = {
            "train_improvement": format_delta(
                base_train["mean_perplexity"],
                ft_train["mean_perplexity"],
            ),
            "test_improvement": format_delta(
                base_test["mean_perplexity"],
                ft_test["mean_perplexity"],
            ),
            "gap_ratio": ft_gap_rel / base_gap_rel if base_gap_rel > 0 else float("inf"),
        }

    # Generate summary
    result["summary"] = _generate_generalization_summary(result, baseline_model is not None)

    return result


def _generate_robustness_summary(result: dict[str, Any], has_baseline: bool) -> str:
    """Generate a human-readable summary of robustness evaluation."""
    lines = []

    ft_ppl = result["finetuned"]["mean_perplexity"]
    eval_mode = result.get("eval_mode", "full")
    mode_label = "Completion" if eval_mode == "completion" else "Full sample"

    lines.append(f"{mode_label} Perplexity Evaluation")
    lines.append(f"  Finetuned model: {ft_ppl:.2f}")

    if has_baseline:
        base_ppl = result["baseline"]["mean_perplexity"]
        change = result["improvement"]["mean_perplexity_change"]
        lines.append(f"  Baseline model: {base_ppl:.2f}")
        lines.append(f"  Improvement: {change}")

        rel_change = result["improvement"]["relative_change"]
        if rel_change < -0.5:
            lines.append("  Interpretation: Excellent improvement (>50% reduction)")
        elif rel_change < -0.2:
            lines.append("  Interpretation: Good improvement (20-50% reduction)")
        elif rel_change < 0:
            lines.append("  Interpretation: Modest improvement (<20% reduction)")
        else:
            lines.append("  Interpretation: No improvement - check training")

    return "\n".join(lines)


def _generate_generalization_summary(result: dict[str, Any], has_baseline: bool) -> str:
    """Generate a human-readable summary of generalization evaluation."""
    lines = []

    eval_mode = result.get("eval_mode", "full")
    mode_label = "Completion" if eval_mode == "completion" else "Full sample"
    lines.append(f"{mode_label} Generalization Evaluation")
    lines.append("")

    # Finetuned model summary
    ft = result["finetuned"]
    lines.append("Finetuned Model:")
    lines.append(f"  Train perplexity: {ft['train']['mean_perplexity']:.2f}")
    lines.append(f"  Test perplexity:  {ft['test']['mean_perplexity']:.2f}")
    lines.append(f"  Gap: {ft['gap']['relative']}")

    if has_baseline:
        lines.append("")
        base = result["baseline"]
        lines.append("Baseline Model:")
        lines.append(f"  Train perplexity: {base['train']['mean_perplexity']:.2f}")
        lines.append(f"  Test perplexity:  {base['test']['mean_perplexity']:.2f}")
        lines.append(f"  Gap: {base['gap']['relative']}")

        lines.append("")
        comp = result["comparison"]
        lines.append("Comparison:")
        lines.append(f"  Train improvement: {comp['train_improvement']}")
        lines.append(f"  Test improvement:  {comp['test_improvement']}")

        gap_ratio = comp["gap_ratio"]
        if gap_ratio > 2.0:
            lines.append(f"  Warning: Generalization gap increased significantly ({gap_ratio:.1f}x baseline)")
            lines.append("  Interpretation: Model may be overfitting")
        elif gap_ratio > 1.2:
            lines.append(f"  Note: Generalization gap slightly increased ({gap_ratio:.1f}x baseline)")
            lines.append("  Interpretation: Some overfitting, but acceptable")
        else:
            lines.append(f"  Generalization gap ratio: {gap_ratio:.1f}x baseline")
            lines.append("  Interpretation: Good generalization")

    return "\n".join(lines)


def compare_models_on_dataset(
    texts: list[str],
    models: list[str],
    eval_mode: str = "completion",
    sample_size: int | None = None,
    seed: int = 42,
    device: str = "auto",
    show_progress: bool = True,
    apply_chat_template: bool = True,
) -> dict[str, Any]:
    """
    Compare multiple models on the same dataset.

    Useful for comparing different finetuning configurations or
    checkpoint selections.

    Args:
        texts: List of text samples.
        models: List of model paths to compare.
        eval_mode: "completion" or "full".
        sample_size: Sample size for evaluation.
        seed: Random seed.
        device: Device to use.
        show_progress: Whether to show progress.
        apply_chat_template: If True (default), convert IRCA-formatted text to
                            the model's native chat template before computing
                            perplexity.

    Returns:
        Dictionary with per-model metrics and ranking.
    """
    if device == "auto":
        device = get_device()

    # Sample once for consistent comparison
    if sample_size is not None and len(texts) > sample_size:
        texts = sample_texts(texts, sample_size, seed)

    results: dict[str, Any] = {
        "eval_mode": eval_mode,
        "sample_count": len(texts),
        "models": {},
    }

    for model_path in models:
        logger.info(f"Evaluating model: {model_path}")
        profile = compute_perplexity_profile(
            texts,
            model_name=model_path,
            device=device,
            show_progress=show_progress,
            eval_mode=eval_mode,
            use_cache=False,
            apply_chat_template=apply_chat_template,
        )
        results["models"][model_path] = profile

    # Rank models by mean perplexity (lower is better)
    ranked = sorted(
        results["models"].items(),
        key=lambda x: x[1]["mean_perplexity"],
    )
    results["ranking"] = [{"model": model, "mean_perplexity": metrics["mean_perplexity"]} for model, metrics in ranked]

    # Generate summary
    lines = ["Model Comparison (ranked by perplexity, lower is better):"]
    for i, item in enumerate(results["ranking"], 1):
        lines.append(f"  {i}. {item['model']}: {item['mean_perplexity']:.2f}")
    results["summary"] = "\n".join(lines)

    return results
