"""
Augmentation effectiveness evaluation.

This module provides the "gold standard" evaluation for augmentation:
comparing test set perplexity after finetuning on original vs augmented data.

If finetuning on augmented data leads to lower test perplexity than
finetuning on original data, then augmentation improved generalization.

Note: This is expensive as it requires training two models.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.diversity.perplexity import compute_perplexity_profile

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


def evaluate_augmentation_effectiveness(
    original_train: Dataset,
    augmented_train: Dataset,
    test_set: Dataset,
    model_name: str,
    text_column: str = "text",
    output_dir: str | Path | None = None,
    finetune_config: dict[str, Any] | None = None,
    max_test_samples: int | None = 200,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Evaluate augmentation effectiveness by comparing test perplexity.

    This is the gold-standard evaluation:
    1. Compute base model perplexity on test set
    2. Finetune on original data → compute test perplexity
    3. Finetune on augmented data → compute test perplexity
    4. Compare: if augmented training → lower test perplexity, augmentation helped

    CAUTION: This is expensive! It requires training two models.

    Args:
        original_train: Original training dataset.
        augmented_train: Augmented training dataset.
        test_set: Test dataset for evaluation.
        model_name: Base model to finetune.
        text_column: Column containing text data.
        output_dir: Directory to save models and results.
        finetune_config: Optional finetuning configuration override.
        max_test_samples: Max test samples for perplexity computation.
        seed: Random seed.
        show_progress: Whether to show progress.

    Returns:
        Dictionary with effectiveness evaluation:
        {
            "base_model_test_ppl": float,
            "original_trained_test_ppl": float,
            "augmented_trained_test_ppl": float,
            "original_improvement_pct": float,
            "augmented_improvement_pct": float,
            "augmentation_delta_pct": float,
            "augmentation_effective": bool,
            "summary": str,
        }
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="effectiveness_eval_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Effectiveness evaluation output directory: {output_dir}")

    # Extract texts
    test_texts = test_set[text_column]
    if max_test_samples and len(test_texts) > max_test_samples:
        import random

        random.seed(seed)
        test_texts = random.sample(list(test_texts), max_test_samples)

    # Default finetune config
    if finetune_config is None:
        finetune_config = {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "max_seq_length": 4096,
            "lora_r": 16,
            "lora_alpha": 32,
        }

    # Step 1: Base model perplexity on test set
    logger.info("Step 1/3: Computing base model perplexity on test set...")
    base_ppl_result = compute_perplexity_profile(
        test_texts,
        model_name=model_name,
        show_progress=show_progress,
    )
    base_ppl = base_ppl_result["mean_perplexity"]
    logger.info(f"Base model test perplexity: {base_ppl:.2f}")

    # Step 2: Finetune on original, evaluate
    logger.info("Step 2/3: Finetuning on original dataset...")
    original_model_path = output_dir / "model_original"
    _finetune_model(
        original_train,
        model_name,
        original_model_path,
        text_column,
        finetune_config,
    )

    original_ppl_result = compute_perplexity_profile(
        test_texts,
        model_name=str(original_model_path),
        show_progress=show_progress,
    )
    original_ppl = original_ppl_result["mean_perplexity"]
    logger.info(f"Original-trained test perplexity: {original_ppl:.2f}")

    # Step 3: Finetune on augmented, evaluate
    logger.info("Step 3/3: Finetuning on augmented dataset...")
    augmented_model_path = output_dir / "model_augmented"
    _finetune_model(
        augmented_train,
        model_name,
        augmented_model_path,
        text_column,
        finetune_config,
    )

    augmented_ppl_result = compute_perplexity_profile(
        test_texts,
        model_name=str(augmented_model_path),
        show_progress=show_progress,
    )
    augmented_ppl = augmented_ppl_result["mean_perplexity"]
    logger.info(f"Augmented-trained test perplexity: {augmented_ppl:.2f}")

    # Compute improvements
    original_improvement = (base_ppl - original_ppl) / base_ppl * 100
    augmented_improvement = (base_ppl - augmented_ppl) / base_ppl * 100
    augmentation_delta = augmented_improvement - original_improvement

    # Determine if augmentation was effective
    augmentation_effective = augmented_ppl < original_ppl

    # Generate summary
    summary_lines = [
        f"Base model test perplexity: {base_ppl:.2f}",
        f"After training on original ({len(original_train)} samples): {original_ppl:.2f} ({original_improvement:+.1f}%)",
        f"After training on augmented ({len(augmented_train)} samples): {augmented_ppl:.2f} ({augmented_improvement:+.1f}%)",
        "",
        f"Augmentation delta: {augmentation_delta:+.1f}%",
        f"Augmentation effective: {'YES' if augmentation_effective else 'NO'}",
    ]

    if augmentation_effective:
        ppl_reduction = (original_ppl - augmented_ppl) / original_ppl * 100
        summary_lines.append(
            f"The augmented model has {ppl_reduction:.1f}% lower test perplexity than the original-trained model."
        )
    else:
        summary_lines.append("Augmentation did not improve generalization. Consider different augmentation strategies.")

    return {
        "base_model_test_ppl": base_ppl,
        "original_trained_test_ppl": original_ppl,
        "augmented_trained_test_ppl": augmented_ppl,
        "original_improvement_pct": original_improvement,
        "augmented_improvement_pct": augmented_improvement,
        "augmentation_delta_pct": augmentation_delta,
        "augmentation_effective": augmentation_effective,
        "test_samples": len(test_texts),
        "original_train_samples": len(original_train),
        "augmented_train_samples": len(augmented_train),
        "model_name": model_name,
        "output_dir": str(output_dir),
        "summary": "\n".join(summary_lines),
    }


def _finetune_model(
    dataset: Dataset,
    base_model: str,
    output_path: Path,
    text_column: str,
    config: dict[str, Any],
) -> None:
    """
    Finetune a model on a dataset using the existing finetuning infrastructure.

    This is a simplified wrapper around the finetuning module.
    """
    try:
        from src.finetuning.model_finetuning import finetune_model
    except ImportError as e:
        raise ImportError(
            "Finetuning module not available. Effectiveness evaluation requires the finetuning infrastructure."
        ) from e

    # Save dataset to temp location
    temp_dataset_path = output_path.parent / f"temp_dataset_{output_path.name}"
    dataset.save_to_disk(str(temp_dataset_path))

    # Run finetuning
    finetune_model(
        model_name=base_model,
        dataset_path=str(temp_dataset_path),
        output_dir=str(output_path),
        num_train_epochs=config.get("num_train_epochs", 1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        max_seq_length=config.get("max_seq_length", 4096),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
    )


def quick_effectiveness_estimate(
    original_texts: list[str],
    augmented_texts: list[str],
    model_name: str,
    sample_size: int = 500,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Quick estimate of augmentation effectiveness without finetuning.

    This computes perplexity profiles for both datasets and compares them.
    A higher perplexity on augmented data suggests it contains more
    "novel" information that the model would need to learn.

    This is NOT as definitive as full effectiveness evaluation, but is
    much faster and can provide useful insights.

    Args:
        original_texts: Original dataset texts.
        augmented_texts: Augmented dataset texts.
        model_name: Model for perplexity computation.
        sample_size: Sample size for large datasets.
        seed: Random seed.
        show_progress: Whether to show progress.

    Returns:
        Dictionary with quick estimate:
        {
            "original_mean_ppl": float,
            "augmented_mean_ppl": float,
            "novelty_score": float,  # augmented_ppl / original_ppl
            "recommendation": str,
        }
    """
    logger.info("Computing quick effectiveness estimate...")

    original_profile = compute_perplexity_profile(
        original_texts,
        model_name=model_name,
        sample_size=sample_size,
        seed=seed,
        show_progress=show_progress,
    )

    augmented_profile = compute_perplexity_profile(
        augmented_texts,
        model_name=model_name,
        sample_size=sample_size,
        seed=seed,
        show_progress=show_progress,
    )

    original_ppl = original_profile["mean_perplexity"]
    augmented_ppl = augmented_profile["mean_perplexity"]

    # Novelty score: how much more "surprising" is the augmented data?
    novelty_score = augmented_ppl / original_ppl if original_ppl > 0 else 1.0

    # Generate recommendation
    if novelty_score > 1.2:
        recommendation = (
            "Strong novelty signal. Augmented data is significantly more challenging "
            "for the model. Likely to improve generalization."
        )
    elif novelty_score > 1.05:
        recommendation = "Moderate novelty signal. Augmented data adds some new patterns. May improve generalization."
    else:
        recommendation = (
            "Weak novelty signal. Augmented data is similar to original. Consider stronger augmentation strategies."
        )

    return {
        "original_mean_ppl": original_ppl,
        "augmented_mean_ppl": augmented_ppl,
        "novelty_score": novelty_score,
        "novelty_increase_pct": (novelty_score - 1) * 100,
        "original_profile": original_profile,
        "augmented_profile": augmented_profile,
        "recommendation": recommendation,
    }
