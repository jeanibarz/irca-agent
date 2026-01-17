"""
Perplexity-based diversity metrics for SFT evaluation.

These metrics measure how "surprising" a dataset is to a specific model,
capturing both semantic and syntactic diversity. They are the primary
recommended metrics for evaluating augmentation effectiveness in SFT.

Key insight: Perplexity measures what we actually care about - will the
model need to learn something new from this data?

Metrics:
- Perplexity Profile: Statistics about model perplexity on a dataset
- Dataset Comparison: Before/after comparison of diversity metrics
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from src.diversity.lexical import compute_lexical_diversity
from src.diversity.utils import (
    COMPLETION_MARKERS,
    PerplexityCache,
    convert_irca_to_chat_template,
    find_completion_token_boundary,
    format_delta,
    get_chat_template_completion_markers,
    get_device,
    load_model_and_tokenizer,
    sample_texts,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def compute_sample_perplexity(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int | None = None,
) -> float:
    """
    Compute perplexity for a single text sample.

    Perplexity = exp(cross-entropy loss)
    Lower perplexity = model finds the text more predictable.

    Args:
        text: Text to compute perplexity for.
        model: The language model.
        tokenizer: The tokenizer.
        max_length: Maximum sequence length (truncate if longer).
                    If None, uses model's max position embeddings.

    Returns:
        Perplexity value.
    """
    import torch

    if not text.strip():
        return float("inf")

    # Determine max length from model config if not specified
    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 1024)

    # Tokenize with truncation to model's max length
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute loss
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()

    # Handle edge cases
    if np.isnan(perplexity) or np.isinf(perplexity):
        return float("inf")

    return perplexity


def compute_completion_perplexity(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int | None = None,
    completion_markers: list[str] | None = None,
) -> float:
    """
    Compute perplexity for the completion part only.

    The full text is tokenized to provide context, but loss is computed
    only on the completion tokens. This is the recommended metric for
    evaluating SFT models.

    Args:
        text: Full text including prompt and completion.
        model: The language model.
        tokenizer: The tokenizer.
        max_length: Maximum sequence length. If None, uses model's max.
        completion_markers: Markers separating prompt from completion.

    Returns:
        Perplexity computed on completion tokens only.

    Example:
        >>> text = "### USER\\nHello\\n### ASSISTANT\\nHi there!"
        >>> ppl = compute_completion_perplexity(text, model, tokenizer)
        >>> # Only "Hi there!" contributes to perplexity
    """
    import torch

    if not text.strip():
        return float("inf")

    # Find completion boundary
    prompt_len = find_completion_token_boundary(text, tokenizer, completion_markers or COMPLETION_MARKERS)

    # If no marker found, fall back to full perplexity
    if prompt_len == 0:
        logger.debug("No completion marker found, falling back to full perplexity")
        return compute_sample_perplexity(text, model, tokenizer, max_length)

    # Determine max length from model config if not specified
    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 1024)

    # Tokenize full text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    # Move to model device
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    total_length = input_ids.shape[1]

    # Check if completion is truncated away
    if prompt_len >= total_length:
        logger.warning(f"Completion truncated: prompt_len={prompt_len}, total_length={total_length}")
        return float("inf")

    # Create labels: -100 for prompt tokens (ignored in loss), actual ids for completion
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100  # Mask prompt tokens

    # Compute loss on completion tokens only
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()

    # Handle edge cases
    if np.isnan(perplexity) or np.isinf(perplexity):
        return float("inf")

    return perplexity


def compute_perplexity_profile(
    texts: list[str],
    model_name: str,
    batch_size: int = 1,
    sample_size: int | None = None,
    seed: int = 42,
    max_length: int | None = None,
    device: str = "auto",
    show_progress: bool = True,
    use_cache: bool = True,
    cache: PerplexityCache | None = None,
    eval_mode: str = "full",
    completion_markers: list[str] | None = None,
    apply_chat_template: bool = True,
    return_perplexities: bool = False,
) -> dict[str, Any]:
    """
    Compute perplexity statistics for a dataset.

    This is the primary metric for measuring dataset diversity for SFT.
    Higher mean perplexity = more challenging/diverse for the model.
    Higher std = more variation in difficulty across samples.

    Args:
        texts: List of text samples to analyze.
        model_name: HuggingFace model name or path.
        batch_size: Batch size for inference (currently only 1 supported).
        sample_size: If set, sample this many texts for efficiency.
        seed: Random seed for sampling.
        max_length: Maximum sequence length.
        device: Device to use ("auto", "cuda", "mps", "cpu").
        show_progress: Whether to show progress bar.
        use_cache: Whether to use perplexity cache.
        cache: Optional PerplexityCache instance.
        eval_mode: "full" for full sample perplexity, "completion" for
                   completion-only perplexity (recommended for SFT eval).
        completion_markers: Markers separating prompt from completion.
                           Only used when eval_mode="completion".
        apply_chat_template: If True (default), convert IRCA-formatted text to
                            the model's native chat template before computing
                            perplexity. This ensures evaluation matches training
                            format. Set to False only for raw text evaluation.
        return_perplexities: If True, include raw perplexity values in output
                            for bootstrap CI computation.

    Returns:
        Dictionary with perplexity statistics:
        {
            "mean_perplexity": float,
            "std_perplexity": float,
            "min_perplexity": float,
            "max_perplexity": float,
            "median_perplexity": float,
            "p90_perplexity": float,
            "p95_perplexity": float,
            "sample_count": int,
            "model": str,
            "device": str,
            "eval_mode": str,
            "apply_chat_template": bool,
        }

    Example:
        >>> texts = ["Hello, how are you?", "The weather is nice today."]
        >>> profile = compute_perplexity_profile(texts, "gpt2")
        >>> print(f"Mean perplexity: {profile['mean_perplexity']:.2f}")

        # Completion-only mode for SFT evaluation
        >>> profile = compute_perplexity_profile(texts, "gpt2", eval_mode="completion")
    """
    if not texts:
        return {
            "mean_perplexity": 0.0,
            "std_perplexity": 0.0,
            "min_perplexity": 0.0,
            "max_perplexity": 0.0,
            "median_perplexity": 0.0,
            "p90_perplexity": 0.0,
            "p95_perplexity": 0.0,
            "sample_count": 0,
            "model": model_name,
            "device": device,
            "eval_mode": eval_mode,
        }

    # Sample if needed
    if sample_size is not None and len(texts) > sample_size:
        texts = sample_texts(texts, sample_size, seed)
        logger.info(f"Sampled {sample_size} texts for perplexity computation")

    # Initialize cache
    if use_cache and cache is None:
        cache = PerplexityCache()

    # Resolve device
    if device == "auto":
        device = get_device()

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)

    # Get model-specific completion markers if applying chat template
    if apply_chat_template and completion_markers is None:
        completion_markers = get_chat_template_completion_markers(tokenizer)
        logger.debug(f"Using chat template completion markers: {completion_markers}")

    # Compute perplexities
    perplexities: list[float] = []
    cache_hits = 0
    conversion_failures = 0

    # Cache key includes eval_mode and chat_template to avoid mixing results
    cache_suffix = f"_{eval_mode}"
    if apply_chat_template:
        cache_suffix += "_chat"

    progress_desc = "Computing completion perplexity" if eval_mode == "completion" else "Computing perplexity"
    if apply_chat_template:
        progress_desc += " (chat template)"
    iterator = tqdm(texts, desc=progress_desc, disable=not show_progress)

    for text in iterator:
        # Check cache first (with eval_mode suffix)
        cache_key = f"{model_name}{cache_suffix}"
        if use_cache and cache is not None:
            cached_ppl = cache.get(text, cache_key)
            if cached_ppl is not None:
                perplexities.append(cached_ppl)
                cache_hits += 1
                continue

        # Convert to chat template if requested
        eval_text = text
        if apply_chat_template:
            converted = convert_irca_to_chat_template(text, tokenizer)
            if converted != text:
                eval_text = converted
            else:
                conversion_failures += 1

        # Compute perplexity based on eval_mode
        if eval_mode == "completion":
            ppl = compute_completion_perplexity(eval_text, model, tokenizer, max_length, completion_markers)
        else:
            ppl = compute_sample_perplexity(eval_text, model, tokenizer, max_length)
        perplexities.append(ppl)

        # Cache the result
        if use_cache and cache is not None:
            cache.set(text, cache_key, ppl)

    if conversion_failures > 0:
        logger.warning(
            f"Chat template conversion failed for {conversion_failures}/{len(texts)} samples. "
            "These samples were evaluated in their original format."
        )

    # Save cache
    if use_cache and cache is not None:
        cache.save(f"{model_name}{cache_suffix}")

    logger.info(
        f"Perplexity computed: {len(perplexities)} samples, "
        f"cache_hits={cache_hits}, apply_chat_template={apply_chat_template}"
    )

    if cache_hits > 0:
        logger.info(f"Cache hits: {cache_hits}/{len(texts)}")

    # Filter out infinite values for statistics
    valid_perplexities = [p for p in perplexities if not np.isinf(p)]

    if not valid_perplexities:
        return {
            "mean_perplexity": float("inf"),
            "std_perplexity": 0.0,
            "min_perplexity": float("inf"),
            "max_perplexity": float("inf"),
            "median_perplexity": float("inf"),
            "p90_perplexity": float("inf"),
            "p95_perplexity": float("inf"),
            "sample_count": len(texts),
            "valid_samples": 0,
            "model": model_name,
            "device": device,
            "eval_mode": eval_mode,
        }

    # Compute statistics
    arr = np.array(valid_perplexities)

    result = {
        "mean_perplexity": float(np.mean(arr)),
        "std_perplexity": float(np.std(arr)),
        "min_perplexity": float(np.min(arr)),
        "max_perplexity": float(np.max(arr)),
        "median_perplexity": float(np.median(arr)),
        "p90_perplexity": float(np.percentile(arr, 90)),
        "p95_perplexity": float(np.percentile(arr, 95)),
        "sample_count": len(texts),
        "valid_samples": len(valid_perplexities),
        "model": model_name,
        "device": device,
        "eval_mode": eval_mode,
        "apply_chat_template": apply_chat_template,
    }

    # Optionally include raw perplexities for bootstrap CI computation
    if return_perplexities:
        result["perplexities"] = valid_perplexities

    return result


def compare_datasets(
    original_texts: list[str],
    augmented_texts: list[str],
    model_name: str | None = None,
    quick: bool = False,
    sample_size: int | None = None,
    seed: int = 42,
    show_progress: bool = True,
    eval_mode: str = "full",
    apply_chat_template: bool = True,
) -> dict[str, Any]:
    """
    Compare diversity metrics between original and augmented datasets.

    This is the primary function for evaluating augmentation effectiveness.
    It computes all diversity metrics (perplexity, lexical, optionally semantic)
    and calculates the delta between the two datasets.

    Args:
        original_texts: Original dataset texts.
        augmented_texts: Augmented dataset texts.
        model_name: Model for perplexity computation. Required if not quick mode.
        quick: If True, only compute model-free metrics (lexical).
        sample_size: Sample size for large datasets.
        seed: Random seed for sampling.
        show_progress: Whether to show progress bars.
        eval_mode: "full" for full sample perplexity, "completion" for
                   completion-only perplexity (recommended for SFT eval).
        apply_chat_template: If True (default), convert IRCA-formatted text to
                            the model's native chat template before computing
                            perplexity. Ensures evaluation matches training format.

    Returns:
        Dictionary with comparison results:
        {
            "original": { ... metrics ... },
            "augmented": { ... metrics ... },
            "delta": {
                "mean_perplexity_change": str,  # e.g., "+25.4%"
                "distinct_2_change": str,
                ...
            },
            "summary": str,  # Human-readable summary
        }

    Example:
        >>> original = ["Hello world"] * 100
        >>> augmented = ["Hello world", "Bonjour monde", ...] * 100
        >>> result = compare_datasets(original, augmented, model_name="gpt2")
        >>> print(result["summary"])

        # For SFT evaluation, use completion mode
        >>> result = compare_datasets(original, augmented, model_name="gpt2", eval_mode="completion")
    """
    if not quick and model_name is None:
        raise ValueError("model_name is required when quick=False")

    result: dict[str, Any] = {
        "original": {},
        "augmented": {},
        "delta": {},
    }

    # Compute lexical diversity (always)
    logger.info("Computing lexical diversity...")
    result["original"]["lexical"] = compute_lexical_diversity(original_texts)
    result["augmented"]["lexical"] = compute_lexical_diversity(augmented_texts)

    # Compute deltas for lexical metrics
    for key in ["distinct_1", "distinct_2", "distinct_3"]:
        orig_val = result["original"]["lexical"][key]
        aug_val = result["augmented"]["lexical"][key]
        result["delta"][f"{key}_change"] = format_delta(orig_val, aug_val)

    # Compute perplexity if not quick mode
    if not quick and model_name:
        mode_desc = "completion" if eval_mode == "completion" else "full sample"
        template_desc = " (with chat template)" if apply_chat_template else ""
        logger.info(f"Computing {mode_desc} perplexity{template_desc} with model {model_name}...")

        result["original"]["perplexity"] = compute_perplexity_profile(
            original_texts,
            model_name=model_name,
            sample_size=sample_size,
            seed=seed,
            show_progress=show_progress,
            eval_mode=eval_mode,
            apply_chat_template=apply_chat_template,
        )

        result["augmented"]["perplexity"] = compute_perplexity_profile(
            augmented_texts,
            model_name=model_name,
            sample_size=sample_size,
            seed=seed,
            show_progress=show_progress,
            eval_mode=eval_mode,
            apply_chat_template=apply_chat_template,
        )

        # Compute perplexity deltas
        orig_ppl = result["original"]["perplexity"]["mean_perplexity"]
        aug_ppl = result["augmented"]["perplexity"]["mean_perplexity"]
        result["delta"]["mean_perplexity_change"] = format_delta(orig_ppl, aug_ppl)

        orig_std = result["original"]["perplexity"]["std_perplexity"]
        aug_std = result["augmented"]["perplexity"]["std_perplexity"]
        result["delta"]["std_perplexity_change"] = format_delta(orig_std, aug_std)

    # Record metadata
    result["original"]["sample_count"] = len(original_texts)
    result["augmented"]["sample_count"] = len(augmented_texts)
    result["delta"]["sample_count_change"] = format_delta(len(original_texts), len(augmented_texts))

    # Generate summary
    result["summary"] = _generate_comparison_summary(result, quick)

    return result


def _generate_comparison_summary(result: dict[str, Any], quick: bool) -> str:
    """Generate a human-readable summary of the comparison."""
    lines = []

    orig_count = result["original"]["sample_count"]
    aug_count = result["augmented"]["sample_count"]
    lines.append(f"Dataset size: {orig_count} → {aug_count} ({result['delta']['sample_count_change']})")

    # Lexical summary
    d2_change = result["delta"]["distinct_2_change"]
    lines.append(f"Lexical diversity (Distinct-2): {d2_change}")

    # Perplexity summary
    if not quick and "perplexity" in result["original"]:
        ppl_change = result["delta"]["mean_perplexity_change"]
        lines.append(f"Perplexity change: {ppl_change}")

        # Interpretation
        orig_ppl = result["original"]["perplexity"]["mean_perplexity"]
        aug_ppl = result["augmented"]["perplexity"]["mean_perplexity"]

        if aug_ppl > orig_ppl * 1.1:
            lines.append(
                "Interpretation: Augmented dataset is significantly more challenging for the model. Good for training robustness."
            )
        elif aug_ppl > orig_ppl * 1.02:
            lines.append("Interpretation: Augmented dataset is moderately more challenging. Some diversity gain.")
        else:
            lines.append("Interpretation: Augmented dataset has similar perplexity. Consider stronger augmentation.")

    return "\n".join(lines)


def compute_quick_diversity(
    texts: list[str],
    include_semantic: bool = False,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sample_size: int | None = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute quick, model-free diversity metrics.

    This is a fast alternative when you don't need perplexity-based metrics.
    Useful for sanity checks or when the target model is unavailable.

    Args:
        texts: List of text samples.
        include_semantic: Whether to include Vendi Score (slower).
        embedding_model: Model for semantic metrics.
        sample_size: Sample size for large datasets.
        seed: Random seed.

    Returns:
        Dictionary with diversity metrics.
    """
    result: dict[str, Any] = {
        "sample_count": len(texts),
    }

    # Lexical diversity
    result["lexical"] = compute_lexical_diversity(texts)

    # Semantic diversity (optional)
    if include_semantic:
        from src.diversity.semantic import compute_semantic_diversity

        result["semantic"] = compute_semantic_diversity(
            texts,
            embedding_model=embedding_model,
            sample_size=sample_size,
            seed=seed,
        )

    return result


def analyze_augmentation_contributions(
    texts: list[str],
    augmentation_labels: list[str],
    model_name: str,
    sample_per_augmentation: int = 100,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Analyze per-augmentation-type contribution to diversity.

    Groups samples by their augmentation type and computes perplexity
    statistics for each group to understand which augmentations add
    the most diversity.

    Args:
        texts: List of augmented texts.
        augmentation_labels: Label for each text indicating augmentation type.
        model_name: Model for perplexity computation.
        sample_per_augmentation: Max samples per augmentation type.
        seed: Random seed.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary mapping augmentation type to statistics:
        {
            "translate:fr": {
                "sample_count": 450,
                "mean_perplexity": 52.3,
                "std_perplexity": 12.1,
            },
            "shuffle": { ... },
            ...
        }
    """
    from collections import defaultdict

    # Group by augmentation type
    groups: dict[str, list[str]] = defaultdict(list)
    for text, label in zip(texts, augmentation_labels, strict=False):
        groups[label].append(text)

    result: dict[str, dict[str, float]] = {}

    for aug_type, aug_texts in groups.items():
        # Sample if too many
        if len(aug_texts) > sample_per_augmentation:
            aug_texts = sample_texts(aug_texts, sample_per_augmentation, seed)

        logger.info(f"Computing perplexity for {aug_type} ({len(aug_texts)} samples)...")

        profile = compute_perplexity_profile(
            aug_texts,
            model_name=model_name,
            show_progress=show_progress,
        )

        result[aug_type] = {
            "sample_count": len(groups[aug_type]),
            "mean_perplexity": profile["mean_perplexity"],
            "std_perplexity": profile["std_perplexity"],
        }

    return result
