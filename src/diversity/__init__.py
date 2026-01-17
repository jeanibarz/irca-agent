"""
Dataset diversity metrics for SFT evaluation.

This module provides tools to measure dataset diversity before and after augmentation,
with a focus on training effectiveness for Supervised Fine-Tuning (SFT).

Primary metrics (model-dependent):
- Perplexity Profile: How "surprising" the dataset is to a specific model
- Augmentation Effectiveness: Did augmentation improve generalization?

Secondary metrics (model-agnostic):
- Vendi Score: Effective number of unique samples in embedding space
- Distinct-n: Lexical diversity via n-gram uniqueness
"""

from src.diversity.evaluation import (
    compare_models_on_dataset,
    evaluate_generalization,
    evaluate_model_robustness,
)
from src.diversity.lexical import (
    compute_distinct_n,
    compute_lexical_diversity,
)
from src.diversity.perplexity import (
    compare_datasets,
    compute_completion_perplexity,
    compute_perplexity_profile,
)
from src.diversity.report import generate_report
from src.diversity.semantic import (
    compute_embedding_coverage,
    compute_vendi_score,
)

__all__ = [
    # Lexical (fast, model-free)
    "compute_distinct_n",
    "compute_lexical_diversity",
    # Semantic (medium, requires embedding model)
    "compute_vendi_score",
    "compute_embedding_coverage",
    # Perplexity (slow, requires LLM)
    "compute_perplexity_profile",
    "compute_completion_perplexity",
    "compare_datasets",
    # Model evaluation
    "evaluate_model_robustness",
    "evaluate_generalization",
    "compare_models_on_dataset",
    # Report generation
    "generate_report",
]
