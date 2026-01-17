"""
Semantic diversity metrics based on embedding space analysis.

These metrics measure diversity in the semantic/meaning space using
sentence embeddings. They are model-agnostic (don't require the target LLM)
but may undervalue syntactic diversity.

Metrics:
- Vendi Score: Effective number of unique samples in embedding space
- Embedding Coverage: Statistics about embedding space distribution
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default embedding model - small and fast
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_sentence_transformer(model_name: str):
    """Load a sentence transformer model lazily."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for semantic diversity metrics. "
            "Install it with: pip install sentence-transformers"
        ) from e


def _compute_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    show_progress: bool = True,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """
    Compute sentence embeddings for a list of texts.

    Args:
        texts: List of text samples to embed.
        model_name: Sentence transformer model to use.
        batch_size: Batch size for encoding.
        show_progress: Whether to show progress bar.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        Array of shape (n_samples, embedding_dim).
    """
    model = _load_sentence_transformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return embeddings


def compute_vendi_score(
    texts: list[str],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    sample_size: int | None = None,
    seed: int = 42,
    show_progress: bool = True,
) -> float:
    """
    Compute Vendi Score for a dataset.

    The Vendi Score (Friedman & Dieng, 2022) measures the effective number
    of unique samples in a dataset. It's computed as the exponential of
    the entropy of the eigenvalue distribution of the similarity matrix.

    sim_matrixey properties:
    - If all samples are identical: Vendi Score ≈ 1
    - If all samples are completely different: Vendi Score ≈ n
    - Duplicating the dataset doesn't change the Vendi Score

    Note: This metric captures semantic diversity well but may undervalue
    syntactic diversity (different surface forms with same meaning).

    Args:
        texts: List of text samples to analyze.
        embedding_model: Sentence transformer model for embeddings.
        batch_size: Batch size for encoding.
        sample_size: If set, sample this many texts for efficiency.
        seed: Random seed for sampling.
        show_progress: Whether to show progress bar.

    Returns:
        Vendi Score (effective number of unique samples).

    Example:
        >>> texts = ["Hello world", "Hello world", "Goodbye world"]
        >>> score = compute_vendi_score(texts)
        >>> print(f"Effective unique samples: {score:.1f}")
        Effective unique samples: 1.8
    """
    if not texts:
        return 0.0

    if len(texts) == 1:
        return 1.0

    # Sample if dataset is too large
    if sample_size is not None and len(texts) > sample_size:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(texts), size=sample_size, replace=False)
        texts = [texts[i] for i in indices]
        logger.info(f"Sampled {sample_size} texts for Vendi Score computation")

    # Compute embeddings
    embeddings = _compute_embeddings(
        texts,
        model_name=embedding_model,
        batch_size=batch_size,
        show_progress=show_progress,
        normalize=True,
    )

    # Compute similarity matrix (cosine similarity for normalized embeddings)
    # sim_matrix[i,j] = embeddings[i] · embeddings[j]
    sim_matrix = embeddings @ embeddings.T

    # Normalize to make eigenvalues sum to 1
    n = len(texts)
    sim_matrix = sim_matrix / n

    # Compute eigenvalues
    try:
        from scipy.linalg import eigvalsh

        eigenvalues = eigvalsh(sim_matrix)
    except ImportError:
        # Fallback to numpy (slower but no scipy required)
        eigenvalues = np.linalg.eigvalsh(sim_matrix)

    # Filter out numerical zeros and negative values (numerical noise)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 1.0

    # Normalize eigenvalues to form a probability distribution
    eigenvalues = eigenvalues / eigenvalues.sum()

    # Compute entropy of eigenvalue distribution
    # H = -Σ λᵢ log(λᵢ)
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    # Vendi Score = exp(entropy)
    return float(np.exp(entropy))


def compute_embedding_coverage(
    texts: list[str],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    sample_size: int | None = 5000,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Compute embedding space coverage metrics.

    These metrics measure how much of the embedding space the dataset covers,
    providing additional insights beyond the Vendi Score.

    Args:
        texts: List of text samples to analyze.
        embedding_model: Sentence transformer model for embeddings.
        batch_size: Batch size for encoding.
        sample_size: Sample size for pairwise distance computation.
        seed: Random seed for sampling.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary with coverage metrics:
        {
            "avg_pairwise_distance": float,  # Mean cosine distance
            "std_pairwise_distance": float,  # Std of cosine distances
            "embedding_spread": float,       # Std of embedding dimensions
            "centroid_distance": float,      # Avg distance to centroid
            "sample_count": int,
        }
    """
    if not texts:
        return {
            "avg_pairwise_distance": 0.0,
            "std_pairwise_distance": 0.0,
            "embedding_spread": 0.0,
            "centroid_distance": 0.0,
            "sample_count": 0,
        }

    # Sample for efficiency if needed
    sampled_texts = texts
    if sample_size is not None and len(texts) > sample_size:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(texts), size=sample_size, replace=False)
        sampled_texts = [texts[i] for i in indices]

    # Compute embeddings
    embeddings = _compute_embeddings(
        sampled_texts,
        model_name=embedding_model,
        batch_size=batch_size,
        show_progress=show_progress,
        normalize=True,
    )

    n = len(embeddings)

    # Compute pairwise cosine distances
    # For normalized embeddings: distance = 1 - similarity = 1 - (a · b)
    similarities = embeddings @ embeddings.T
    distances = 1 - similarities

    # Zero out diagonal (self-distances)
    np.fill_diagonal(distances, 0)

    # Compute statistics (excluding diagonal)
    n_pairs = n * (n - 1)
    avg_distance = distances.sum() / n_pairs if n_pairs > 0 else 0.0
    std_distance = np.sqrt((distances**2).sum() / n_pairs - avg_distance**2) if n_pairs > 0 else 0.0

    # Embedding spread: average std across dimensions
    embedding_spread = float(embeddings.std(axis=0).mean())

    # Centroid distance: average distance to the centroid
    centroid = embeddings.mean(axis=0)
    centroid_normalized = centroid / (np.linalg.norm(centroid) + 1e-10)
    centroid_distances = 1 - (embeddings @ centroid_normalized)
    avg_centroid_distance = float(centroid_distances.mean())

    return {
        "avg_pairwise_distance": float(avg_distance),
        "std_pairwise_distance": float(std_distance),
        "embedding_spread": embedding_spread,
        "centroid_distance": avg_centroid_distance,
        "sample_count": n,
    }


def compute_semantic_diversity(
    texts: list[str],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    sample_size: int | None = 5000,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Compute comprehensive semantic diversity metrics.

    Combines Vendi Score with embedding coverage metrics.

    Args:
        texts: List of text samples to analyze.
        embedding_model: Sentence transformer model for embeddings.
        batch_size: Batch size for encoding.
        sample_size: Sample size for large datasets.
        seed: Random seed for sampling.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary with all semantic diversity metrics:
        {
            "vendi_score": float,
            "effective_diversity_ratio": float,  # vendi_score / sample_count
            "avg_pairwise_distance": float,
            "std_pairwise_distance": float,
            "embedding_spread": float,
            "centroid_distance": float,
            "sample_count": int,
            "embedding_model": str,
        }
    """
    if not texts:
        return {
            "vendi_score": 0.0,
            "effective_diversity_ratio": 0.0,
            "avg_pairwise_distance": 0.0,
            "std_pairwise_distance": 0.0,
            "embedding_spread": 0.0,
            "centroid_distance": 0.0,
            "sample_count": 0,
            "embedding_model": embedding_model,
        }

    # Compute Vendi Score
    vendi = compute_vendi_score(
        texts,
        embedding_model=embedding_model,
        batch_size=batch_size,
        sample_size=sample_size,
        seed=seed,
        show_progress=show_progress,
    )

    # Compute coverage metrics
    coverage = compute_embedding_coverage(
        texts,
        embedding_model=embedding_model,
        batch_size=batch_size,
        sample_size=sample_size,
        seed=seed,
        show_progress=False,  # Already showed progress for vendi
    )

    return {
        "vendi_score": vendi,
        "effective_diversity_ratio": vendi / len(texts) if texts else 0.0,
        **coverage,
        "embedding_model": embedding_model,
    }
