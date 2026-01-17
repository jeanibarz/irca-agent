"""
Utilities for diversity computation.

Provides:
- Device detection (CUDA/MPS/CPU)
- Model loading helpers
- Sampling utilities
- Perplexity caching
- Chat template conversion for consistent evaluation
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def convert_irca_to_chat_template(
    text: str,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """
    Convert IRCA-formatted text to model's native chat template format.

    This ensures diversity evaluation uses the same format as training,
    providing accurate perplexity measurements.

    Args:
        text: IRCA-formatted text (with ### INSTRUCTIONS, ### USER QUERY, etc.)
        tokenizer: Model's tokenizer with apply_chat_template method.

    Returns:
        Text formatted with the model's native chat template.

    Example:
        >>> text = "### INSTRUCTIONS\\nYou are helpful.\\n### USER QUERY\\nHi"
        >>> formatted = convert_irca_to_chat_template(text, qwen_tokenizer)
        >>> # Returns: "<|im_start|>system\\nYou are helpful.<|im_end|>..."
    """
    from src.core.prompt_builder import parse_corrected_agent_trace
    from src.formatting.chat_template import apply_chat_template, irca_to_messages

    try:
        # Parse IRCA format to components
        parsed = parse_corrected_agent_trace(text)

        # Convert to messages format
        messages = irca_to_messages(parsed)

        # Apply model's chat template
        formatted = apply_chat_template(tokenizer, messages, add_generation_prompt=False)

        # Ensure EOS token is present
        if tokenizer.eos_token and not formatted.rstrip().endswith(tokenizer.eos_token):
            formatted = formatted.rstrip() + tokenizer.eos_token

        return formatted

    except Exception as e:
        logger.warning(f"Failed to convert IRCA to chat template: {e}. Using original text.")
        return text


def get_chat_template_completion_markers(tokenizer: PreTrainedTokenizer) -> list[str]:
    """
    Get completion markers specific to a model's chat template.

    Different models use different markers to indicate assistant responses:
    - Qwen: <|im_start|>assistant
    - Llama3: <|start_header_id|>assistant<|end_header_id|>
    - Mistral: [/INST]

    Args:
        tokenizer: Model's tokenizer.

    Returns:
        List of completion markers for this model.
    """
    markers = []

    # Try to detect model type from tokenizer
    vocab = getattr(tokenizer, "vocab", {}) or {}
    added_tokens = getattr(tokenizer, "added_tokens_encoder", {}) or {}
    all_tokens = {**vocab, **added_tokens}

    # Qwen/ChatML style
    if "<|im_start|>" in all_tokens or "im_start" in str(tokenizer.special_tokens_map):
        markers.extend(
            [
                "<|im_start|>assistant\n",
                "<|im_start|>assistant",
            ]
        )

    # Llama3 style
    if "<|start_header_id|>" in all_tokens:
        markers.extend(
            [
                "<|start_header_id|>assistant<|end_header_id|>\n",
                "<|start_header_id|>assistant<|end_header_id|>",
            ]
        )

    # Mistral/Llama2 style
    if "[INST]" in all_tokens or hasattr(tokenizer, "inst_token"):
        markers.append("[/INST]")

    # Generic ChatML
    if "<|assistant|>" in all_tokens:
        markers.append("<|assistant|>")

    # Fallback to IRCA markers if no chat template detected
    if not markers:
        markers = [
            "### ITERATIVE RESOLUTION CYCLE",
            "### ASSISTANT",
        ]

    return markers


def get_device() -> str:
    """
    Detect the best available device for computation.

    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def sample_texts(
    texts: list[str],
    sample_size: int,
    seed: int = 42,
) -> list[str]:
    """
    Randomly sample texts from a list.

    Args:
        texts: List of texts to sample from.
        sample_size: Number of texts to sample.
        seed: Random seed for reproducibility.

    Returns:
        Sampled list of texts.
    """
    if len(texts) <= sample_size:
        return texts

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(texts), size=sample_size, replace=False)
    return [texts[i] for i in indices]


def is_lora_adapter(model_path: str) -> bool:
    """
    Check if a model path is a LoRA adapter directory.

    Args:
        model_path: Path to model or adapter.

    Returns:
        True if the path contains adapter_config.json.
    """
    adapter_config = Path(model_path) / "adapter_config.json"
    return adapter_config.exists()


def get_lora_base_model(adapter_path: str) -> str:
    """
    Get the base model name from a LoRA adapter config.

    Args:
        adapter_path: Path to LoRA adapter directory.

    Returns:
        Base model name or path.

    Raises:
        ValueError: If adapter_config.json is missing or invalid.
    """
    adapter_config_path = Path(adapter_path) / "adapter_config.json"

    if not adapter_config_path.exists():
        raise ValueError(f"No adapter_config.json found at {adapter_path}")

    with open(adapter_config_path) as f:
        config = json.load(f)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("adapter_config.json missing 'base_model_name_or_path'")

    return base_model


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    torch_dtype: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer for perplexity computation.

    Supports both full models and LoRA adapters. For LoRA adapters,
    automatically loads the base model and applies the adapter.

    Args:
        model_name: HuggingFace model name, local path, or LoRA adapter path.
        device: Device to load model on ("auto", "cuda", "mps", "cpu").
        torch_dtype: Torch dtype ("auto", "float16", "bfloat16", "float32").

    Returns:
        Tuple of (model, tokenizer).

    Example:
        # Load a full model
        model, tokenizer = load_model_and_tokenizer("gpt2")

        # Load a LoRA adapter (auto-detects base model)
        model, tokenizer = load_model_and_tokenizer("models/my-lora-adapter")
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required for perplexity computation. "
            "Install them with: pip install transformers torch"
        ) from e

    # Resolve device
    if device == "auto":
        device = get_device()

    # Resolve dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, "auto")

    # Check if this is a LoRA adapter
    if is_lora_adapter(model_name):
        return _load_lora_model(model_name, device, dtype)

    logger.info(f"Loading model {model_name} on {device} with dtype {torch_dtype}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    return model, tokenizer


def _load_lora_model(
    adapter_path: str,
    device: str,
    dtype: torch.dtype | str,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a LoRA adapter with its base model.

    Args:
        adapter_path: Path to LoRA adapter directory.
        device: Device to load on.
        dtype: Torch dtype.

    Returns:
        Tuple of (model with adapter, tokenizer).
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        if "peft" in str(e).lower():
            raise ImportError("PEFT is required for loading LoRA adapters. Install it with: pip install peft") from e
        raise

    # Get base model from adapter config
    base_model_name = get_lora_base_model(adapter_path)

    logger.info(f"Loading LoRA adapter from {adapter_path}")
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Device: {device}, dtype: {dtype}")

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info(f"Loading base model {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )

    # Apply LoRA adapter
    logger.info(f"Applying LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    logger.info("LoRA model loaded successfully")

    return model, tokenizer


def text_hash(text: str) -> str:
    """
    Compute a hash of a text for caching purposes.

    Args:
        text: Text to hash.

    Returns:
        SHA256 hash (first 16 characters).
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class PerplexityCache:
    """
    Cache for per-sample perplexity values.

    Stores perplexity values keyed by (text_hash, model_name) to avoid
    recomputation on repeated calls.
    """

    def __init__(self, cache_dir: str | Path = ".diversity_cache"):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, float] = {}

    def _cache_key(self, text_hash: str, model_name: str) -> str:
        """Generate a cache key."""
        model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:8]
        return f"{text_hash}_{model_hash}"

    def _cache_file(self, model_name: str) -> Path:
        """Get the cache file path for a model."""
        model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:16]
        return self.cache_dir / f"ppl_{model_hash}.json"

    def _load_file_cache(self, model_name: str) -> dict[str, float]:
        """Load cache from file."""
        cache_file = self._cache_file(model_name)
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_file_cache(self, model_name: str, cache: dict[str, float]):
        """Save cache to file."""
        cache_file = self._cache_file(model_name)
        try:
            with open(cache_file, "w") as f:
                json.dump(cache, f)
        except OSError as e:
            logger.warning(f"Failed to save perplexity cache: {e}")

    def get(self, text: str, model_name: str) -> float | None:
        """
        Get cached perplexity for a text.

        Args:
            text: The text to look up.
            model_name: Model name used for perplexity computation.

        Returns:
            Cached perplexity value, or None if not cached.
        """
        key = self._cache_key(text_hash(text), model_name)

        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check file cache
        file_cache = self._load_file_cache(model_name)
        if key in file_cache:
            self._memory_cache[key] = file_cache[key]
            return file_cache[key]

        return None

    def set(self, text: str, model_name: str, perplexity: float):
        """
        Cache a perplexity value.

        Args:
            text: The text.
            model_name: Model name used for perplexity computation.
            perplexity: Computed perplexity value.
        """
        key = self._cache_key(text_hash(text), model_name)
        self._memory_cache[key] = perplexity

    def save(self, model_name: str):
        """
        Persist memory cache to disk.

        Args:
            model_name: Model name to save cache for.
        """
        file_cache = self._load_file_cache(model_name)

        # Merge memory cache into file cache
        model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:8]
        for key, value in self._memory_cache.items():
            if key.endswith(f"_{model_hash}"):
                file_cache[key] = value

        self._save_file_cache(model_name, file_cache)

    def clear(self, model_name: str | None = None):
        """
        Clear the cache.

        Args:
            model_name: If provided, only clear cache for this model.
                       If None, clear all caches.
        """
        self._memory_cache.clear()

        if model_name:
            cache_file = self._cache_file(model_name)
            if cache_file.exists():
                cache_file.unlink()
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("ppl_*.json"):
                cache_file.unlink()

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        file_count = len(list(self.cache_dir.glob("ppl_*.json")))
        memory_count = len(self._memory_cache)

        return {
            "memory_entries": memory_count,
            "cache_files": file_count,
        }


# Completion extraction markers - covers multiple model formats
# Order matters: more specific patterns should come before generic ones
COMPLETION_MARKERS = [
    # IRCA format markers
    "### ITERATIVE RESOLUTION CYCLE",  # Primary IRCA marker
    "### ASSISTANT",  # Alternative
    "### RESPONSE",  # Alternative
    # Mistral/Llama2 instruction format
    "[/INST]",
    # ChatML format (Qwen, Yi, some Llama variants)
    "<|im_start|>assistant\n",  # Qwen with newline
    "<|im_start|>assistant",  # Qwen without newline
    "<|assistant|>",  # Generic ChatML
    # Llama3 format
    "<|start_header_id|>assistant<|end_header_id|>\n",  # Full Llama3 marker
    "<|start_header_id|>assistant<|end_header_id|>",
]


def extract_completion(
    text: str,
    markers: list[str] | None = None,
) -> tuple[str, str]:
    """
    Extract prompt and completion from formatted text.

    Searches for known markers that separate the prompt/context from
    the model's completion/response.

    Args:
        text: Full text containing prompt and completion.
        markers: List of markers to search for. If None, uses defaults.

    Returns:
        Tuple of (prompt_with_marker, completion).
        If no marker found, returns ("", text).
    """
    if markers is None:
        markers = COMPLETION_MARKERS

    for marker in markers:
        if marker in text:
            idx = text.find(marker)
            prompt = text[: idx + len(marker)]
            completion = text[idx + len(marker) :].strip()
            return prompt, completion

    # No marker found - treat entire text as completion
    return "", text


def find_completion_token_boundary(
    text: str,
    tokenizer: PreTrainedTokenizer,
    markers: list[str] | None = None,
) -> int:
    """
    Find the token index where the completion starts.

    This is used to mask prompt tokens during loss computation.

    Args:
        text: Full text containing prompt and completion.
        tokenizer: Tokenizer to use.
        markers: Completion markers to search for.

    Returns:
        Token index where completion starts.
        Returns 0 if no marker found (evaluate all tokens).
    """
    prompt, completion = extract_completion(text, markers)

    if not prompt:
        return 0

    # Tokenize prompt to find boundary
    prompt_tokens = tokenizer(prompt, add_special_tokens=True)
    return len(prompt_tokens["input_ids"])


def format_number(value: float, precision: int = 2) -> str:
    """
    Format a number for display.

    Args:
        value: Number to format.
        precision: Decimal precision.

    Returns:
        Formatted string.
    """
    if abs(value) >= 1000:
        return f"{value:,.{precision}f}"
    else:
        return f"{value:.{precision}f}"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a decimal as a percentage.

    Args:
        value: Decimal value (e.g., 0.25 for 25%).
        precision: Decimal precision.

    Returns:
        Formatted string (e.g., "25.0%").
    """
    return f"{value * 100:.{precision}f}%"


def format_delta(
    original: float,
    new: float,
    precision: int = 1,
    as_percentage: bool = True,
) -> str:
    """
    Format the change between two values.

    Args:
        original: Original value.
        new: New value.
        precision: Decimal precision.
        as_percentage: Whether to show as percentage change.

    Returns:
        Formatted string (e.g., "+25.0%" or "-10.5").
    """
    if original == 0:
        return "N/A"

    if as_percentage:
        delta = (new - original) / original * 100
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.{precision}f}%"
    else:
        delta = new - original
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.{precision}f}"
