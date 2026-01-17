"""
Centralized constants for IRCA format markers.

This module provides a single source of truth for all IRCA structural markers,
preventing inconsistencies across the codebase and enabling validation.

Implements FR-VAL-002: Marker Integrity Verification
"""

from __future__ import annotations

import os

# =============================================================================
# IRCA Format Markers
# =============================================================================

# Primary section markers
MARKER_INSTRUCTIONS = "### INSTRUCTIONS"
MARKER_FUNCTIONS = "### FUNCTIONS AVAILABLE"
MARKER_USER_QUERY = "### USER QUERY"
MARKER_ITERATIVE_CYCLE = "### ITERATIVE RESOLUTION CYCLE"
MARKER_FINAL_ANSWER = "### FINAL ANSWER"

# Secondary markers (used in some variations)
MARKER_EXAMPLE = "EXAMPLE:"
MARKER_FUNCTIONS_INTRO = "The functions available to you are described below."

# =============================================================================
# Required Markers for Valid IRCA Format
# =============================================================================

# Markers that MUST be present for a valid IRCA-formatted sample
REQUIRED_MARKERS = [
    MARKER_INSTRUCTIONS,
    MARKER_FUNCTIONS,
    MARKER_USER_QUERY,
    MARKER_ITERATIVE_CYCLE,
]

# Markers that SHOULD be present in complete samples (validation warnings if missing)
EXPECTED_MARKERS = [
    MARKER_INSTRUCTIONS,
    MARKER_FUNCTIONS,
    MARKER_USER_QUERY,
    MARKER_ITERATIVE_CYCLE,
    MARKER_FINAL_ANSWER,
]

# Marker order (for sequential validation)
MARKER_ORDER = [
    MARKER_INSTRUCTIONS,
    MARKER_EXAMPLE,  # Optional, but should come after INSTRUCTIONS
    MARKER_FUNCTIONS,
    MARKER_USER_QUERY,
    MARKER_ITERATIVE_CYCLE,
    MARKER_FINAL_ANSWER,  # Should be inside ITERATIVE_CYCLE content
]

# =============================================================================
# Model Family Completion Markers
# =============================================================================
# These markers indicate where the model's response/completion begins
# after the prompt. Order matters: more specific patterns should come first.

# Qwen / ChatML style models
QWEN_COMPLETION_MARKERS = [
    "<|im_start|>assistant\n",
    "<|im_start|>assistant",
]

# Mistral / Llama2 instruction format
MISTRAL_COMPLETION_MARKERS = [
    "[/INST]",
]

# Llama3 style models
LLAMA3_COMPLETION_MARKERS = [
    "<|start_header_id|>assistant<|end_header_id|>\n",
    "<|start_header_id|>assistant<|end_header_id|>",
]

# Generic ChatML
GENERIC_CHATML_MARKERS = [
    "<|assistant|>",
]

# IRCA fallback markers (when no chat template detected)
IRCA_COMPLETION_MARKERS = [
    "### ITERATIVE RESOLUTION CYCLE",
    "### ASSISTANT",
    "### RESPONSE",
]

# All completion markers combined (order matters for matching)
ALL_COMPLETION_MARKERS = (
    IRCA_COMPLETION_MARKERS
    + MISTRAL_COMPLETION_MARKERS
    + QWEN_COMPLETION_MARKERS
    + GENERIC_CHATML_MARKERS
    + LLAMA3_COMPLETION_MARKERS
)

# =============================================================================
# Model Family Detection
# =============================================================================

# Model family names
MODEL_FAMILY_QWEN = "qwen"
MODEL_FAMILY_MISTRAL = "mistral"
MODEL_FAMILY_LLAMA3 = "llama3"
MODEL_FAMILY_CHATML = "chatml"
MODEL_FAMILY_IRCA = "irca"
MODEL_FAMILY_UNKNOWN = "unknown"

# Model family to completion markers mapping
MODEL_FAMILY_MARKERS = {
    MODEL_FAMILY_QWEN: QWEN_COMPLETION_MARKERS,
    MODEL_FAMILY_MISTRAL: MISTRAL_COMPLETION_MARKERS,
    MODEL_FAMILY_LLAMA3: LLAMA3_COMPLETION_MARKERS,
    MODEL_FAMILY_CHATML: GENERIC_CHATML_MARKERS,
    MODEL_FAMILY_IRCA: IRCA_COMPLETION_MARKERS,
}

# =============================================================================
# EOS Tokens by Model Family
# =============================================================================

EOS_TOKENS = {
    MODEL_FAMILY_QWEN: "<|im_end|>",
    MODEL_FAMILY_MISTRAL: "</s>",
    MODEL_FAMILY_LLAMA3: "<|eot_id|>",
}

# =============================================================================
# Validation Constants
# =============================================================================

# Minimum expected length for different sections (characters)
MIN_SECTION_LENGTHS = {
    "system_instructions": 50,
    "user_query": 10,
    "assistant_completion": 20,
}

# Maximum expected length (to catch concatenation errors)
MAX_SECTION_LENGTHS = {
    "system_instructions": 10000,
    "user_query": 5000,
    "assistant_completion": 50000,
}

# =============================================================================
# Unsloth Configuration
# =============================================================================

# Environment variable to disable Unsloth (for testing or fallback)
ENV_DISABLE_UNSLOTH = "IRCA_DISABLE_UNSLOTH"
ENV_TORCHDYNAMO_DISABLE = "TORCHDYNAMO_DISABLE"


def is_unsloth_disabled() -> bool:
    """Check if Unsloth is disabled via environment variable."""
    value = os.environ.get(ENV_DISABLE_UNSLOTH, "").lower()
    return value in ("1", "true", "yes")


def get_unsloth_availability() -> tuple[bool, str | None]:
    """
    Check if Unsloth is available and return availability status with reason.

    Returns:
        Tuple of (is_available, reason_if_not_available)
        - (True, None) if Unsloth is available
        - (False, reason) if Unsloth is not available

    Usage:
        available, reason = get_unsloth_availability()
        if not available:
            logger.warning(f"Unsloth not available: {reason}")
    """
    if is_unsloth_disabled():
        return False, f"Disabled via {ENV_DISABLE_UNSLOTH} environment variable"

    try:
        # Set environment before import attempt
        os.environ[ENV_TORCHDYNAMO_DISABLE] = "1"
        from unsloth import FastModel  # noqa: F401

        return True, None
    except ImportError as e:
        return False, f"Import failed: {e}"
    except Exception as e:
        return False, f"Initialization failed: {e}"


# Unsloth model mappings (base model -> Unsloth-optimized version)
UNSLOTH_MODEL_MAPPING: dict[str, str] = {
    # Qwen 3 series
    "qwen3-4b": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "qwen3-8b": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    # Qwen 2.5 series
    "qwen-4b": "unsloth/Qwen2.5-4B-Instruct-bnb-4bit",
    "qwen-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "qwen-14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # Mistral series
    "mistral": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "mistral-v3": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "ministral-3b": "unsloth/Ministral-3B-bnb-4bit",
    # TinyLlama
    "tinyllama": "unsloth/tinyllama-bnb-4bit",
}

# Reverse mapping: HuggingFace base model -> Unsloth-optimized version
BASE_TO_UNSLOTH: dict[str, str] = {
    # Qwen 3 series
    "Qwen/Qwen3-4B": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "Qwen/Qwen3-8B": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    # Qwen 2.5 series
    "Qwen/Qwen2.5-4B-Instruct": "unsloth/Qwen2.5-4B-Instruct-bnb-4bit",
    "Qwen/Qwen2.5-7B-Instruct": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "Qwen/Qwen2.5-14B-Instruct": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # Mistral series
    "mistralai/Mistral-7B-Instruct-v0.2": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "mistralai/Mistral-7B-Instruct-v0.3": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "mistralai/Ministral-3B-2410": "unsloth/Ministral-3B-bnb-4bit",
    # TinyLlama
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "unsloth/tinyllama-bnb-4bit",
}
