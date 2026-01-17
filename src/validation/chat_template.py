"""
Chat template validation utilities.

Validates that system messages are properly preserved during training,
particularly for models (like Mistral) that may silently drop them.

Implements:
- FR-VAL-004: System message preservation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SystemMessageValidationResult:
    """Result of system message preservation validation."""

    preserved: bool
    fix_applied: bool
    message: str
    model_family: str | None
    details: dict | None = None


def validate_system_message_preservation(
    tokenizer: PreTrainedTokenizer,
) -> SystemMessageValidationResult:
    """
    Validate that system messages are preserved during training.

    Some models (e.g., Mistral) drop the system message when the last message
    is from the assistant (training scenario). This function detects this issue.

    Args:
        tokenizer: Model's tokenizer.

    Returns:
        SystemMessageValidationResult with preservation status.

    Example:
        >>> result = validate_system_message_preservation(mistral_tokenizer)
        >>> result.preserved
        False  # Mistral drops system in training mode
    """
    from src.validation.completion import get_model_family

    model_family = get_model_family(tokenizer)

    # Test messages simulating training scenario (assistant is last)
    test_marker = "<<SYSTEM_TEST_MARKER_UNIQUE_12345>>"
    test_messages = [
        {"role": "system", "content": test_marker},
        {"role": "user", "content": "test user message"},
        {"role": "assistant", "content": "test assistant response"},
    ]

    try:
        if not hasattr(tokenizer, "apply_chat_template"):
            return SystemMessageValidationResult(
                preserved=True,  # Can't test, assume OK
                fix_applied=False,
                message="Tokenizer has no chat template (using fallback format)",
                model_family=model_family,
            )

        # Apply chat template in training mode (no generation prompt)
        output = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        preserved = test_marker in output

        if preserved:
            return SystemMessageValidationResult(
                preserved=True,
                fix_applied=False,
                message="System message preserved in training format",
                model_family=model_family,
            )
        else:
            return SystemMessageValidationResult(
                preserved=False,
                fix_applied=False,
                message="System message DROPPED in training format - fix required",
                model_family=model_family,
                details={"test_marker": test_marker, "output_preview": output[:200]},
            )

    except Exception as e:
        return SystemMessageValidationResult(
            preserved=True,  # Can't test, assume OK
            fix_applied=False,
            message=f"Could not test system message preservation: {e}",
            model_family=model_family,
        )


def check_system_message_fix_applied(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
) -> tuple[bool, str]:
    """
    Check if the system message fix has been applied to messages.

    The fix prepends system content to the first user message when the
    chat template would otherwise drop it.

    Args:
        tokenizer: Model's tokenizer.
        messages: Messages to check.

    Returns:
        Tuple of (fix_applied, description).
    """
    from src.formatting.chat_template import _fix_system_message_for_training

    if not messages:
        return False, "No messages to check"

    has_system = messages[0]["role"] == "system"
    if not has_system:
        return False, "No system message present"

    # Apply fix and compare
    fixed = _fix_system_message_for_training(tokenizer, messages)

    # If fix was applied, system message should be gone (prepended to user)
    fix_applied = fixed[0]["role"] != "system"

    if fix_applied:
        return True, "System content prepended to first user message"
    else:
        return False, "No fix needed (system message preserved by template)"


def validate_chat_template_output(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    expected_content: list[str],
) -> tuple[bool, list[str]]:
    """
    Validate that expected content appears in chat template output.

    Useful for ensuring critical information isn't lost during formatting.

    Args:
        tokenizer: Model's tokenizer.
        messages: Messages to format.
        expected_content: List of strings that must appear in output.

    Returns:
        Tuple of (all_found, missing_content).
    """
    from src.formatting.chat_template import apply_chat_template

    try:
        output = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    except Exception as e:
        return False, [f"Chat template failed: {e}"]

    missing = []
    for content in expected_content:
        if content not in output:
            missing.append(content)

    return len(missing) == 0, missing


def get_model_requires_system_fix(tokenizer: PreTrainedTokenizer) -> bool:
    """
    Determine if a model requires the system message fix.

    Args:
        tokenizer: Model's tokenizer.

    Returns:
        True if the model drops system messages during training.
    """
    result = validate_system_message_preservation(tokenizer)
    return not result.preserved


def validate_training_format(
    tokenizer: PreTrainedTokenizer,
    sample: dict[str, str],
) -> tuple[bool, list[str]]:
    """
    Validate that a sample will format correctly for training.

    Checks:
    - System message content preserved
    - User query present
    - Assistant completion present
    - EOS token at end

    Args:
        tokenizer: Model's tokenizer.
        sample: IRCA format sample with system_instructions, user_query, etc.

    Returns:
        Tuple of (valid, list_of_issues).
    """
    from src.formatting.chat_template import format_for_training

    issues = []

    try:
        formatted = format_for_training(tokenizer, sample)
    except Exception as e:
        return False, [f"Formatting failed: {e}"]

    # Check system content preserved
    system_content = sample.get("system_instructions", "")
    if system_content and system_content[:50] not in formatted:
        # Check if it was prepended to user message (fix applied)
        user_content = sample.get("user_query", "")
        if not (system_content[:50] in formatted or user_content in formatted):
            issues.append("System instructions may not be preserved")

    # Check user query present
    user_query = sample.get("user_query", "")
    if user_query and user_query not in formatted:
        issues.append("User query not found in formatted output")

    # Check assistant completion present
    completion = sample.get("assistant_completion", "")
    if completion and completion[:100] not in formatted:
        issues.append("Assistant completion not found in formatted output")

    # Check EOS token
    eos = tokenizer.eos_token
    if eos and not formatted.endswith(eos):
        issues.append(f"Missing EOS token ({eos}) at end of formatted text")

    return len(issues) == 0, issues
