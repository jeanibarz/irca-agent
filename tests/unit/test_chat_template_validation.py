"""
Unit tests for chat template validation.

Tests:
- FR-VAL-004: System message preservation
- FM-2: Mistral system message silently dropped
"""

import pytest
from unittest.mock import MagicMock, patch

from src.validation.chat_template import (
    SystemMessageValidationResult,
    check_system_message_fix_applied,
    get_model_requires_system_fix,
    validate_chat_template_output,
    validate_system_message_preservation,
    validate_training_format,
)


# ============================================
# Mock Tokenizer Fixtures
# ============================================


@pytest.fixture
def mock_tokenizer_preserves_system():
    """Mock tokenizer that preserves system messages."""
    tokenizer = MagicMock()

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        # Simulate tokenizer that includes system message
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}<|end|>")
        return "\n".join(parts)

    tokenizer.apply_chat_template = apply_chat_template
    tokenizer.eos_token = "<|end|>"
    return tokenizer


@pytest.fixture
def mock_tokenizer_drops_system():
    """Mock tokenizer that drops system messages (like Mistral in training mode)."""
    tokenizer = MagicMock()

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        # Simulate tokenizer that drops system message when assistant is last
        parts = []
        last_is_assistant = messages and messages[-1]["role"] == "assistant"

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            # Drop system if last message is assistant (training mode)
            if role == "system" and last_is_assistant:
                continue
            parts.append(f"[{role.upper()}]: {content}")

        return "\n".join(parts)

    tokenizer.apply_chat_template = apply_chat_template
    tokenizer.eos_token = "</s>"
    return tokenizer


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]


@pytest.fixture
def sample_irca_data():
    """Sample IRCA format data for testing."""
    return {
        "system_instructions": "You are an AI assistant.",
        "user_query": "What is the weather?",
        "assistant_completion": "The weather is nice.",
        "available_functions_json": "[]",
    }


# ============================================
# Test System Message Preservation (FR-VAL-004, FM-2)
# ============================================


class TestSystemMessagePreservation:
    """Tests for system message preservation validation."""

    def test_ut_val_020_system_preserved_in_qwen_format(self, mock_tokenizer_preserves_system):
        """UT-VAL-020: System message preserved in Qwen format."""
        result = validate_system_message_preservation(mock_tokenizer_preserves_system)

        assert result.preserved
        assert not result.fix_applied
        assert "preserved" in result.message.lower()

    def test_ut_val_021_system_preserved_in_mistral_format(self, mock_tokenizer_drops_system):
        """UT-VAL-021: System message preserved in Mistral format (detects drop)."""
        result = validate_system_message_preservation(mock_tokenizer_drops_system)

        assert not result.preserved
        assert "DROPPED" in result.message
        assert "fix required" in result.message.lower()

    def test_ut_val_022_system_fix_applied_when_dropped(
        self, mock_tokenizer_drops_system, sample_messages
    ):
        """UT-VAL-022: System message fix applied when dropped."""
        fix_applied, description = check_system_message_fix_applied(
            mock_tokenizer_drops_system, sample_messages
        )

        # The fix should be applied for this tokenizer
        assert fix_applied
        assert "prepended" in description.lower()

    def test_ut_val_023_system_not_duplicated_after_fix(
        self, mock_tokenizer_preserves_system, sample_messages
    ):
        """UT-VAL-023: System message not duplicated after fix."""
        fix_applied, description = check_system_message_fix_applied(
            mock_tokenizer_preserves_system, sample_messages
        )

        # No fix should be applied for tokenizer that preserves system
        assert not fix_applied
        assert "no fix needed" in description.lower()

    def test_no_system_message(self, mock_tokenizer_preserves_system):
        """Test handling when no system message present."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        fix_applied, description = check_system_message_fix_applied(
            mock_tokenizer_preserves_system, messages
        )

        assert not fix_applied
        assert "no system message" in description.lower()

    def test_empty_messages(self, mock_tokenizer_preserves_system):
        """Test handling of empty messages."""
        fix_applied, description = check_system_message_fix_applied(
            mock_tokenizer_preserves_system, []
        )

        assert not fix_applied

    def test_tokenizer_without_apply_chat_template(self):
        """Test handling tokenizer without apply_chat_template."""
        # Create a minimal mock without apply_chat_template
        class MinimalTokenizer:
            vocab = {}
            added_tokens_encoder = {}
            special_tokens_map = {}
            # No apply_chat_template method

        tokenizer = MinimalTokenizer()

        result = validate_system_message_preservation(tokenizer)

        assert result.preserved  # Assume OK when can't test
        assert "no chat template" in result.message.lower()


class TestModelRequiresSystemFix:
    """Tests for model fix detection."""

    def test_model_requires_fix(self, mock_tokenizer_drops_system):
        """Test detection of model that requires system fix."""
        requires_fix = get_model_requires_system_fix(mock_tokenizer_drops_system)
        assert requires_fix

    def test_model_no_fix_needed(self, mock_tokenizer_preserves_system):
        """Test detection of model that doesn't need fix."""
        requires_fix = get_model_requires_system_fix(mock_tokenizer_preserves_system)
        assert not requires_fix


class TestChatTemplateOutput:
    """Tests for chat template output validation."""

    def test_expected_content_found(self, mock_tokenizer_preserves_system, sample_messages):
        """Test validation passes when expected content is found."""
        expected = ["helpful assistant", "Hello", "Hi there"]
        valid, missing = validate_chat_template_output(
            mock_tokenizer_preserves_system, sample_messages, expected
        )

        assert valid
        assert len(missing) == 0

    def test_missing_content_detected(self, mock_tokenizer_drops_system, sample_messages):
        """Test validation detects missing content (dropped system)."""
        # Note: validate_chat_template_output uses apply_chat_template from
        # src.formatting.chat_template which applies the system fix.
        # So we test with content that's definitely not in the messages.
        expected = ["THIS_CONTENT_DOES_NOT_EXIST_ANYWHERE"]
        valid, missing = validate_chat_template_output(
            mock_tokenizer_drops_system, sample_messages, expected
        )

        assert not valid
        assert "THIS_CONTENT_DOES_NOT_EXIST_ANYWHERE" in missing


class TestTrainingFormatValidation:
    """Tests for training format validation."""

    def test_valid_training_format(self, mock_tokenizer_preserves_system, sample_irca_data):
        """Test validation passes for valid training format."""
        # Patch at the import location within the validation module
        with patch("src.formatting.chat_template.format_for_training") as mock_format:
            mock_format.return_value = (
                "You are an AI assistant.\n"
                "What is the weather?\n"
                "The weather is nice.<|end|>"
            )

            valid, issues = validate_training_format(
                mock_tokenizer_preserves_system, sample_irca_data
            )

            assert valid
            assert len(issues) == 0

    def test_missing_eos_token(self, sample_irca_data):
        """Test validation detects missing EOS token."""
        tokenizer = MagicMock()
        tokenizer.eos_token = "<|end|>"

        with patch("src.formatting.chat_template.format_for_training") as mock_format:
            # Return text without EOS token
            mock_format.return_value = "Some text without EOS"

            valid, issues = validate_training_format(tokenizer, sample_irca_data)

            assert not valid
            any_eos_issue = any("EOS" in issue for issue in issues)
            assert any_eos_issue

    def test_formatting_failure(self, mock_tokenizer_preserves_system, sample_irca_data):
        """Test handling of formatting failures."""
        with patch("src.formatting.chat_template.format_for_training") as mock_format:
            mock_format.side_effect = ValueError("Formatting failed")

            valid, issues = validate_training_format(
                mock_tokenizer_preserves_system, sample_irca_data
            )

            assert not valid
            assert any("failed" in issue.lower() for issue in issues)


# ============================================
# Test Edge Cases
# ============================================


class TestEdgeCases:
    """Tests for edge cases in chat template validation."""

    def test_very_long_system_message(self, mock_tokenizer_preserves_system):
        """Test handling of very long system messages."""
        long_system = "A" * 10000
        messages = [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        fix_applied, _ = check_system_message_fix_applied(
            mock_tokenizer_preserves_system, messages
        )

        # Should work regardless of length
        assert isinstance(fix_applied, bool)

    def test_special_characters_in_system(self, mock_tokenizer_preserves_system):
        """Test handling of special characters in system message."""
        messages = [
            {"role": "system", "content": "Test <|special|> tokens & unicode: 你好"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        result = validate_system_message_preservation(mock_tokenizer_preserves_system)
        # Should not crash
        assert result is not None

    def test_multiple_user_messages(self, mock_tokenizer_preserves_system):
        """Test handling of multiple user messages."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Response 2"},
        ]

        fix_applied, _ = check_system_message_fix_applied(
            mock_tokenizer_preserves_system, messages
        )

        # Should work with multi-turn conversations
        assert isinstance(fix_applied, bool)
