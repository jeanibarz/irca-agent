"""Unit tests for model evaluation functionality."""

from unittest.mock import MagicMock, patch

import pytest

from src.diversity.utils import (
    COMPLETION_MARKERS,
    extract_completion,
    find_completion_token_boundary,
)


class TestExtractCompletion:
    """Tests for completion extraction utilities."""

    def test_extract_with_irca_marker(self):
        """Test extraction with IRCA marker."""
        text = """### INSTRUCTIONS
Do something

### USER QUERY
What is 2+2?

### ITERATIVE RESOLUTION CYCLE
Let me think...
The answer is 4.

### FINAL ANSWER
4"""
        prompt, completion = extract_completion(text)

        assert "### ITERATIVE RESOLUTION CYCLE" in prompt
        assert prompt.endswith("### ITERATIVE RESOLUTION CYCLE")
        assert completion.startswith("Let me think")
        assert "The answer is 4" in completion
        assert "### FINAL ANSWER" in completion

    def test_extract_with_assistant_marker(self):
        """Test extraction with ASSISTANT marker."""
        text = """### USER
Hello

### ASSISTANT
Hi there!"""
        prompt, completion = extract_completion(text)

        assert "### ASSISTANT" in prompt
        assert completion == "Hi there!"

    def test_extract_with_inst_marker(self):
        """Test extraction with [/INST] marker (Llama format)."""
        text = "[INST] What is AI? [/INST]AI is artificial intelligence."
        prompt, completion = extract_completion(text)

        assert "[/INST]" in prompt
        assert completion == "AI is artificial intelligence."

    def test_extract_with_chatml_marker(self):
        """Test extraction with ChatML format."""
        text = "<|user|>\nHello\n<|assistant|>\nHi!"
        prompt, completion = extract_completion(text)

        assert "<|assistant|>" in prompt
        assert completion == "Hi!"

    def test_extract_no_marker(self):
        """Test when no marker is found."""
        text = "This is just plain text without markers."
        prompt, completion = extract_completion(text)

        assert prompt == ""
        assert completion == text

    def test_extract_custom_markers(self):
        """Test with custom markers."""
        text = "Question: What? ANSWER: This is the answer."
        prompt, completion = extract_completion(text, markers=["ANSWER:"])

        assert "ANSWER:" in prompt
        assert completion == "This is the answer."

    def test_extract_multiple_markers_uses_first(self):
        """Test that first matching marker is used."""
        text = """### ASSISTANT
First response

### ITERATIVE RESOLUTION CYCLE
Second response"""
        # ITERATIVE is listed first in COMPLETION_MARKERS
        prompt, completion = extract_completion(text)

        # Should use ITERATIVE since it's listed first
        assert "### ITERATIVE RESOLUTION CYCLE" in prompt
        assert completion == "Second response"

    def test_extract_empty_completion(self):
        """Test when completion is empty."""
        text = "### ASSISTANT\n"
        prompt, completion = extract_completion(text)

        assert "### ASSISTANT" in prompt
        assert completion == ""


class TestFindCompletionTokenBoundary:
    """Tests for token boundary finding."""

    def test_boundary_with_marker(self):
        """Test finding boundary with marker."""
        text = "Prompt text\n### ITERATIVE RESOLUTION CYCLE\nCompletion text"

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}

        boundary = find_completion_token_boundary(text, mock_tokenizer)

        # Should call tokenizer with prompt portion
        mock_tokenizer.assert_called_once()
        assert boundary == 5  # Length of tokenized prompt

    def test_boundary_no_marker(self):
        """Test when no marker is found."""
        text = "Plain text without markers"

        mock_tokenizer = MagicMock()

        boundary = find_completion_token_boundary(text, mock_tokenizer)

        assert boundary == 0
        mock_tokenizer.assert_not_called()


class TestLoRADetection:
    """Tests for LoRA adapter detection."""

    def test_is_lora_adapter_with_config(self, tmp_path):
        """Test detection of LoRA adapter directory."""
        from src.diversity.utils import is_lora_adapter

        # Create a mock adapter directory
        adapter_dir = tmp_path / "my-adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "gpt2"}')

        assert is_lora_adapter(str(adapter_dir)) is True

    def test_is_lora_adapter_without_config(self, tmp_path):
        """Test non-LoRA directory detection."""
        from src.diversity.utils import is_lora_adapter

        # Create a regular directory without adapter_config.json
        regular_dir = tmp_path / "regular-model"
        regular_dir.mkdir()
        (regular_dir / "config.json").write_text("{}")

        assert is_lora_adapter(str(regular_dir)) is False

    def test_is_lora_adapter_nonexistent_path(self):
        """Test with non-existent path."""
        from src.diversity.utils import is_lora_adapter

        assert is_lora_adapter("/nonexistent/path") is False

    def test_get_lora_base_model(self, tmp_path):
        """Test extracting base model from adapter config."""
        from src.diversity.utils import get_lora_base_model

        adapter_dir = tmp_path / "my-adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "Qwen/Qwen3-4B", "peft_type": "LORA"}'
        )

        base_model = get_lora_base_model(str(adapter_dir))
        assert base_model == "Qwen/Qwen3-4B"

    def test_get_lora_base_model_missing_field(self, tmp_path):
        """Test error when base_model_name_or_path is missing."""
        import pytest

        from src.diversity.utils import get_lora_base_model

        adapter_dir = tmp_path / "bad-adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')

        with pytest.raises(ValueError, match="missing"):
            get_lora_base_model(str(adapter_dir))


class TestCompletionMarkersConstant:
    """Tests for the completion markers constant."""

    def test_irca_marker_present(self):
        """IRCA marker should be first/primary."""
        assert "### ITERATIVE RESOLUTION CYCLE" in COMPLETION_MARKERS
        assert COMPLETION_MARKERS[0] == "### ITERATIVE RESOLUTION CYCLE"

    def test_common_markers_present(self):
        """Common markers should be included."""
        assert "### ASSISTANT" in COMPLETION_MARKERS
        assert "[/INST]" in COMPLETION_MARKERS
        assert "<|assistant|>" in COMPLETION_MARKERS


class TestEvaluationFunctions:
    """Tests for high-level evaluation functions."""

    @pytest.fixture
    def mock_model_loading(self):
        """Patch model loading to avoid GPU requirements."""
        with patch("src.diversity.evaluation.load_model_and_tokenizer") as mock:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock.return_value = (mock_model, mock_tokenizer)
            yield mock

    @pytest.fixture
    def mock_perplexity(self):
        """Patch perplexity computation."""
        with patch("src.diversity.evaluation.compute_perplexity_profile") as mock:
            mock.return_value = {
                "mean_perplexity": 50.0,
                "std_perplexity": 10.0,
                "min_perplexity": 20.0,
                "max_perplexity": 100.0,
                "median_perplexity": 45.0,
                "p90_perplexity": 80.0,
                "p95_perplexity": 90.0,
                "sample_count": 100,
                "valid_samples": 98,
                "model": "test-model",
                "device": "cpu",
                "eval_mode": "completion",
            }
            yield mock

    def test_evaluate_robustness_single_model(self, mock_perplexity):
        """Test robustness evaluation with single model."""
        from src.diversity.evaluation import evaluate_model_robustness

        texts = ["Sample 1", "Sample 2"]
        result = evaluate_model_robustness(
            texts=texts,
            finetuned_model="test-model",
            eval_mode="completion",
        )

        assert "finetuned" in result
        assert result["eval_mode"] == "completion"
        assert "baseline" not in result
        assert "summary" in result

        mock_perplexity.assert_called_once()

    def test_evaluate_robustness_with_baseline(self, mock_perplexity):
        """Test robustness evaluation with baseline comparison."""
        from src.diversity.evaluation import evaluate_model_robustness

        # Different return values for finetuned vs baseline
        mock_perplexity.side_effect = [
            {
                "mean_perplexity": 25.0,  # Finetuned - lower
                "std_perplexity": 5.0,
                "min_perplexity": 10.0,
                "max_perplexity": 50.0,
                "median_perplexity": 22.0,
                "p90_perplexity": 40.0,
                "p95_perplexity": 45.0,
                "sample_count": 100,
                "valid_samples": 98,
                "model": "finetuned",
                "device": "cpu",
                "eval_mode": "completion",
            },
            {
                "mean_perplexity": 75.0,  # Baseline - higher
                "std_perplexity": 15.0,
                "min_perplexity": 30.0,
                "max_perplexity": 150.0,
                "median_perplexity": 70.0,
                "p90_perplexity": 120.0,
                "p95_perplexity": 135.0,
                "sample_count": 100,
                "valid_samples": 95,
                "model": "baseline",
                "device": "cpu",
                "eval_mode": "completion",
            },
        ]

        texts = ["Sample 1", "Sample 2"]
        result = evaluate_model_robustness(
            texts=texts,
            finetuned_model="finetuned-model",
            baseline_model="baseline-model",
            eval_mode="completion",
        )

        assert "finetuned" in result
        assert "baseline" in result
        assert "improvement" in result

        # Check improvement calculation (should be negative = improvement)
        assert result["improvement"]["relative_change"] < 0
        assert "-" in result["improvement"]["mean_perplexity_change"]

    def test_evaluate_generalization(self, mock_perplexity):
        """Test generalization evaluation."""
        from src.diversity.evaluation import evaluate_generalization

        # Different values for train vs test
        mock_perplexity.side_effect = [
            {
                "mean_perplexity": 20.0,
                "std_perplexity": 5.0,
                "min_perplexity": 10.0,
                "max_perplexity": 40.0,
                "median_perplexity": 18.0,
                "p90_perplexity": 30.0,
                "p95_perplexity": 35.0,
                "sample_count": 100,
                "valid_samples": 98,
                "model": "ft",
                "device": "cpu",
                "eval_mode": "completion",
            },  # FT train
            {
                "mean_perplexity": 30.0,
                "std_perplexity": 8.0,
                "min_perplexity": 15.0,
                "max_perplexity": 60.0,
                "median_perplexity": 28.0,
                "p90_perplexity": 45.0,
                "p95_perplexity": 50.0,
                "sample_count": 50,
                "valid_samples": 48,
                "model": "ft",
                "device": "cpu",
                "eval_mode": "completion",
            },  # FT test
        ]

        train_texts = ["Train 1", "Train 2"]
        test_texts = ["Test 1"]

        result = evaluate_generalization(
            train_texts=train_texts,
            test_texts=test_texts,
            finetuned_model="finetuned-model",
            eval_mode="completion",
        )

        assert "finetuned" in result
        assert "train" in result["finetuned"]
        assert "test" in result["finetuned"]
        assert "gap" in result["finetuned"]

        # Gap should be positive (test > train)
        assert result["finetuned"]["gap"]["absolute"] > 0

    def test_compare_models(self, mock_perplexity):
        """Test multi-model comparison."""
        from src.diversity.evaluation import compare_models_on_dataset

        mock_perplexity.side_effect = [
            {
                "mean_perplexity": 30.0,
                "std_perplexity": 5.0,
                "min_perplexity": 10.0,
                "max_perplexity": 50.0,
                "median_perplexity": 28.0,
                "p90_perplexity": 40.0,
                "p95_perplexity": 45.0,
                "sample_count": 100,
                "valid_samples": 98,
                "model": "model1",
                "device": "cpu",
                "eval_mode": "completion",
            },
            {
                "mean_perplexity": 50.0,
                "std_perplexity": 10.0,
                "min_perplexity": 20.0,
                "max_perplexity": 100.0,
                "median_perplexity": 45.0,
                "p90_perplexity": 80.0,
                "p95_perplexity": 90.0,
                "sample_count": 100,
                "valid_samples": 95,
                "model": "model2",
                "device": "cpu",
                "eval_mode": "completion",
            },
        ]

        texts = ["Sample 1"]
        result = compare_models_on_dataset(
            texts=texts,
            models=["model1", "model2"],
            eval_mode="completion",
        )

        assert "models" in result
        assert "ranking" in result
        assert len(result["ranking"]) == 2

        # model1 should be ranked first (lower perplexity)
        assert result["ranking"][0]["model"] == "model1"
