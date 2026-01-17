"""
Pytest Configuration and Fixtures

This module provides shared fixtures for all tests.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


# ============================================
# Sample Data Fixtures
# ============================================


@pytest.fixture
def sample_functions_json() -> str:
    """Sample function definitions in JSON format."""
    functions = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    ]
    return json.dumps(functions)


@pytest.fixture
def sample_user_query() -> str:
    """Sample user query."""
    return "What's the weather like in Paris today?"


@pytest.fixture
def sample_prompt_data() -> dict:
    """Sample data for prompt building."""
    return {
        "system_instructions": "You are a helpful assistant.",
        "example": "User: Hello\nAssistant: Hi there!<|wait|>",
        "available_functions_json": '[{"name": "test_func"}]',
        "user_query": "Help me with something",
        "assistant_completion": "Thought: I should help the user.",
    }


@pytest.fixture
def sample_training_sample() -> dict:
    """Sample training data format."""
    return {
        "corrected_agent_trace": [
            {
                "value": """
### INSTRUCTIONS
You are an AI assistant that uses functions.

EXAMPLE:
User: Test
Assistant: OK<|wait|>

The functions available to you are described below.

### FUNCTIONS AVAILABLE
[{"name": "get_weather", "description": "Get weather"}]

### USER QUERY
What is the weather?

### ITERATIVE RESOLUTION CYCLE
Thought: I should get the weather.
Action choice: call function
Call function: {"name": "get_weather"}, "parameters": {"location": "Paris"}<|wait|>
Output[abc123]: {"temp": 20, "condition": "sunny"}
Thought: I have the weather info.
Action choice: final answer

### FINAL ANSWER
The weather in Paris is 20 degrees and sunny.<|wait|>
"""
            }
        ]
    }


# ============================================
# Mock Fixtures
# ============================================


@pytest.fixture
def mock_settings():
    """Mock Settings object."""
    from pathlib import Path

    settings = MagicMock()
    settings.workspace_dir = Path("/workspace")
    settings.models_dir = Path("models")
    settings.datasets_dir = Path("datasets")
    settings.finetuned_models_dir = Path("finetuned_models")
    settings.models_path = Path("/workspace/models")
    settings.datasets_path = Path("/workspace/datasets")
    settings.finetuned_models_path = Path("/workspace/models/finetuned_models")
    settings.huggingface_token = "test_token"
    settings.argilla_api_url = "http://localhost:6900"
    settings.argilla_api_key = "test_key"
    settings.lora_r = 128
    settings.lora_alpha = 64
    settings.lora_dropout = 0.05
    settings.num_train_epochs = 5
    settings.learning_rate = 1e-3
    settings.dataset = "test/dataset"

    settings.get_training_config.return_value = {
        "dataset": "test/dataset",
        "base_model": "test/model",
        "model_name": "test_model",
        "lora_r": 128,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "num_train_epochs": 5,
        "learning_rate": 1e-3,
        "output_dir": "/workspace/models/finetuned_models/test_model",
    }

    return settings


@pytest.fixture
def mock_language_model():
    """Mock language model that simulates guidance model behavior."""

    class MockLM:
        def __init__(self):
            self._values = {}

        def __add__(self, other):
            return self

        def __getitem__(self, key):
            defaults = {
                "thought": "I should help the user with their request.",
                "action_choice": "call function",
                "fct_name": "get_weather",
                "fct_parameters": '{"location": "Paris"}',
                "function_output": '{"temp": 20}',
                "final_answer": "The weather in Paris is 20 degrees.",
            }
            return self._values.get(key, defaults.get(key, ""))

        def __setitem__(self, key, value):
            self._values[key] = value

    return MockLM()
