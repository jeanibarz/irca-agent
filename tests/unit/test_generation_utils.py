"""
Tests for Generation Router Utilities

Tests the utility functions in src/server/routers/generation.py.
Covers security-critical math evaluation and helper functions.
"""

import pytest


class TestGenerateOutputId:
    """Tests for generate_output_id function."""

    def test_output_id_format(self) -> None:
        """Output ID should be 12 alphanumeric characters."""
        from src.server.routers.generation import generate_output_id

        output_id = generate_output_id()
        assert len(output_id) == 12
        assert output_id.isalnum()

    def test_output_id_uniqueness(self) -> None:
        """Multiple calls should produce different IDs."""
        from src.server.routers.generation import generate_output_id

        ids = [generate_output_id() for _ in range(100)]
        # All should be unique
        assert len(set(ids)) == 100


class TestSafeMathEval:
    """
    Tests for safe_math_eval function.

    This is security-critical - validates that the eval replacement
    only allows basic arithmetic and rejects dangerous inputs.
    """

    def test_basic_addition(self) -> None:
        """Basic addition works."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("2 + 3") == 5.0

    def test_basic_subtraction(self) -> None:
        """Basic subtraction works."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("10 - 4") == 6.0

    def test_basic_multiplication(self) -> None:
        """Basic multiplication works."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("3 * 4") == 12.0

    def test_basic_division(self) -> None:
        """Basic division works."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("10 / 2") == 5.0

    def test_complex_expression(self) -> None:
        """Complex expressions with parentheses work."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("(1 + 2) * 3") == 9.0
        assert safe_math_eval("10 / 2 * 3") == 15.0

    def test_negative_numbers(self) -> None:
        """Negative numbers are supported."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("-5 + 3") == -2.0
        assert safe_math_eval("5 + -3") == 2.0

    def test_floating_point(self) -> None:
        """Floating point numbers work."""
        from src.server.routers.generation import safe_math_eval

        assert abs(safe_math_eval("1.5 * 2") - 3.0) < 0.001

    def test_empty_expression(self) -> None:
        """Empty expression returns 0."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("") == 0.0
        assert safe_math_eval("   ") == 0.0

    def test_rejects_function_calls(self) -> None:
        """Function calls are rejected (security)."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("__import__('os').system('ls')")

    def test_rejects_attribute_access(self) -> None:
        """Attribute access is rejected (security)."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("x.y")

    def test_rejects_variables(self) -> None:
        """Variable names are rejected."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("x + 1")

    def test_rejects_string_literals(self) -> None:
        """String literals are rejected."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("'hello' + 'world'")

    def test_rejects_list_comprehension(self) -> None:
        """List comprehensions are rejected."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("[x for x in range(10)]")

    def test_rejects_power_operator(self) -> None:
        """Power operator is rejected for safety."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("2 ** 100")

    def test_rejects_bitwise_operators(self) -> None:
        """Bitwise operators are rejected."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("1 | 2")

        with pytest.raises(ValueError):
            safe_math_eval("1 & 2")


class TestGenerateMockOutput:
    """Tests for generate_mock_output function."""

    def test_weather_mock(self) -> None:
        """Weather mock returns expected structure."""
        from src.server.routers.generation import generate_mock_output

        result = generate_mock_output("get_weather", {"location": "Paris"})
        assert "temperature" in result
        assert "condition" in result

    def test_stock_mock(self) -> None:
        """Stock price mock returns expected structure."""
        from src.server.routers.generation import generate_mock_output

        result = generate_mock_output("get_stock_price", {"ticker": "AAPL"})
        assert "price" in result
        assert "ticker" in result

    def test_unknown_function_default(self) -> None:
        """Unknown functions get a default mock output."""
        from src.server.routers.generation import generate_mock_output

        result = generate_mock_output("unknown_function", {})
        assert result is not None


class TestModelManagerSingleton:
    """Tests for ModelManager singleton pattern."""

    def test_singleton_returns_same_instance(self) -> None:
        """get_instance returns the same instance."""
        from src.server.model_manager import ModelManager

        # Reset singleton
        ModelManager._instance = None

        instance1 = ModelManager.get_instance()
        instance2 = ModelManager.get_instance()

        assert instance1 is instance2

    def test_singleton_thread_safety(self) -> None:
        """Singleton is thread-safe (FM-03)."""
        import threading

        from src.server.model_manager import ModelManager

        # Reset singleton
        ModelManager._instance = None

        instances = []
        errors = []

        def get_instance():
            try:
                instances.append(ModelManager.get_instance())
            except Exception as e:
                errors.append(e)

        # Create multiple threads to get instance simultaneously
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors
        assert len(errors) == 0

        # All instances are the same
        assert len(instances) == 10
        assert all(inst is instances[0] for inst in instances)

    def test_model_manager_has_required_attributes(self) -> None:
        """ModelManager has expected attributes after init."""
        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Check required attributes
        assert hasattr(manager, "models")
        assert hasattr(manager, "tokenizers")
        assert hasattr(manager, "loaded_configs")
        assert hasattr(manager, "_lock")
        assert hasattr(manager, "_last_activity")

    def test_models_dict_tracks_loaded_models(self) -> None:
        """Models dict tracks loaded models."""
        from unittest.mock import MagicMock

        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()
        manager.models.clear()

        # Initially empty
        assert "default" not in manager.models

        # Add mock model
        manager.models["test"] = MagicMock()
        assert "test" in manager.models

    def test_loaded_configs_tracks_model_configs(self) -> None:
        """Loaded configs dict tracks model configurations."""
        from unittest.mock import MagicMock

        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()
        manager.loaded_configs.clear()

        # Add config
        manager.loaded_configs["default"] = {"base": "test-model", "adapter": "/path/to/adapter"}

        assert "default" in manager.loaded_configs
        assert manager.loaded_configs["default"]["base"] == "test-model"
        assert manager.loaded_configs["default"]["adapter"] == "/path/to/adapter"

    def test_tokenizers_dict_tracks_loaded_tokenizers(self) -> None:
        """Tokenizers dict tracks loaded tokenizers."""
        from unittest.mock import MagicMock

        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()
        manager.tokenizers.clear()

        # Initially empty
        assert "default" not in manager.tokenizers

        # Add mock tokenizer
        manager.tokenizers["test"] = MagicMock()
        assert "test" in manager.tokenizers
