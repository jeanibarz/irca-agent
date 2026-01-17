"""
Security Tests for Failure Mode Fixes

Tests for critical and high severity fixes from failure mode audit:
- VG-01: Safe math evaluator (FM-01)
- VG-02: Input validation (FM-13)
- VG-03: Thread-safe singletons (FM-03, FM-04)
"""

import concurrent.futures
import threading

import pytest


class TestSafeMathEvaluator:
    """VG-01: Tests for safe_math_eval function (FM-01 fix)."""

    def test_basic_addition(self):
        """Safe eval handles basic addition."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("2 + 3") == 5.0

    def test_basic_subtraction(self):
        """Safe eval handles basic subtraction."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("10 - 4") == 6.0

    def test_multiplication(self):
        """Safe eval handles multiplication."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("3 * 4") == 12.0

    def test_division(self):
        """Safe eval handles division."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("15 / 3") == 5.0

    def test_complex_expression(self):
        """Safe eval handles complex expressions with parentheses."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("(2 + 3) * 4") == 20.0
        assert safe_math_eval("10 / (2 + 3)") == 2.0

    def test_negative_numbers(self):
        """Safe eval handles negative numbers."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("-5") == -5.0
        assert safe_math_eval("-5 + 10") == 5.0

    def test_floats(self):
        """Safe eval handles floating point numbers."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("2.5 + 2.5") == 5.0
        assert safe_math_eval("1.5 * 2") == 3.0

    def test_empty_expression_returns_zero(self):
        """Safe eval returns 0 for empty expressions."""
        from src.server.routers.generation import safe_math_eval

        assert safe_math_eval("") == 0.0
        assert safe_math_eval("   ") == 0.0

    def test_rejects_function_calls(self):
        """Safe eval rejects function call attempts (security)."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("__import__('os').system('ls')")

    def test_rejects_attribute_access(self):
        """Safe eval rejects attribute access (security)."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("().__class__.__bases__")

    def test_rejects_string_literals(self):
        """Safe eval rejects string literals."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("'hello'")

    def test_rejects_list_comprehensions(self):
        """Safe eval rejects list comprehensions."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("[x for x in range(10)]")

    def test_rejects_lambda(self):
        """Safe eval rejects lambda expressions."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("lambda x: x + 1")

    def test_rejects_exec(self):
        """Safe eval rejects exec-like constructs."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("exec('print(1)')")

    def test_rejects_eval(self):
        """Safe eval rejects nested eval."""
        from src.server.routers.generation import safe_math_eval

        with pytest.raises(ValueError):
            safe_math_eval("eval('1+1')")


class TestInputValidation:
    """VG-02: Tests for API input validation (FM-13 fix)."""

    def test_message_role_validation(self):
        """Message role must be one of allowed values."""
        from pydantic import ValidationError

        from src.server.schemas import Message

        # Valid roles
        Message(role="user", content="hello")
        Message(role="assistant", content="hello")
        Message(role="system", content="hello")
        Message(role="function", content="hello")

        # Invalid role
        with pytest.raises(ValidationError):
            Message(role="invalid", content="hello")

    def test_max_tokens_upper_bound(self):
        """max_tokens must be <= 32768."""
        from pydantic import ValidationError

        from src.server.schemas import GenerationRequest, Message

        messages = [Message(role="user", content="hello")]

        # Valid max_tokens
        req = GenerationRequest(messages=messages, max_tokens=32768)
        assert req.max_tokens == 32768

        # Invalid max_tokens (too high)
        with pytest.raises(ValidationError):
            GenerationRequest(messages=messages, max_tokens=100000)

    def test_max_tokens_lower_bound(self):
        """max_tokens must be >= 1."""
        from pydantic import ValidationError

        from src.server.schemas import GenerationRequest, Message

        messages = [Message(role="user", content="hello")]

        # Valid
        GenerationRequest(messages=messages, max_tokens=1)

        # Invalid (zero)
        with pytest.raises(ValidationError):
            GenerationRequest(messages=messages, max_tokens=0)

        # Invalid (negative)
        with pytest.raises(ValidationError):
            GenerationRequest(messages=messages, max_tokens=-1)

    def test_temperature_bounds(self):
        """temperature must be 0.0-2.0."""
        from pydantic import ValidationError

        from src.server.schemas import GenerationRequest, Message

        messages = [Message(role="user", content="hello")]

        # Valid bounds
        GenerationRequest(messages=messages, temperature=0.0)
        GenerationRequest(messages=messages, temperature=2.0)

        # Invalid
        with pytest.raises(ValidationError):
            GenerationRequest(messages=messages, temperature=-0.1)

        with pytest.raises(ValidationError):
            GenerationRequest(messages=messages, temperature=2.1)

    def test_top_p_bounds(self):
        """top_p must be 0.0-1.0."""
        from pydantic import ValidationError

        from src.server.schemas import GenerationRequest, Message

        messages = [Message(role="user", content="hello")]

        # Valid bounds
        GenerationRequest(messages=messages, top_p=0.0)
        GenerationRequest(messages=messages, top_p=1.0)

        # Invalid
        with pytest.raises(ValidationError):
            GenerationRequest(messages=messages, top_p=-0.1)

        with pytest.raises(ValidationError):
            GenerationRequest(messages=messages, top_p=1.1)

    def test_messages_required(self):
        """At least one message is required."""
        from pydantic import ValidationError

        from src.server.schemas import GenerationRequest

        with pytest.raises(ValidationError):
            GenerationRequest(messages=[])

    def test_content_length_limit(self):
        """Message content has max length."""
        from pydantic import ValidationError

        from src.server.schemas import Message

        # Valid (under limit)
        Message(role="user", content="x" * 100000)

        # Invalid (over limit)
        with pytest.raises(ValidationError):
            Message(role="user", content="x" * 100001)


class TestThreadSafeSingletons:
    """VG-03: Tests for thread-safe singletons (FM-03, FM-04)."""

    def test_settings_singleton_returns_same_instance(self):
        """get_settings returns the same instance on repeated calls."""
        from src.config.settings import clear_settings_cache, get_settings

        # Clear any existing cache
        clear_settings_cache()

        instance1 = get_settings()
        instance2 = get_settings()

        assert instance1 is instance2

    def test_settings_singleton_thread_safe(self):
        """get_settings is thread-safe under concurrent access."""
        from src.config.settings import clear_settings_cache, get_settings

        # Clear any existing cache
        clear_settings_cache()

        instances = []
        errors = []

        def get_instance():
            try:
                instances.append(get_settings())
            except Exception as e:
                errors.append(e)

        # Run 10 concurrent threads
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All instances should be the same object
        assert len(instances) == 10
        first = instances[0]
        for inst in instances[1:]:
            assert inst is first

    def test_event_broadcaster_singleton_returns_same_instance(self):
        """EventBroadcaster.get_instance returns the same instance."""
        from src.server.events import EventBroadcaster

        # Reset singleton for test
        original = EventBroadcaster._instance
        try:
            EventBroadcaster._instance = None

            instance1 = EventBroadcaster.get_instance()
            instance2 = EventBroadcaster.get_instance()

            assert instance1 is instance2
        finally:
            EventBroadcaster._instance = original

    def test_event_broadcaster_singleton_has_lock(self):
        """EventBroadcaster has _instance_lock for thread safety."""
        from src.server.events import EventBroadcaster

        assert hasattr(EventBroadcaster, "_instance_lock")
        assert isinstance(EventBroadcaster._instance_lock, type(threading.Lock()))

    def test_event_broadcaster_thread_safe(self):
        """EventBroadcaster.get_instance is thread-safe under concurrent access."""
        from src.server.events import EventBroadcaster

        # Reset singleton for test
        original = EventBroadcaster._instance
        try:
            EventBroadcaster._instance = None

            instances = []
            errors = []

            def get_instance():
                try:
                    instances.append(EventBroadcaster.get_instance())
                except Exception as e:
                    errors.append(e)

            # Run 10 concurrent threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(get_instance) for _ in range(10)]
                concurrent.futures.wait(futures)

            # No errors should occur
            assert len(errors) == 0, f"Errors occurred: {errors}"

            # All instances should be the same object
            assert len(instances) == 10
            first = instances[0]
            for inst in instances[1:]:
                assert inst is first
        finally:
            EventBroadcaster._instance = original


class TestGenerationTimeout:
    """VG-04: Tests for generation timeout (FM-17 fix)."""

    def test_generation_timeout_setting_exists(self):
        """Settings has generation_timeout_seconds field."""
        from src.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()

        assert hasattr(settings, "generation_timeout_seconds")
        assert settings.generation_timeout_seconds >= 10
        assert settings.generation_timeout_seconds <= 600

    def test_generation_timeout_default_value(self):
        """Default generation timeout is 120 seconds."""
        from src.config.settings import clear_settings_cache, get_settings

        clear_settings_cache()
        settings = get_settings()

        assert settings.generation_timeout_seconds == 120

    def test_generation_timeout_setting_bounds(self):
        """generation_timeout_seconds has proper bounds validation."""
        from pydantic import ValidationError

        from src.config.settings import Settings

        # Valid bounds
        Settings(generation_timeout_seconds=10)
        Settings(generation_timeout_seconds=600)

        # Invalid (too low)
        with pytest.raises(ValidationError):
            Settings(generation_timeout_seconds=5)

        # Invalid (too high)
        with pytest.raises(ValidationError):
            Settings(generation_timeout_seconds=1000)
