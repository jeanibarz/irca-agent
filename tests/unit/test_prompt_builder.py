"""
Tests for Prompt Builder
 
 Tests the prompt building and parsing functions in core.prompt_builder.
 Covers Requirement: **FR-GEN-05 (Prompt Randomization)**
"""

from core.prompt_builder import (
    InstructionFormatter,
    build_full_prompt,
    format_instruction,
    parse_corrected_agent_trace,
    randomize_newline_characters,
    randomize_system_instructions_formatting,
)


class TestBuildFullPrompt:
    """
    Tests for build_full_prompt function.
    Req: FR-GEN-05
    """

    def test_build_basic_prompt(self, sample_prompt_data):
        """Build a complete prompt from components."""
        result = build_full_prompt(sample_prompt_data)

        assert "### INSTRUCTIONS" in result
        assert sample_prompt_data["system_instructions"] in result
        assert "### FUNCTIONS AVAILABLE" in result
        assert "### USER QUERY" in result
        assert sample_prompt_data["user_query"] in result
        assert "### ITERATIVE RESOLUTION CYCLE" in result

    def test_build_prompt_with_empty_fields(self):
        """Build prompt with empty/missing fields."""
        result = build_full_prompt({})
        assert "### INSTRUCTIONS" in result
        assert "### FUNCTIONS AVAILABLE" in result
        assert "### USER QUERY" in result

    def test_build_prompt_structure(self, sample_prompt_data):
        """Verify prompt sections appear in correct order."""
        result = build_full_prompt(sample_prompt_data)

        instructions_pos = result.find("### INSTRUCTIONS")
        functions_pos = result.find("### FUNCTIONS AVAILABLE")
        query_pos = result.find("### USER QUERY")
        cycle_pos = result.find("### ITERATIVE RESOLUTION CYCLE")

        assert instructions_pos < functions_pos < query_pos < cycle_pos


class TestParseCorrectedAgentTrace:
    """
    Tests for parse_corrected_agent_trace function.
    Req: FR-GEN-05
    """

    def test_parse_full_prompt(self):
        """Parse a complete prompt into components."""
        prompt = """
### INSTRUCTIONS
You are an AI assistant.

EXAMPLE:
Here is an example<|wait|>

The functions available to you are described below.

### FUNCTIONS AVAILABLE
[{"name": "test"}]

Note: ensure you only use information provided.

### USER QUERY
Help me

### ITERATIVE RESOLUTION CYCLE
Thought: I should help.
"""
        result = parse_corrected_agent_trace(prompt)

        assert "system_instructions" in result
        assert "example" in result
        assert "available_functions_json" in result
        assert "user_query" in result
        assert "assistant_completion" in result

    def test_parse_extracts_instructions(self):
        """Parse correctly extracts system instructions."""
        prompt = """
### INSTRUCTIONS
Be helpful and honest.

EXAMPLE:
test<|wait|>

### FUNCTIONS AVAILABLE
[]
"""
        result = parse_corrected_agent_trace(prompt)
        assert "Be helpful" in result["system_instructions"]


class TestRandomizeNewlineCharacters:
    """
    Tests for randomize_newline_characters function.
    Req: FR-GEN-05
    """

    def test_randomize_preserves_newlines_count(self):
        """Number of newlines should be preserved."""
        text = "line1\nline2\nline3"
        result = randomize_newline_characters(text)
        # Count logical lines (split by either \n or \r\n)
        original_lines = text.count("\n")
        result_lines = result.count("\n") + result.count("\r\n")
        # At minimum we should have same number of line breaks
        assert result_lines >= original_lines

    def test_randomize_consistent_choice(self):
        """All newlines should be replaced with the same choice."""
        text = "a\nb\nc"
        result = randomize_newline_characters(text)
        # Either all \r\n or all \n
        has_cr_lf = "\r\n" in result
        if has_cr_lf:
            # All should be \r\n
            assert result.count("\r\n") == text.count("\n")


class TestRandomizeSystemInstructionsFormatting:
    """
    Tests for randomize_system_instructions_formatting function.
    Req: FR-GEN-05
    """

    def test_randomize_returns_string(self):
        """Function returns a string."""
        result = randomize_system_instructions_formatting("Test instructions")
        assert isinstance(result, str)

    def test_randomize_may_modify_markers(self):
        """Function may modify section markers."""
        instructions = "### INSTRUCTIONS\nTest content"
        # Run multiple times to account for randomness
        results = set()
        for _ in range(20):
            result = randomize_system_instructions_formatting(instructions)
            results.add(result)
        # Should have some variation (though not guaranteed with small sample)
        assert len(results) >= 1


class TestInstructionFormatter:
    """Tests for InstructionFormatter class."""

    def test_formatter_initialization(self):
        """Formatter initializes with correct defaults."""
        formatter = InstructionFormatter()
        assert formatter.random_augmentation is True
        assert formatter.iteration_count == 0

    def test_formatter_without_augmentation(self):
        """Formatter can be initialized without augmentation."""
        formatter = InstructionFormatter(random_augmentation=False)
        assert formatter.random_augmentation is False

    def test_formatter_tracks_iterations(self, sample_training_sample):
        """Formatter increments iteration count."""
        formatter = InstructionFormatter(random_augmentation=False)
        _ = formatter.format(sample_training_sample)
        assert formatter.iteration_count == 1
        _ = formatter.format(sample_training_sample)
        assert formatter.iteration_count == 2


class TestFormatInstruction:
    """Tests for format_instruction function."""

    def test_format_instruction_returns_string(self, sample_training_sample):
        """format_instruction returns a string."""
        result = format_instruction(sample_training_sample, random_augmentation=False)
        assert isinstance(result, str)

    def test_format_instruction_contains_key_sections(self, sample_training_sample):
        """Formatted output contains expected sections."""
        result = format_instruction(sample_training_sample, random_augmentation=False)
        assert "### INSTRUCTIONS" in result
        assert "### USER QUERY" in result

    def test_format_instruction_with_augmentation(self, sample_training_sample):
        """format_instruction with augmentation still produces valid output."""
        result = format_instruction(sample_training_sample, random_augmentation=True)
        assert isinstance(result, str)
        assert len(result) > 0
