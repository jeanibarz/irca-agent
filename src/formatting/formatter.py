"""
Dataset formatter for converting raw structured data to training-ready text.

Implements FR-DATA-24: Dataset Format Command.
Implements FR-DATA-25: Staged Augmentation.
Implements FR-DATA-26: Format Baseline.
"""

import hashlib
import json
import logging
import random
from collections import defaultdict
from collections.abc import Generator
from typing import Any

from datasets import Dataset
from src.formatting.config import FormatConfig
from src.formatting.steps import create_step

logger = logging.getLogger(__name__)


class DatasetFormatter:
    """
    Formats raw structured datasets to training-ready text format.

    The formatting process has three stages:
    1. Structural: Augmentations on parsed components (before formatting)
    2. Formatting: Augmentations during text construction
    3. Presentation: Augmentations on final text

    Implements ADR-003: Dataset Format Pipeline.
    """

    def __init__(self, config: FormatConfig):
        """
        Initialize the formatter.

        Args:
            config: Format configuration
        """
        self.config = config
        self.rng = random.Random(config.seed)
        self.statistics = {
            "input_samples": 0,
            "output_samples": 0,
            "variants_generated": 0,
            "duplicates_removed": 0,
            "augmentation_counts": defaultdict(int),
            "parse_errors": 0,
        }

    def format(self, dataset: Dataset) -> tuple[Dataset, dict[str, Any]]:
        """
        Format the dataset according to configuration.

        Args:
            dataset: Input HuggingFace Dataset with structured columns

        Returns:
            Tuple of (formatted dataset with 'text' column, statistics dict)
        """
        self.statistics["input_samples"] = len(dataset)

        # Generate all formatted samples (with variants if multiply > 1)
        all_samples = list(self._generate_formatted_samples(dataset))
        self.statistics["variants_generated"] = len(all_samples)

        # Deduplicate if enabled and multiplying
        if self.config.deduplicate and self.config.multiply > 1:
            all_samples = self._deduplicate(all_samples)

        self.statistics["output_samples"] = len(all_samples)

        # Convert to Dataset
        output_dataset = Dataset.from_list(all_samples)

        return output_dataset, dict(self.statistics)

    def _generate_formatted_samples(self, dataset: Dataset) -> Generator[dict[str, Any], None, None]:
        """
        Generate formatted samples from the dataset.

        Yields formatted samples, potentially multiple variants per input sample.
        """
        for idx, sample in enumerate(dataset):
            sample_dict = dict(sample)

            # Generate variants (1 if no multiplication)
            for variant_id in range(self.config.multiply):
                try:
                    formatted = self._format_single_sample(sample_dict, variant_id)

                    yield {
                        **{k: v for k, v in sample_dict.items() if k != self.config.source_column},
                        "text": formatted["text"],
                        "original_index": idx,
                        "variant_id": variant_id,
                        "augmentations": formatted.get("augmentations", []),
                    }
                except Exception as e:
                    logger.warning(f"Error formatting sample {idx}: {e}")
                    self.statistics["parse_errors"] += 1

                    # For baseline (variant_id=0), still emit with error text
                    if variant_id == 0:
                        yield {
                            **{k: v for k, v in sample_dict.items() if k != self.config.source_column},
                            "text": f"[PARSE ERROR: {e}]",
                            "original_index": idx,
                            "variant_id": variant_id,
                            "augmentations": [],
                        }

    def _format_single_sample(self, sample: dict[str, Any], variant_id: int) -> dict[str, Any]:
        """
        Format a single sample through all stages.

        Args:
            sample: Input sample with structured data
            variant_id: Variant identifier (0 = baseline)

        Returns:
            Dict with 'text' and 'augmentations' keys
        """
        augmentations: list[str] = []

        # Step 1: Parse the source column
        parsed = self._parse_source(sample)

        # Step 2: Apply structural augmentations (on parsed components)
        if variant_id > 0 or self.config.structural:
            for step_config in self.config.structural:
                if self.rng.random() < step_config.probability:
                    step = create_step(step_config, self.rng)
                    parsed = step.apply(parsed)
                    aug_name = step.get_name()
                    augmentations.append(aug_name)
                    self.statistics["augmentation_counts"][aug_name] += 1

        # Step 3: Apply formatting augmentations (during build)
        if variant_id > 0 or self.config.formatting:
            for step_config in self.config.formatting:
                if self.rng.random() < step_config.probability:
                    step = create_step(step_config, self.rng)
                    parsed = step.apply(parsed)
                    aug_name = step.get_name()
                    augmentations.append(aug_name)
                    self.statistics["augmentation_counts"][aug_name] += 1

        # Step 4: Build the formatted text
        text = self._build_text(parsed)

        # Step 5: Apply presentation augmentations (on final text)
        result = {"text": text}
        if variant_id > 0 or self.config.presentation:
            for step_config in self.config.presentation:
                if self.rng.random() < step_config.probability:
                    step = create_step(step_config, self.rng)
                    result = step.apply(result)
                    aug_name = step.get_name()
                    augmentations.append(aug_name)
                    self.statistics["augmentation_counts"][aug_name] += 1

        return {"text": result["text"], "augmentations": augmentations}

    def _parse_source(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Parse the source column into components.

        Args:
            sample: Input sample with source column

        Returns:
            Dict with parsed components:
            - system_instructions
            - example
            - available_functions (as list)
            - available_functions_json (as string)
            - user_query
            - assistant_completion
            - final_answer (extracted from completion if present)
        """
        from src.core.prompt_builder import parse_corrected_agent_trace

        source_col = self.config.source_column
        raw_data = sample.get(source_col)

        if raw_data is None:
            raise ValueError(f"Source column '{source_col}' not found in sample")

        # Handle different source formats
        if isinstance(raw_data, list) and len(raw_data) > 0:
            # Format: [{"value": "...", ...}]
            if isinstance(raw_data[0], dict) and "value" in raw_data[0]:
                full_prompt = raw_data[0]["value"]
            else:
                full_prompt = str(raw_data[0])
        elif isinstance(raw_data, str):
            full_prompt = raw_data
        else:
            raise ValueError(f"Unsupported source column format: {type(raw_data)}")

        # Normalize newlines
        full_prompt = full_prompt.replace("\r\n", "\n")

        # Parse into components
        parsed = parse_corrected_agent_trace(full_prompt)

        # Parse functions JSON into list
        functions_json = parsed.get("available_functions_json", "")
        try:
            if functions_json.strip():
                parsed["available_functions"] = json.loads(functions_json)
            else:
                parsed["available_functions"] = []
        except json.JSONDecodeError:
            parsed["available_functions"] = []

        # Extract final answer from completion if present
        completion = parsed.get("assistant_completion", "")
        final_answer_marker = "### FINAL ANSWER"
        if final_answer_marker in completion:
            parts = completion.rsplit(final_answer_marker, 1)
            if len(parts) > 1:
                parsed["final_answer"] = parts[1].strip()
                parsed["completion_before_answer"] = parts[0]

        return parsed

    def _build_text(self, parsed: dict[str, Any]) -> str:
        """
        Build formatted text from parsed components.

        Args:
            parsed: Dict with parsed components

        Returns:
            Formatted prompt string
        """
        # Get JSON indent preference (set by IndentFunctionsStep)
        json_indent = parsed.get("_json_indent")

        # Serialize functions
        functions = parsed.get("available_functions", [])
        if isinstance(functions, list):
            available_functions_json = json.dumps(functions, indent=json_indent)
        else:
            available_functions_json = parsed.get("available_functions_json", "[]")

        # Build the completion (may have modified final answer)
        completion = parsed.get("assistant_completion", "")
        if "final_answer" in parsed and "completion_before_answer" in parsed:
            # Reconstruct with potentially translated final answer
            completion = parsed["completion_before_answer"] + "### FINAL ANSWER\n" + parsed["final_answer"]

        # Build full prompt using the standard format
        full_prompt = f"""
### INSTRUCTIONS
{parsed.get("system_instructions", "")}

EXAMPLE:
{parsed.get("example", "")}

The functions available to you are described below.

### FUNCTIONS AVAILABLE
{available_functions_json}

### USER QUERY
{parsed.get("user_query", "")}

### ITERATIVE RESOLUTION CYCLE
{completion}
"""
        return full_prompt.strip()

    def _deduplicate(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Remove duplicate variants within each original sample.

        Args:
            samples: List of formatted samples

        Returns:
            Deduplicated list
        """
        seen_per_original: dict[int, set[str]] = defaultdict(set)
        deduplicated = []

        for sample in samples:
            text = sample.get("text", "")
            text_hash = hashlib.md5(text.encode()).hexdigest()
            original_idx = sample.get("original_index", 0)

            if text_hash in seen_per_original[original_idx]:
                self.statistics["duplicates_removed"] += 1
                continue

            seen_per_original[original_idx].add(text_hash)
            deduplicated.append(sample)

        return deduplicated

    def get_statistics(self) -> dict[str, Any]:
        """Get current statistics."""
        return dict(self.statistics)


def format_dataset(
    dataset: Dataset,
    config: FormatConfig | None = None,
    no_augment: bool = False,
) -> tuple[Dataset, dict[str, Any]]:
    """
    Convenience function to format a dataset.

    Args:
        dataset: Input dataset with structured columns
        config: Format configuration (None for baseline)
        no_augment: If True, ignore all augmentations in config

    Returns:
        Tuple of (formatted dataset, statistics)
    """
    from src.formatting.config import create_baseline_config

    if config is None or no_augment:
        effective_config = create_baseline_config()
        if config is not None:
            # Keep non-augmentation settings
            effective_config.seed = config.seed
            effective_config.source_column = config.source_column
    else:
        effective_config = config

    formatter = DatasetFormatter(effective_config)
    return formatter.format(dataset)
