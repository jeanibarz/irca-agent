"""
Augmentation pipeline executor.

Implements FR-DATA-11, FR-DATA-13, FR-DATA-14.
"""

import hashlib
import random
from collections import defaultdict
from collections.abc import Generator
from typing import Any

from datasets import Dataset
from src.augmentation.config import AugmentationConfig
from src.augmentation.steps import create_step


class AugmentationPipeline:
    """
    Pipeline-based dataset augmentation.

    Implements:
    - FR-DATA-11: Dataset Size Multiplication
    - FR-DATA-13: Feature Combination
    - FR-DATA-14: Post-Generation Deduplication
    """

    def __init__(self, config: AugmentationConfig):
        """
        Initialize the augmentation pipeline.

        Args:
            config: Validated augmentation configuration
        """
        self.config = config
        self.rng = random.Random(config.seed)
        self.statistics = {
            "input_samples": 0,
            "variants_generated": 0,
            "duplicates_removed": 0,
            "augmentation_counts": defaultdict(int),
            "empty_variants": 0,
        }

    def augment(self, dataset: Dataset) -> tuple[Dataset, dict[str, Any]]:
        """
        Augment the dataset according to the configuration.

        Args:
            dataset: Input HuggingFace Dataset

        Returns:
            Tuple of (augmented dataset, statistics dictionary)
        """
        self.statistics["input_samples"] = len(dataset)

        # Phase 1: Generate all variants
        all_samples = list(self._generate_variants(dataset))
        self.statistics["variants_generated"] = len(all_samples)

        # Phase 2: Deduplicate if enabled
        if self.config.deduplicate:
            all_samples = self._deduplicate(all_samples)

        # Convert to Dataset
        output_dataset = Dataset.from_list(all_samples)

        return output_dataset, dict(self.statistics)

    def _generate_variants(self, dataset: Dataset) -> Generator[dict[str, Any], None, None]:
        """
        Generate all variants for the dataset.

        Yields original samples (variant_id=0) and augmented variants.
        """
        for idx, sample in enumerate(dataset):
            # Convert to dict if needed
            sample_dict = dict(sample)

            # Always emit original (variant_id=0)
            yield {
                **sample_dict,
                "original_index": idx,
                "variant_id": 0,
                "augmentations": [],
            }

            # Generate (multiply - 1) variants
            for variant_id in range(1, self.config.multiply):
                augmented, applied = self._apply_pipeline(sample_dict)

                if not applied:
                    self.statistics["empty_variants"] += 1

                # Track augmentation counts
                for aug_name in applied:
                    self.statistics["augmentation_counts"][aug_name] += 1

                yield {
                    **augmented,
                    "original_index": idx,
                    "variant_id": variant_id,
                    "augmentations": applied,
                }

    def _apply_pipeline(self, sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """
        Apply the augmentation pipeline to a sample.

        Args:
            sample: Input sample dictionary

        Returns:
            Tuple of (augmented sample, list of applied augmentation names)
        """
        augmented = sample.copy()
        applied = []

        for step_config in self.config.pipeline:
            # Roll against step probability
            if self.rng.random() < step_config.probability:
                step = create_step(step_config, self.rng)
                augmented = step.apply(augmented)
                applied.append(step.get_name())

        return augmented, applied

    def _deduplicate(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Remove duplicate variants within each original sample.

        Implements FR-DATA-14: Post-Generation Deduplication.
        """
        seen_per_original: dict[int, set[str]] = defaultdict(set)
        deduplicated = []

        for sample in samples:
            text = sample.get("text", "")
            text_hash = hashlib.md5(text.encode()).hexdigest()
            original_idx = sample["original_index"]

            if text_hash in seen_per_original[original_idx]:
                self.statistics["duplicates_removed"] += 1
                continue

            seen_per_original[original_idx].add(text_hash)
            deduplicated.append(sample)

        return deduplicated

    def get_statistics(self) -> dict[str, Any]:
        """Get current statistics."""
        return dict(self.statistics)
