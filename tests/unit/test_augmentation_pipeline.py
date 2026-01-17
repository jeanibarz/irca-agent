"""
Unit tests for augmentation pipeline.

Tests FR-DATA-11, FR-DATA-13, FR-DATA-14.
"""

import os
import sys
import unittest

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from augmentation.config import AugmentationConfig, PipelineStep
from augmentation.pipeline import AugmentationPipeline
from datasets import Dataset


class TestDatasetMultiplication(unittest.TestCase):
    """
    Tests for FR-DATA-11: Dataset Size Multiplication.
    """

    def setUp(self):
        """Create a simple test dataset."""
        self.test_data = {
            "text": [f"Sample text {i}" for i in range(10)],
            "label": list(range(10)),
        }
        self.dataset = Dataset.from_dict(self.test_data)

    def test_multiply_by_two(self):
        """Dataset size is doubled with multiply=2."""
        config = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,  # Disable to get exact count
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.0),  # No-op
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, stats = pipeline.augment(self.dataset)

        # 10 original + 10 variants = 20
        self.assertEqual(len(result), 20)
        self.assertEqual(stats["input_samples"], 10)

    def test_multiply_by_three(self):
        """Dataset size is tripled with multiply=3."""
        config = AugmentationConfig(
            multiply=3,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.0),
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, stats = pipeline.augment(self.dataset)

        # 10 original + 20 variants = 30
        self.assertEqual(len(result), 30)

    def test_original_samples_preserved(self):
        """Original samples always have variant_id=0."""
        config = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=1.0),
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, _ = pipeline.augment(self.dataset)

        # Check that we have 10 samples with variant_id=0
        originals = [r for r in result if r["variant_id"] == 0]
        self.assertEqual(len(originals), 10)

        # Originals should have empty augmentations
        for original in originals:
            self.assertEqual(original["augmentations"], [])

    def test_variant_ids_sequential(self):
        """Variant IDs are sequential within each original sample."""
        config = AugmentationConfig(
            multiply=3,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.0),
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, _ = pipeline.augment(self.dataset)

        # Group by original_index and check variant_ids
        from collections import defaultdict

        by_original = defaultdict(list)
        for r in result:
            by_original[r["original_index"]].append(r["variant_id"])

        for _original_idx, variant_ids in by_original.items():
            self.assertEqual(sorted(variant_ids), [0, 1, 2])


class TestFeatureCombination(unittest.TestCase):
    """
    Tests for FR-DATA-13: Feature Combination.
    """

    def setUp(self):
        """Create a simple test dataset."""
        self.test_data = {"text": ["Test sample"] * 100}
        self.dataset = Dataset.from_dict(self.test_data)

    def test_multiple_augmentations_applied(self):
        """Multiple augmentations can be applied to the same variant."""
        config = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=1.0),
                PipelineStep(type="newline_variation", probability=1.0),
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, _ = pipeline.augment(self.dataset)

        # Check variants (not originals)
        variants = [r for r in result if r["variant_id"] > 0]

        # With probability=1.0, all variants should have both augmentations
        for variant in variants:
            self.assertEqual(len(variant["augmentations"]), 2)
            self.assertIn("shuffle_functions", variant["augmentations"])
            self.assertIn("newline_variation", variant["augmentations"])

    def test_independent_probabilities(self):
        """Each step has independent probability."""
        config = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.5),
                PipelineStep(type="newline_variation", probability=0.5),
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, stats = pipeline.augment(self.dataset)

        variants = [r for r in result if r["variant_id"] > 0]

        # Count augmentations
        shuffle_count = sum(1 for v in variants if "shuffle_functions" in v["augmentations"])
        newline_count = sum(1 for v in variants if "newline_variation" in v["augmentations"])

        # With 100 samples and 0.5 probability, expect roughly 50 each (allow variance)
        self.assertGreater(shuffle_count, 30)
        self.assertLess(shuffle_count, 70)
        self.assertGreater(newline_count, 30)
        self.assertLess(newline_count, 70)

    def test_augmentations_recorded_in_order(self):
        """Augmentations are recorded in pipeline order."""
        config = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="newline_variation", probability=1.0),
                PipelineStep(type="shuffle_functions", probability=1.0),
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, _ = pipeline.augment(Dataset.from_dict({"text": ["Test"]}))

        variant = [r for r in result if r["variant_id"] > 0][0]
        self.assertEqual(
            variant["augmentations"],
            ["newline_variation", "shuffle_functions"],
        )


class TestDeduplication(unittest.TestCase):
    """
    Tests for FR-DATA-14: Post-Generation Deduplication.
    """

    def test_duplicates_removed(self):
        """Duplicate variants are removed when deduplicate=True."""
        # Create dataset where duplicates are likely
        test_data = {"text": ["Identical text"] * 5}
        dataset = Dataset.from_dict(test_data)

        config = AugmentationConfig(
            multiply=3,
            seed=42,
            deduplicate=True,
            pipeline=[
                # No-op step - all variants will be identical to original
                PipelineStep(type="shuffle_functions", probability=0.0),
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, stats = pipeline.augment(dataset)

        # Without deduplication: 5 * 3 = 15
        # With deduplication: 5 (only originals, variants are duplicates)
        self.assertEqual(len(result), 5)
        self.assertEqual(stats["duplicates_removed"], 10)  # 2 duplicates per sample * 5 samples

    def test_deduplication_disabled(self):
        """Duplicates kept when deduplicate=False."""
        test_data = {"text": ["Identical text"] * 5}
        dataset = Dataset.from_dict(test_data)

        config = AugmentationConfig(
            multiply=3,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.0),
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, stats = pipeline.augment(dataset)

        # All 15 samples kept (5 * 3)
        self.assertEqual(len(result), 15)
        self.assertEqual(stats["duplicates_removed"], 0)

    def test_deduplication_within_sample(self):
        """Deduplication only removes duplicates within same original sample."""
        # Two different original samples
        test_data = {"text": ["Text A", "Text A"]}  # Same text, different samples
        dataset = Dataset.from_dict(test_data)

        config = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=True,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.0),  # No change
            ],
        )
        pipeline = AugmentationPipeline(config)
        result, stats = pipeline.augment(dataset)

        # 2 originals + 0 unique variants (duplicates removed per sample)
        # But cross-sample "duplicates" are kept
        # Sample 0: original + 1 duplicate removed = 1
        # Sample 1: original + 1 duplicate removed = 1
        # Total: 2
        self.assertEqual(len(result), 2)


class TestStatistics(unittest.TestCase):
    """Tests for pipeline statistics."""

    def test_statistics_collected(self):
        """Pipeline collects accurate statistics."""
        test_data = {"text": [f"Sample {i}" for i in range(10)]}
        dataset = Dataset.from_dict(test_data)

        config = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=1.0),
            ],
        )
        pipeline = AugmentationPipeline(config)
        _, stats = pipeline.augment(dataset)

        self.assertEqual(stats["input_samples"], 10)
        self.assertEqual(stats["variants_generated"], 20)  # 10 * 2
        self.assertEqual(stats["augmentation_counts"]["shuffle_functions"], 10)  # 10 variants

    def test_empty_variants_counted(self):
        """Empty variants (no augmentations) are counted."""
        test_data = {"text": ["Sample"] * 10}
        dataset = Dataset.from_dict(test_data)

        config = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.0),  # Never applies
            ],
        )
        pipeline = AugmentationPipeline(config)
        _, stats = pipeline.augment(dataset)

        self.assertEqual(stats["empty_variants"], 10)  # All variants are empty


class TestReproducibility(unittest.TestCase):
    """Tests for reproducibility with seed."""

    def test_same_seed_same_result(self):
        """Same seed produces identical results."""
        test_data = {"text": [f"Sample {i}" for i in range(20)]}
        dataset = Dataset.from_dict(test_data)

        config = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.5),
            ],
        )

        # Run twice with same seed
        pipeline1 = AugmentationPipeline(config)
        result1, _ = pipeline1.augment(dataset)

        pipeline2 = AugmentationPipeline(config)
        result2, _ = pipeline2.augment(dataset)

        # Compare augmentations lists
        augs1 = [r["augmentations"] for r in result1]
        augs2 = [r["augmentations"] for r in result2]
        self.assertEqual(augs1, augs2)

    def test_different_seed_different_result(self):
        """Different seeds produce different results."""
        test_data = {"text": [f"Sample {i}" for i in range(100)]}
        dataset = Dataset.from_dict(test_data)

        config1 = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.5),
            ],
        )

        config2 = AugmentationConfig(
            multiply=2,
            seed=123,
            deduplicate=False,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.5),
            ],
        )

        pipeline1 = AugmentationPipeline(config1)
        result1, _ = pipeline1.augment(dataset)

        pipeline2 = AugmentationPipeline(config2)
        result2, _ = pipeline2.augment(dataset)

        # Compare augmentations - should be different
        augs1 = [r["augmentations"] for r in result1]
        augs2 = [r["augmentations"] for r in result2]
        self.assertNotEqual(augs1, augs2)


if __name__ == "__main__":
    unittest.main()
