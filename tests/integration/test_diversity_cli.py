"""
Integration tests for diversity CLI command.

Tests FR-DATA-19: Diversity CLI.
Tests FR-DATA-21: Diversity Comparison.
Tests FR-DATA-23: Quick Diversity Mode.
Tests FR-DIV-10: HTML Report Generation.
Tests FR-DIV-12: Report CLI Flag.
Tests FR-DIV-14: Report Comparison Mode.
Tests FR-DIV-15: Report Single Dataset Mode.
Tests NFR-DIV-01: Report Portability.
Tests NFR-DIV-02: Report File Size.
"""

import json
import os
import shutil
import tempfile
import unittest

from click.testing import CliRunner

import datasets
from src.cli.commands.dataset import dataset


class TestDiversityCLIQuickMode(unittest.TestCase):
    """Tests for diversity CLI in quick mode (no model required)."""

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.test_dir, "test_dataset")

        # Create sample dataset
        data = {"text": [f"Sample text number {i} with some content." for i in range(20)]}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.dataset_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_quick_mode_single_dataset(self):
        """Quick mode works for single dataset analysis."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path, "--quick"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
        self.assertIn("Lexical Diversity", result.output)
        self.assertIn("Distinct-1", result.output)
        self.assertIn("Distinct-2", result.output)

    def test_quick_mode_json_output(self):
        """Quick mode with JSON output."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path, "--quick", "--output", "json"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

        # Parse JSON output
        output_json = json.loads(result.output)
        self.assertIn("lexical", output_json)
        self.assertIn("distinct_1", output_json["lexical"])
        self.assertIn("sample_count", output_json)

    def test_quick_mode_requires_dataset(self):
        """Quick mode requires dataset argument."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "--quick"],
            obj={"verbose": False},
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(
            "--dataset",
            result.output.lower() + result.exception.__str__().lower() if result.exception else result.output.lower(),
        )


class TestDiversityCLIComparison(unittest.TestCase):
    """Tests for diversity CLI comparison mode."""

    def setUp(self):
        """Create test directory and sample datasets."""
        self.test_dir = tempfile.mkdtemp()
        self.original_path = os.path.join(self.test_dir, "original_dataset")
        self.augmented_path = os.path.join(self.test_dir, "augmented_dataset")

        # Create original dataset
        original_data = {"text": [f"Original text {i}." for i in range(10)]}
        original_ds = datasets.Dataset.from_dict(original_data)
        original_ds.save_to_disk(self.original_path)

        # Create augmented dataset (larger, more diverse)
        augmented_data = {
            "text": [f"Original text {i}." for i in range(10)]
            + [f"Augmented sample {i} with different content." for i in range(20)]
        }
        augmented_ds = datasets.Dataset.from_dict(augmented_data)
        augmented_ds.save_to_disk(self.augmented_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_comparison_mode_quick(self):
        """Comparison mode works in quick mode."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "diversity",
                "-o",
                self.original_path,
                "-a",
                self.augmented_path,
                "--quick",
            ],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
        self.assertIn("Original", result.output)
        self.assertIn("Augmented", result.output)
        self.assertIn("Change", result.output)

    def test_comparison_mode_json(self):
        """Comparison mode with JSON output."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "diversity",
                "-o",
                self.original_path,
                "-a",
                self.augmented_path,
                "--quick",
                "--output",
                "json",
            ],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

        output_json = json.loads(result.output)
        self.assertIn("original", output_json)
        self.assertIn("augmented", output_json)
        self.assertIn("delta", output_json)

    def test_comparison_shows_size_change(self):
        """Comparison mode shows dataset size change."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "diversity",
                "-o",
                self.original_path,
                "-a",
                self.augmented_path,
                "--quick",
            ],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        # Should show 10 -> 30 samples
        self.assertIn("10", result.output)
        self.assertIn("30", result.output)


class TestDiversityCLIErrors(unittest.TestCase):
    """Tests for diversity CLI error handling."""

    def setUp(self):
        """Create test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.test_dir, "test_dataset")

        # Create sample dataset
        data = {"text": ["Sample text."]}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.dataset_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_model_required_without_quick(self):
        """Model is required when not in quick mode."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path],
            obj={"verbose": False},
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--model", result.output)

    def test_invalid_dataset_path(self):
        """Error for non-existent dataset."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", "/nonexistent/path", "--quick"],
            obj={"verbose": False},
        )

        self.assertNotEqual(result.exit_code, 0)
        # Error message could be "not found" or "neither a `dataset` directory"
        output_lower = result.output.lower()
        self.assertTrue(
            "not found" in output_lower or "neither a" in output_lower or "error" in output_lower,
            f"Expected error message, got: {result.output}",
        )

    def test_cannot_mix_modes(self):
        """Cannot use --dataset with --original/--augmented."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "diversity",
                "-d",
                self.dataset_path,
                "-o",
                self.dataset_path,
                "-a",
                self.dataset_path,
                "--quick",
            ],
            obj={"verbose": False},
        )

        self.assertNotEqual(result.exit_code, 0)

    def test_missing_text_column(self):
        """Error when dataset missing text column."""
        # Create dataset without text column
        bad_path = os.path.join(self.test_dir, "bad_dataset")
        bad_data = {"content": ["Sample."]}  # Not "text"
        bad_ds = datasets.Dataset.from_dict(bad_data)
        bad_ds.save_to_disk(bad_path)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", bad_path, "--quick"],
            obj={"verbose": False},
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("text", result.output.lower())


class TestDiversityCLIMetrics(unittest.TestCase):
    """Tests for diversity metric correctness via CLI."""

    def setUp(self):
        """Create test datasets with known diversity patterns."""
        self.test_dir = tempfile.mkdtemp()

        # Low diversity dataset (repetitive)
        self.low_diversity_path = os.path.join(self.test_dir, "low_diversity")
        low_data = {"text": ["hello world"] * 20}
        low_ds = datasets.Dataset.from_dict(low_data)
        low_ds.save_to_disk(self.low_diversity_path)

        # High diversity dataset (unique)
        self.high_diversity_path = os.path.join(self.test_dir, "high_diversity")
        high_data = {"text": [f"unique sentence number {i} with varied vocabulary word{i}" for i in range(20)]}
        high_ds = datasets.Dataset.from_dict(high_data)
        high_ds.save_to_disk(self.high_diversity_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_low_diversity_detected(self):
        """Low diversity dataset has low distinct-n scores."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.low_diversity_path, "--quick", "--output", "json"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        output = json.loads(result.output)

        # Repetitive text should have very low distinct-n
        self.assertLess(output["lexical"]["distinct_2"], 0.1)

    def test_high_diversity_detected(self):
        """High diversity dataset has higher distinct-n scores."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.high_diversity_path, "--quick", "--output", "json"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        output = json.loads(result.output)

        # Diverse text should have higher distinct-n (threshold 0.4 for bigrams)
        self.assertGreater(output["lexical"]["distinct_2"], 0.4)

    def test_comparison_shows_improvement(self):
        """Comparison between low and high diversity shows improvement."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "diversity",
                "-o",
                self.low_diversity_path,
                "-a",
                self.high_diversity_path,
                "--quick",
                "--output",
                "json",
            ],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        output = json.loads(result.output)

        # Augmented (high diversity) should have better metrics
        orig_d2 = output["original"]["lexical"]["distinct_2"]
        aug_d2 = output["augmented"]["lexical"]["distinct_2"]
        self.assertGreater(aug_d2, orig_d2)


class TestDiversityCLIReport(unittest.TestCase):
    """Tests for diversity CLI report generation (FR-DIV-10, FR-DIV-12)."""

    def setUp(self):
        """Create test directory and sample datasets."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.test_dir, "test_dataset")
        self.original_path = os.path.join(self.test_dir, "original_dataset")
        self.augmented_path = os.path.join(self.test_dir, "augmented_dataset")
        self.report_path = os.path.join(self.test_dir, "report.html")

        # Create sample dataset for single mode
        data = {"text": [f"Sample text number {i} with some content." for i in range(20)]}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.dataset_path)

        # Create original dataset for comparison
        original_data = {"text": [f"Original text {i}." for i in range(10)]}
        original_ds = datasets.Dataset.from_dict(original_data)
        original_ds.save_to_disk(self.original_path)

        # Create augmented dataset (larger, more diverse)
        augmented_data = {
            "text": [f"Original text {i}." for i in range(10)]
            + [f"Augmented sample {i} with different content." for i in range(20)]
        }
        augmented_ds = datasets.Dataset.from_dict(augmented_data)
        augmented_ds.save_to_disk(self.augmented_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_report_flag_creates_file(self):
        """--report flag creates HTML report file (FR-DIV-10, FR-DIV-12)."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path, "--quick", "--report", self.report_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
        self.assertTrue(os.path.exists(self.report_path), "Report file not created")
        self.assertIn("Report generated", result.output)

    def test_report_single_mode_content(self):
        """Single dataset report contains expected content (FR-DIV-15)."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path, "--quick", "--report", self.report_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        # Check report content
        with open(self.report_path, encoding="utf-8") as f:
            content = f.read()

        # Verify HTML structure
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("<html", content)
        self.assertIn("</html>", content)

        # Verify metrics data is embedded
        self.assertIn("distinct_1", content)
        self.assertIn("distinct_2", content)
        self.assertIn("reportData", content)

    def test_report_comparison_mode_content(self):
        """Comparison mode report contains both datasets (FR-DIV-14)."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "diversity",
                "-o",
                self.original_path,
                "-a",
                self.augmented_path,
                "--quick",
                "--report",
                self.report_path,
            ],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

        with open(self.report_path, encoding="utf-8") as f:
            content = f.read()

        # Verify comparison data
        self.assertIn("original", content.lower())
        self.assertIn("augmented", content.lower())
        self.assertIn("comparison_mode", content)

    def test_report_self_contained(self):
        """Report is self-contained with inline styles (NFR-DIV-01)."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path, "--quick", "--report", self.report_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        with open(self.report_path, encoding="utf-8") as f:
            content = f.read()

        # Should have inline styles, not external stylesheet links
        self.assertIn("<style>", content)
        self.assertIn("</style>", content)
        # Should not reference external local stylesheets
        self.assertNotIn('href="./', content)
        self.assertNotIn("href='./", content)

    def test_report_file_size_limit(self):
        """Report file size is within acceptable limits (NFR-DIV-02)."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "diversity",
                "-o",
                self.original_path,
                "-a",
                self.augmented_path,
                "--quick",
                "--report",
                self.report_path,
            ],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        # Check file size (should be under 500KB)
        file_size = os.path.getsize(self.report_path)
        max_size = 500 * 1024  # 500KB
        self.assertLess(file_size, max_size, f"Report too large: {file_size / 1024:.1f}KB > 500KB")

    def test_report_contains_chartjs(self):
        """Report includes Chart.js for visualizations (FR-DIV-11)."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path, "--quick", "--report", self.report_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        with open(self.report_path, encoding="utf-8") as f:
            content = f.read()

        # Should reference Chart.js (either CDN or inline)
        self.assertTrue("chart.js" in content.lower() or "Chart" in content, "Report missing Chart.js reference")

    def test_report_includes_metadata(self):
        """Report includes metadata section (FR-DIV-13)."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "diversity",
                "-o",
                self.original_path,
                "-a",
                self.augmented_path,
                "--quick",
                "--report",
                self.report_path,
            ],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        with open(self.report_path, encoding="utf-8") as f:
            content = f.read()

        # Should contain metadata
        self.assertIn("generated_at", content)
        self.assertIn("metadata", content)

    def test_report_with_json_output(self):
        """Report can be generated alongside JSON output."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "diversity",
                "-d",
                self.dataset_path,
                "--quick",
                "--output",
                "json",
                "--report",
                self.report_path,
            ],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        # Both JSON output and report should be generated
        self.assertTrue(os.path.exists(self.report_path))

        # Extract JSON portion from output (ends at closing brace)
        # The output may have "Report generated: ..." after the JSON
        output_lines = result.output.strip().split("\n")
        json_lines = []
        brace_count = 0
        for line in output_lines:
            json_lines.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count == 0 and json_lines:
                break

        json_str = "\n".join(json_lines)
        output_json = json.loads(json_str)
        self.assertIn("lexical", output_json)

    def test_report_path_shorthand(self):
        """Report can be generated using -r shorthand."""
        short_report_path = os.path.join(self.test_dir, "short_report.html")
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path, "--quick", "-r", short_report_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
        self.assertTrue(os.path.exists(short_report_path), "Report file not created with -r")

    def test_report_valid_html(self):
        """Report contains valid HTML structure."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path, "--quick", "--report", self.report_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        with open(self.report_path, encoding="utf-8") as f:
            content = f.read()

        # Check essential HTML elements
        self.assertIn("<head>", content)
        self.assertIn("</head>", content)
        self.assertIn("<body>", content)
        self.assertIn("</body>", content)
        self.assertIn("<script>", content)
        self.assertIn("</script>", content)

    def test_report_dark_mode_support(self):
        """Report includes dark mode CSS support (NFR-DIV-03)."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.dataset_path, "--quick", "--report", self.report_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        with open(self.report_path, encoding="utf-8") as f:
            content = f.read()

        # Check for dark mode media query
        self.assertIn("prefers-color-scheme", content)


if __name__ == "__main__":
    unittest.main()
