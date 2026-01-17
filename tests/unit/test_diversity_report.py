"""
Unit tests for diversity report generation.

Tests FR-DIV-10: HTML Report Generation.
Tests FR-DIV-11: Report Visualizations.
Tests NFR-DIV-01: Report Portability.
"""

import json
import tempfile
from pathlib import Path

from src.diversity.report import (
    _compute_histogram_bins,
    _escape_html,
    _prepare_report_data,
    generate_report,
)


class TestGenerateReport:
    """Tests for the main report generation function."""

    def test_generate_report_creates_file(self):
        """Report generation creates an HTML file."""
        metrics = {
            "original": {
                "sample_count": 100,
                "lexical": {
                    "distinct_1": 0.85,
                    "distinct_2": 0.42,
                    "distinct_3": 0.28,
                    "type_token_ratio": 0.32,
                    "total_tokens": 5000,
                },
            },
            "augmented": {
                "sample_count": 300,
                "lexical": {
                    "distinct_1": 0.91,
                    "distinct_2": 0.58,
                    "distinct_3": 0.39,
                    "type_token_ratio": 0.28,
                    "total_tokens": 15000,
                },
            },
            "delta": {
                "sample_count_change": "+200.0%",
                "distinct_1_change": "+7.1%",
                "distinct_2_change": "+38.1%",
                "distinct_3_change": "+39.3%",
            },
            "summary": "Test summary",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            result = generate_report(metrics, output_path)

            assert result == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_generate_report_html_structure(self):
        """Generated report has valid HTML structure."""
        metrics = {
            "dataset": "test-dataset",
            "sample_count": 100,
            "lexical": {
                "distinct_1": 0.85,
                "distinct_2": 0.42,
                "distinct_3": 0.28,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_report(metrics, output_path)

            content = output_path.read_text()

            # Check HTML structure
            assert "<!DOCTYPE html>" in content
            assert "<html" in content
            assert "</html>" in content
            assert "<head>" in content
            assert "</head>" in content
            assert "<body>" in content
            assert "</body>" in content

    def test_generate_report_contains_chartjs(self):
        """Generated report includes Chart.js."""
        metrics = {"dataset": "test", "sample_count": 10, "lexical": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_report(metrics, output_path)

            content = output_path.read_text()

            # Check Chart.js is included
            assert "chart.js" in content.lower() or "Chart" in content

    def test_generate_report_contains_styles(self):
        """Generated report includes CSS styles."""
        metrics = {"dataset": "test", "sample_count": 10, "lexical": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_report(metrics, output_path)

            content = output_path.read_text()

            # Check CSS is included
            assert "<style>" in content
            assert "</style>" in content

    def test_generate_report_self_contained(self):
        """Generated report is self-contained (no external file references)."""
        metrics = {
            "original": {
                "sample_count": 100,
                "lexical": {"distinct_1": 0.85, "distinct_2": 0.42, "distinct_3": 0.28},
            },
            "augmented": {
                "sample_count": 300,
                "lexical": {"distinct_1": 0.91, "distinct_2": 0.58, "distinct_3": 0.39},
            },
            "delta": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_report(metrics, output_path)

            content = output_path.read_text()

            # Check no local file references (only CDN is allowed)
            assert 'src="./' not in content
            assert 'href="./' not in content
            # CSS should be inline
            assert 'rel="stylesheet"' not in content

    def test_generate_report_custom_title(self):
        """Report can have a custom title."""
        metrics = {"dataset": "test", "sample_count": 10, "lexical": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_report(metrics, output_path, title="My Custom Report")

            content = output_path.read_text()

            assert "My Custom Report" in content

    def test_generate_report_with_perplexity(self):
        """Report includes perplexity data when available."""
        metrics = {
            "original": {
                "sample_count": 100,
                "lexical": {"distinct_1": 0.85, "distinct_2": 0.42, "distinct_3": 0.28},
                "perplexity": {
                    "mean_perplexity": 52.3,
                    "std_perplexity": 12.1,
                    "min_perplexity": 8.5,
                    "max_perplexity": 120.4,
                    "median_perplexity": 48.2,
                    "p90_perplexity": 78.5,
                    "p95_perplexity": 95.2,
                    "model": "gpt2",
                },
            },
            "augmented": {
                "sample_count": 300,
                "lexical": {"distinct_1": 0.91, "distinct_2": 0.58, "distinct_3": 0.39},
                "perplexity": {
                    "mean_perplexity": 58.7,
                    "std_perplexity": 15.8,
                    "min_perplexity": 10.2,
                    "max_perplexity": 150.1,
                    "median_perplexity": 54.1,
                    "p90_perplexity": 89.2,
                    "p95_perplexity": 108.4,
                    "model": "gpt2",
                },
            },
            "delta": {
                "mean_perplexity_change": "+12.2%",
                "std_perplexity_change": "+30.6%",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_report(metrics, output_path)

            content = output_path.read_text()

            # Check perplexity data is in the JSON
            assert "mean_perplexity" in content
            assert "52.3" in content or "52,3" in content  # Allow locale variations


class TestPrepareReportData:
    """Tests for report data preparation."""

    def test_prepare_comparison_mode(self):
        """Data preparation detects comparison mode."""
        metrics = {
            "original": {"sample_count": 100},
            "augmented": {"sample_count": 300},
            "delta": {},
        }

        data = _prepare_report_data(metrics)

        assert data["comparison_mode"] is True
        assert "original" in data
        assert "augmented" in data

    def test_prepare_single_mode(self):
        """Data preparation detects single dataset mode."""
        metrics = {
            "dataset": "test-dataset",
            "sample_count": 100,
            "lexical": {"distinct_1": 0.5},
        }

        data = _prepare_report_data(metrics)

        assert data["comparison_mode"] is False
        assert "dataset" in data

    def test_prepare_includes_metadata(self):
        """Data preparation includes metadata."""
        metrics = {
            "dataset": "test",
            "sample_count": 10,
            "metadata": {"git_commit": "abc123"},
        }

        data = _prepare_report_data(metrics)

        assert data["metadata"]["git_commit"] == "abc123"

    def test_prepare_includes_generated_at(self):
        """Data preparation adds generation timestamp."""
        metrics = {"dataset": "test", "sample_count": 10}

        data = _prepare_report_data(metrics)

        assert "generated_at" in data
        assert len(data["generated_at"]) > 0

    def test_prepare_augmentation_contributions(self):
        """Data preparation includes augmentation contributions."""
        metrics = {
            "original": {"sample_count": 100},
            "augmented": {"sample_count": 300},
            "delta": {},
            "augmentation_contributions": {
                "translate:fr": {"mean_perplexity": 72.3, "sample_count": 100},
                "baseline": {"mean_perplexity": 52.3, "sample_count": 100},
            },
        }

        data = _prepare_report_data(metrics)

        assert "augmentation_contributions" in data
        assert "translate:fr" in data["augmentation_contributions"]


class TestComputeHistogramBins:
    """Tests for histogram bin computation."""

    def test_histogram_basic(self):
        """Histogram bins are computed correctly."""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        bins = _compute_histogram_bins(values, num_bins=5)

        assert len(bins) == 5
        assert all("label" in b for b in bins)
        assert all("count" in b for b in bins)
        assert all("min" in b for b in bins)
        assert all("max" in b for b in bins)

    def test_histogram_empty(self):
        """Histogram handles empty input."""
        bins = _compute_histogram_bins([])
        assert bins == []

    def test_histogram_single_value(self):
        """Histogram handles single value."""
        bins = _compute_histogram_bins([42], num_bins=3)
        # Single value creates degenerate bins
        assert len(bins) == 3

    def test_histogram_filters_inf(self):
        """Histogram filters out infinite values."""
        values = [10, 20, float("inf"), 30, float("inf"), 40]
        bins = _compute_histogram_bins(values, num_bins=3)

        # Should have bins without inf values
        total_count = sum(b["count"] for b in bins)
        assert total_count == 4  # Only finite values counted

    def test_histogram_bin_counts_sum(self):
        """Histogram bin counts sum to total."""
        import random

        random.seed(42)
        values = [random.uniform(0, 100) for _ in range(100)]
        bins = _compute_histogram_bins(values, num_bins=10)

        total_count = sum(b["count"] for b in bins)
        assert total_count == 100


class TestEscapeHtml:
    """Tests for HTML escaping."""

    def test_escape_basic(self):
        """Basic HTML escaping works."""
        assert _escape_html("<script>") == "&lt;script&gt;"
        assert _escape_html("a & b") == "a &amp; b"
        assert _escape_html('"quoted"') == "&quot;quoted&quot;"

    def test_escape_preserves_safe(self):
        """Safe characters are preserved."""
        assert _escape_html("Hello World") == "Hello World"
        assert _escape_html("123") == "123"

    def test_escape_multiple_entities(self):
        """Multiple entities are escaped."""
        text = '<a href="link">Click & Go</a>'
        escaped = _escape_html(text)
        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "&amp;" in escaped
        assert "&quot;" in escaped


class TestReportIntegration:
    """Integration tests for report generation."""

    def test_report_with_full_metrics(self):
        """Report generation with comprehensive metrics."""
        metrics = {
            "original": {
                "sample_count": 1000,
                "lexical": {
                    "distinct_1": 0.85,
                    "distinct_2": 0.42,
                    "distinct_3": 0.28,
                    "type_token_ratio": 0.32,
                    "total_tokens": 50000,
                    "sample_count": 1000,
                },
                "perplexity": {
                    "mean_perplexity": 52.3,
                    "std_perplexity": 12.1,
                    "min_perplexity": 8.5,
                    "max_perplexity": 120.4,
                    "median_perplexity": 48.2,
                    "p90_perplexity": 78.5,
                    "p95_perplexity": 95.2,
                    "model": "gpt2",
                    "device": "cuda",
                    "valid_samples": 998,
                },
                "vendi_score": 847.2,
            },
            "augmented": {
                "sample_count": 3000,
                "lexical": {
                    "distinct_1": 0.91,
                    "distinct_2": 0.58,
                    "distinct_3": 0.39,
                    "type_token_ratio": 0.28,
                    "total_tokens": 150000,
                    "sample_count": 3000,
                },
                "perplexity": {
                    "mean_perplexity": 58.7,
                    "std_perplexity": 15.8,
                    "min_perplexity": 10.2,
                    "max_perplexity": 150.1,
                    "median_perplexity": 54.1,
                    "p90_perplexity": 89.2,
                    "p95_perplexity": 108.4,
                    "model": "gpt2",
                    "device": "cuda",
                    "valid_samples": 2995,
                },
                "vendi_score": 1234.5,
            },
            "delta": {
                "sample_count_change": "+200.0%",
                "distinct_1_change": "+7.1%",
                "distinct_2_change": "+38.1%",
                "distinct_3_change": "+39.3%",
                "mean_perplexity_change": "+12.2%",
                "std_perplexity_change": "+30.6%",
                "vendi_score_change": "+45.7%",
            },
            "summary": "Dataset size: 1000 → 3000 (+200.0%)\nLexical diversity (Distinct-2): +38.1%\nPerplexity change: +12.2%",
            "augmentation_contributions": {
                "translate:fr": {"mean_perplexity": 72.3, "sample_count": 450},
                "translate:es": {"mean_perplexity": 68.1, "sample_count": 420},
                "shuffle": {"mean_perplexity": 54.2, "sample_count": 1200},
                "baseline": {"mean_perplexity": 52.3, "sample_count": 1000},
            },
            "metadata": {
                "original_path": "datasets/baseline",
                "augmented_path": "datasets/augmented",
                "model": "gpt2",
                "git_commit": "abc1234",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "full_report.html"
            result = generate_report(metrics, output_path, title="Full Test Report")

            assert result.exists()
            content = result.read_text()

            # Check key data is present
            assert "1000" in content or "1,000" in content  # Original sample count
            assert "3000" in content or "3,000" in content  # Augmented sample count
            assert "gpt2" in content  # Model name
            assert "translate:fr" in content  # Augmentation type

            # Check report is reasonably sized (should include data + JS)
            size_kb = result.stat().st_size / 1024
            assert size_kb > 10  # At least 10KB (has content)
            assert size_kb < 500  # Less than 500KB (not bloated)

    def test_report_valid_json_embedded(self):
        """Report contains valid embedded JSON data."""
        metrics = {
            "dataset": "test",
            "sample_count": 100,
            "lexical": {"distinct_1": 0.5, "distinct_2": 0.3},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_report(metrics, output_path)

            content = output_path.read_text()

            # Extract JSON from reportData = {...}
            import re

            match = re.search(r"const reportData = ({[\s\S]*?});", content)
            assert match is not None

            # Parse the JSON
            json_str = match.group(1)
            data = json.loads(json_str)

            assert "generated_at" in data
            assert "comparison_mode" in data
