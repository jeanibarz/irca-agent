# ADR-004: HTML Diversity Report Generation

* Status: proposed
* Deciders: Jean
* Date: 2026-01-16

Technical Story: Enable browser-based visualization of diversity evaluation results for human-friendly analysis

## Context and Problem Statement

The diversity evaluation system (`src/diversity/`) computes rich metrics across three tiers:

1. **Lexical** (instant): distinct-n, TTR, vocabulary coverage, hapax ratio
2. **Semantic** (minutes): Vendi Score, embedding coverage, pairwise distances
3. **Perplexity** (hours): mean, std, percentiles, model-dependent metrics

Currently, these metrics are returned as Python dictionaries and displayed as plain text in the CLI output. This creates several problems:

### Current Limitations

1. **Poor readability**: Nested dictionaries with many numeric values are hard to parse visually
2. **No visualizations**: Distribution data (perplexity values, n-gram frequencies) cannot be charted
3. **No comparison view**: Side-by-side original vs augmented metrics require manual inspection
4. **No persistence**: Results are lost when the terminal session ends
5. **No sharing**: Cannot easily share results with teammates for review

### Example of Current CLI Output

```
Dataset size: 1000 → 3000 (+200.0%)
Lexical diversity (Distinct-2): +25.4%
Perplexity change: +12.3%
Interpretation: Augmented dataset is significantly more challenging for the model.
```

This summary hides the rich underlying data:
- Distribution of perplexity values
- Per-augmentation-type contributions
- Vocabulary statistics and top n-grams
- Embedding space coverage metrics

## Decision Drivers

* **Human readability**: Metrics should be immediately understandable
* **Visual clarity**: Charts and graphs for distribution data
* **Self-contained**: Single HTML file, no external dependencies
* **Portable**: Works offline, shareable via email/Slack
* **Interactive**: Tooltips, collapsible sections, data exploration
* **Extensible**: Easy to add new metrics/visualizations
* **Integration**: Works with existing CLI workflow

## Considered Options

### Option 1: Plain HTML with Tables (Minimal)

**Description**: Generate simple HTML with styled tables showing all metrics.

```html
<table class="metrics">
  <tr><th>Metric</th><th>Original</th><th>Augmented</th><th>Delta</th></tr>
  <tr><td>Distinct-2</td><td>0.42</td><td>0.58</td><td class="positive">+38.1%</td></tr>
</table>
```

| Pros | Cons |
|------|------|
| Simple to implement | No visualizations for distributions |
| Works everywhere | Static, no interactivity |
| Tiny file size | Limited data exploration |
| | Still hard to interpret percentiles |

### Option 2: Interactive HTML with Chart.js (Recommended)

**Description**: Self-contained HTML with embedded Chart.js for interactive visualizations.

```html
<!DOCTYPE html>
<html>
<head>
  <script>/* Chart.js minified inline */</script>
  <style>/* CSS inline */</style>
</head>
<body>
  <div class="metric-card">
    <h3>Perplexity Distribution</h3>
    <canvas id="ppl-histogram"></canvas>
  </div>
  <script>
    const data = /* JSON metrics embedded */;
    new Chart('ppl-histogram', {...});
  </script>
</body>
</html>
```

| Pros | Cons |
|------|------|
| Rich visualizations | Larger file size (~200KB with Chart.js) |
| Interactive (hover, zoom) | More complex implementation |
| Self-contained single file | Requires embedding JS library |
| Professional appearance | |
| Responsive design | |

### Option 3: Jupyter Notebook Export

**Description**: Generate a Jupyter notebook with matplotlib/plotly visualizations.

| Pros | Cons |
|------|------|
| Full Python ecosystem | Requires Jupyter to view |
| Highly customizable | Not self-contained |
| Can re-run analysis | Larger audience can't open |
| | Heavyweight dependency |

### Option 4: JSON Export + Separate Viewer

**Description**: Export metrics as JSON, provide a separate HTML viewer tool.

| Pros | Cons |
|------|------|
| Clean data/presentation separation | Two files to manage |
| JSON reusable for other tools | Requires serving viewer |
| | Not self-contained |

## Decision Outcome

**Chosen option: Option 2 - Interactive HTML with Chart.js**

This provides the best balance of visual richness, portability, and user experience. A single self-contained HTML file can be opened in any browser, shared via any medium, and provides interactive exploration of the metrics.

## Detailed Design

### CLI Integration

Add `--report` flag to the `irca dataset diversity` command:

```bash
# Generate HTML report alongside CLI output
irca dataset diversity \
  -o datasets/baseline \
  -a datasets/augmented \
  --model gpt2 \
  --report diversity-report.html

# Quick mode (no perplexity)
irca dataset diversity \
  -o datasets/baseline \
  -a datasets/augmented \
  --quick \
  --report quick-report.html

# Single dataset analysis
irca dataset diversity \
  -d datasets/formatted \
  --report single-dataset.html
```

### Report Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│  IRCA Diversity Report                            Generated: ...    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────── SUMMARY CARD ───────────────┐                     │
│  │  📊 Dataset Size    │  1000 → 3000 (+200%) │                     │
│  │  📈 Diversity Gain  │  +38.1% (Distinct-2) │                     │
│  │  🎯 Perplexity Δ    │  +12.3% (harder)     │                     │
│  │  ✅ Verdict         │  Effective augment.  │                     │
│  └────────────────────────────────────────────┘                     │
│                                                                     │
│  ┌─────────────── LEXICAL METRICS ─────────────┐                    │
│  │                                              │                    │
│  │  [Bar Chart: Distinct-n Comparison]          │                    │
│  │  ┌──────┬──────┬──────┐                      │                    │
│  │  │ D-1  │ D-2  │ D-3  │  ← Original (blue)   │                    │
│  │  │ ████ │ ███  │ ██   │  ← Augmented (green) │                    │
│  │  └──────┴──────┴──────┘                      │                    │
│  │                                              │                    │
│  │  Vocabulary: 12,450 unique tokens            │                    │
│  │  Hapax Ratio: 45.2% (single-occurrence)      │                    │
│  │  Type-Token Ratio: 0.32                      │                    │
│  │                                              │                    │
│  │  [Collapsible: Top 20 Bigrams]               │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                     │
│  ┌─────────────── SEMANTIC METRICS ────────────┐                    │
│  │                                              │                    │
│  │  Vendi Score: 847.2 → 1,234.5 (+45.7%)       │                    │
│  │  (Effective unique samples)                  │                    │
│  │                                              │                    │
│  │  [Gauge Chart: Diversity Ratio]              │                    │
│  │  Original: 0.85 ─────●───── Augmented: 0.91  │                    │
│  │                                              │                    │
│  │  Embedding Coverage:                         │                    │
│  │  • Avg Pairwise Distance: 0.42 → 0.48        │                    │
│  │  • Spread: 0.18 → 0.21                       │                    │
│  │  • Centroid Distance: 0.35 → 0.38            │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                     │
│  ┌─────────────── PERPLEXITY ANALYSIS ─────────┐                    │
│  │                                              │                    │
│  │  [Histogram: Perplexity Distribution]        │                    │
│  │       ▄                                      │                    │
│  │      ▄█▄                                     │                    │
│  │     ▄███▄    ← Original (blue outline)       │                    │
│  │    ▄█████▄   ← Augmented (green fill)        │                    │
│  │  ──────────────────────────                  │                    │
│  │   10   50   100   150   200  (perplexity)    │                    │
│  │                                              │                    │
│  │  Statistics:                                 │                    │
│  │  ┌────────────┬──────────┬──────────┬───────┐│                    │
│  │  │ Metric     │ Original │ Augmented│ Delta ││                    │
│  │  ├────────────┼──────────┼──────────┼───────┤│                    │
│  │  │ Mean       │ 52.3     │ 58.7     │ +12.2%││                    │
│  │  │ Median     │ 48.2     │ 54.1     │ +12.2%││                    │
│  │  │ Std Dev    │ 12.1     │ 15.8     │ +30.6%││                    │
│  │  │ P90        │ 78.5     │ 89.2     │ +13.6%││                    │
│  │  │ P95        │ 95.2     │ 108.4    │ +13.9%││                    │
│  │  └────────────┴──────────┴──────────┴───────┘│                    │
│  │                                              │                    │
│  │  Model: gpt2 | Device: cuda                  │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                     │
│  ┌─────────────── AUGMENTATION BREAKDOWN ──────┐                    │
│  │                                              │                    │
│  │  [Horizontal Bar: Per-Augmentation PPL]      │                    │
│  │                                              │                    │
│  │  translate:fr   ████████████████  72.3       │                    │
│  │  translate:es   ███████████████   68.1       │                    │
│  │  shuffle        ████████          54.2       │                    │
│  │  baseline       ███████           52.3       │                    │
│  │                                              │                    │
│  │  Interpretation: Translation adds the most   │                    │
│  │  diversity, especially French (+38.2%)       │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                     │
│  ┌─────────────── METADATA ────────────────────┐                    │
│  │  Generated: 2026-01-16 14:30:00 UTC          │                    │
│  │  Original: datasets/baseline (1000 samples)  │                    │
│  │  Augmented: datasets/augmented (3000 samples)│                    │
│  │  Perplexity Model: gpt2                      │                    │
│  │  Embedding Model: all-MiniLM-L6-v2           │                    │
│  │  Git Commit: abc1234                         │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Architecture

```
src/diversity/
├── __init__.py
├── lexical.py           # Existing
├── semantic.py          # Existing
├── perplexity.py        # Existing
├── effectiveness.py     # Existing
├── utils.py             # Existing
└── report.py            # NEW: HTML report generator
```

#### New Module: `src/diversity/report.py`

```python
"""HTML report generator for diversity metrics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Jinja2 for templating (already in deps for Guidance)
from jinja2 import Environment, PackageLoader

class DiversityReportGenerator:
    """Generate interactive HTML reports for diversity metrics."""

    def __init__(self):
        self.env = Environment(
            loader=PackageLoader('src.diversity', 'templates'),
            autoescape=True,
        )

    def generate(
        self,
        metrics: dict[str, Any],
        output_path: Path,
        title: str = "Diversity Report",
    ) -> Path:
        """Generate HTML report from diversity metrics."""
        template = self.env.get_template('report.html.j2')

        html = template.render(
            title=title,
            metrics=metrics,
            generated_at=datetime.utcnow().isoformat(),
            chartjs=self._get_chartjs_inline(),
        )

        output_path.write_text(html)
        return output_path

    def _get_chartjs_inline(self) -> str:
        """Return minified Chart.js for inline embedding."""
        # Bundled Chart.js (~80KB minified)
        ...
```

#### Template Structure

```
src/diversity/templates/
├── report.html.j2           # Main report template
├── components/
│   ├── summary_card.html.j2
│   ├── lexical_section.html.j2
│   ├── semantic_section.html.j2
│   ├── perplexity_section.html.j2
│   └── augmentation_breakdown.html.j2
└── static/
    ├── chart.min.js         # Chart.js (embedded inline)
    └── styles.css           # Report styles (embedded inline)
```

### Data Requirements

The report generator expects metrics in this structure:

```python
{
    "comparison_mode": True,  # or False for single dataset

    # Dataset metadata
    "original": {
        "path": "datasets/baseline",
        "sample_count": 1000,
    },
    "augmented": {  # Optional if comparison_mode=False
        "path": "datasets/augmented",
        "sample_count": 3000,
    },

    # Lexical metrics
    "lexical": {
        "original": {
            "distinct_1": 0.85,
            "distinct_2": 0.42,
            "distinct_3": 0.28,
            "type_token_ratio": 0.32,
            "vocabulary_size": 8450,
            "hapax_legomena": 3820,
            "hapax_ratio": 0.452,
            "top_bigrams": [  # Optional
                [["the", "user"], 1234],
                [["function", "call"], 892],
                ...
            ],
        },
        "augmented": {...},
        "delta": {
            "distinct_2_change": "+38.1%",
            ...
        },
    },

    # Semantic metrics (optional)
    "semantic": {
        "original": {
            "vendi_score": 847.2,
            "effective_diversity_ratio": 0.847,
            "avg_pairwise_distance": 0.42,
            "std_pairwise_distance": 0.12,
            "embedding_spread": 0.18,
            "centroid_distance": 0.35,
        },
        "augmented": {...},
        "delta": {...},
    },

    # Perplexity metrics (optional)
    "perplexity": {
        "model": "gpt2",
        "device": "cuda",
        "original": {
            "mean_perplexity": 52.3,
            "std_perplexity": 12.1,
            "min_perplexity": 8.5,
            "max_perplexity": 120.4,
            "median_perplexity": 48.2,
            "p90_perplexity": 78.5,
            "p95_perplexity": 95.2,
            "histogram": [  # For distribution chart
                {"bin": "0-20", "count": 45},
                {"bin": "20-40", "count": 234},
                {"bin": "40-60", "count": 456},
                ...
            ],
        },
        "augmented": {...},
        "delta": {...},
    },

    # Per-augmentation breakdown (optional)
    "augmentation_contributions": {
        "translate:fr": {"mean_perplexity": 72.3, "sample_count": 450},
        "translate:es": {"mean_perplexity": 68.1, "sample_count": 420},
        "shuffle": {"mean_perplexity": 54.2, "sample_count": 1200},
        "baseline": {"mean_perplexity": 52.3, "sample_count": 1000},
    },

    # Report metadata
    "metadata": {
        "generated_at": "2026-01-16T14:30:00Z",
        "git_commit": "abc1234",
        "git_branch": "main",
        "irca_version": "0.5.0",
    },
}
```

### Chart Specifications

#### 1. Distinct-n Comparison (Grouped Bar Chart)

```javascript
{
  type: 'bar',
  data: {
    labels: ['Distinct-1', 'Distinct-2', 'Distinct-3'],
    datasets: [
      { label: 'Original', data: [0.85, 0.42, 0.28], backgroundColor: '#3b82f6' },
      { label: 'Augmented', data: [0.91, 0.58, 0.39], backgroundColor: '#22c55e' },
    ]
  },
  options: {
    scales: { y: { beginAtZero: true, max: 1 } },
    plugins: { tooltip: { callbacks: { label: (ctx) => `${ctx.parsed.y.toFixed(3)}` } } }
  }
}
```

#### 2. Perplexity Distribution (Overlapping Histogram)

```javascript
{
  type: 'bar',
  data: {
    labels: ['0-20', '20-40', '40-60', '60-80', '80-100', '100+'],
    datasets: [
      { label: 'Original', data: [...], backgroundColor: 'rgba(59, 130, 246, 0.5)', borderColor: '#3b82f6' },
      { label: 'Augmented', data: [...], backgroundColor: 'rgba(34, 197, 94, 0.5)', borderColor: '#22c55e' },
    ]
  },
  options: {
    barPercentage: 0.9,
    categoryPercentage: 0.9,
  }
}
```

#### 3. Augmentation Contribution (Horizontal Bar)

```javascript
{
  type: 'bar',
  data: {
    labels: ['translate:fr', 'translate:es', 'shuffle', 'baseline'],
    datasets: [{
      label: 'Mean Perplexity',
      data: [72.3, 68.1, 54.2, 52.3],
      backgroundColor: ['#f59e0b', '#f59e0b', '#6366f1', '#94a3b8'],
    }]
  },
  options: { indexAxis: 'y' }
}
```

### Styling Guidelines

- **Color palette**:
  - Original: Blue (`#3b82f6`)
  - Augmented: Green (`#22c55e`)
  - Positive delta: Green
  - Negative delta: Red
  - Neutral: Gray (`#94a3b8`)
- **Typography**: System font stack, monospace for numbers
- **Layout**: Responsive grid, cards with subtle shadows
- **Dark mode**: Respect `prefers-color-scheme`

### Error Handling

The report should gracefully handle missing data:

```html
{% if perplexity %}
  <section class="perplexity">...</section>
{% else %}
  <section class="perplexity unavailable">
    <p>Perplexity metrics not computed. Run without --quick to include.</p>
  </section>
{% endif %}
```

## Consequences

### Positive

* **Human-friendly output**: Complex metrics become visually intuitive
* **Shareable**: Single HTML file works anywhere
* **Persistent**: Results saved for later reference
* **Interactive**: Hover tooltips, collapsible sections
* **Professional**: Publication-ready visualizations
* **Offline**: No internet required to view

### Negative

* **File size**: ~250KB per report (Chart.js + data)
* **Implementation effort**: New module, templates, bundling
* **Maintenance**: Chart.js updates need manual bundling
* **No live updates**: Static snapshot at generation time

### Neutral

* **CLI output unchanged**: Report is optional addition
* **Existing metrics untouched**: Report consumes existing data structures

## Implementation Plan

### Phase 1: Core Report Infrastructure
1. Create `src/diversity/report.py` with `DiversityReportGenerator`
2. Create Jinja2 template structure
3. Bundle Chart.js (minified, ~80KB)
4. Implement summary card and metric tables

### Phase 2: Visualizations
1. Add distinct-n grouped bar chart
2. Add perplexity histogram
3. Add augmentation contribution chart
4. Add Vendi score gauge

### Phase 3: CLI Integration
1. Add `--report` flag to diversity command
2. Compute histogram data during perplexity analysis
3. Pass augmentation labels through pipeline
4. Add metadata collection (git, version)

### Phase 4: Polish
1. Responsive design testing
2. Dark mode support
3. Accessibility improvements (ARIA labels)
4. Print stylesheet

## Requirements

This ADR introduces the following requirements:

| Req ID | Name | Summary |
|--------|------|---------|
| FR-DIV-10 | HTML Report Generation | Generate self-contained HTML report from diversity metrics |
| FR-DIV-11 | Report Visualizations | Include Chart.js charts for distributions and comparisons |
| FR-DIV-12 | Report CLI Flag | Add `--report` flag to diversity command |
| FR-DIV-13 | Report Metadata | Include generation timestamp, git info, model info |
| NFR-DIV-01 | Report Portability | Single file, works offline, no external dependencies |
| NFR-DIV-02 | Report File Size | Maximum 500KB including all data and libraries |

## Acceptance Tests

### FR-DIV-10: HTML Report Generation
```gherkin
Scenario: Generate report from comparison metrics
  Given diversity metrics comparing two datasets
  When I generate an HTML report
  Then the report file is created
  And it opens in a web browser
  And it displays all computed metrics
```

### FR-DIV-11: Report Visualizations
```gherkin
Scenario: Charts render correctly
  Given a diversity report with perplexity data
  When I open the report in Chrome/Firefox/Safari
  Then the perplexity histogram displays
  And the distinct-n bar chart displays
  And charts are interactive (hover shows values)
```

### FR-DIV-12: Report CLI Flag
```gherkin
Scenario: Generate report via CLI
  Given two formatted datasets
  When I run "irca dataset diversity -o base -a aug --report out.html"
  Then out.html is created
  And CLI output still displays summary
```

### NFR-DIV-01: Report Portability
```gherkin
Scenario: Report works offline
  Given a generated report file
  When I disconnect from the internet
  And I open the report in a browser
  Then all content displays correctly
  And charts render properly
```

## Links

* Depends on: [ADR-003 Dataset Format Pipeline](./ADR-003-dataset-format-pipeline.md)
* Related: [RFC-001 Dataset Diversity Metrics](../rfcs/RFC-001-dataset-diversity-metrics.md)
* Traceability: [TRACEABILITY_MATRIX.md](../TRACEABILITY_MATRIX.md)
