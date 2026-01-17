"""
Experiment Command

Commands for running and evaluating experiments per ADR-008 methodology.
"""

import json
import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group()
def experiment() -> None:
    """Experiment evaluation and reporting commands."""
    pass


@experiment.command("eval-checkpoints")
@click.option(
    "--checkpoints-dir",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing model checkpoints",
)
@click.option(
    "--train-set",
    type=click.Path(exists=True),
    required=True,
    help="Training set for train perplexity evaluation",
)
@click.option(
    "--test-set",
    type=click.Path(exists=True),
    required=True,
    help="Test set for test perplexity evaluation",
)
@click.option(
    "--baseline-model",
    "-b",
    type=str,
    default=None,
    help="Baseline model for comparison (e.g., mistralai/Mistral-7B-Instruct-v0.3)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output JSON file for results",
)
@click.option(
    "--eval-mode",
    type=click.Choice(["completion", "full"]),
    default="completion",
    help="Evaluation mode: completion (recommended) or full",
)
@click.option(
    "--no-chat-template",
    is_flag=True,
    help="Skip chat template conversion (use raw IRCA format)",
)
@click.pass_context
def eval_checkpoints(
    ctx: click.Context,
    checkpoints_dir: str,
    train_set: str,
    test_set: str,
    baseline_model: str | None,
    output: str | None,
    eval_mode: str,
    no_chat_template: bool,
) -> None:
    """
    Evaluate all checkpoints in a directory on train and test sets.

    This command implements the ADR-008 methodology for fair augmentation
    comparison by evaluating each checkpoint's perplexity on both train
    and test sets, computing generalization gaps.

    Examples:

        # Evaluate all checkpoints in a model directory
        irca experiment eval-checkpoints \\
          -c models/baseline \\
          --train-set datasets/train_formatted \\
          --test-set datasets/test_formatted \\
          -o results/baseline.json

        # Compare against baseline model
        irca experiment eval-checkpoints \\
          -c models/augmented \\
          --train-set datasets/train_augmented \\
          --test-set datasets/test_formatted \\
          -b mistralai/Mistral-7B-Instruct-v0.3 \\
          -o results/augmented.json
    """
    import torch

    import datasets
    from src.diversity.perplexity import compute_perplexity_profile
    from src.diversity.statistics import bootstrap_confidence_interval

    verbose = ctx.obj.get("verbose", False)
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    click.echo("🔬 Checkpoint Evaluation (ADR-008 Methodology)")
    click.echo(f"   Checkpoints: {checkpoints_dir}")
    click.echo(f"   Train set: {train_set}")
    click.echo(f"   Test set: {test_set}")
    click.echo(f"   Eval mode: {eval_mode}")
    if baseline_model:
        click.echo(f"   Baseline: {baseline_model}")

    # Load datasets
    click.echo("\n📊 Loading datasets...")
    try:
        train_ds = datasets.load_from_disk(train_set)
        test_ds = datasets.load_from_disk(test_set)

        if isinstance(train_ds, datasets.DatasetDict):
            train_ds = train_ds["train"]
        if isinstance(test_ds, datasets.DatasetDict):
            test_ds = test_ds["train"]

        train_texts = train_ds["text"]
        test_texts = test_ds["text"]
        click.echo(f"   Train samples: {len(train_texts)}")
        click.echo(f"   Test samples: {len(test_texts)}")
    except Exception as e:
        click.secho(f"Error loading datasets: {e}", fg="red", err=True)
        sys.exit(1)

    # Discover checkpoints
    checkpoints_path = Path(checkpoints_dir)
    checkpoint_dirs = sorted(
        [d for d in checkpoints_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1]),
    )

    # Also include the final model (parent directory) if adapter_config.json exists
    if (checkpoints_path / "adapter_config.json").exists():
        checkpoint_dirs.append(checkpoints_path)

    if not checkpoint_dirs:
        click.secho(f"No checkpoints found in {checkpoints_dir}", fg="red", err=True)
        sys.exit(1)

    click.echo(f"\n📁 Found {len(checkpoint_dirs)} checkpoint(s):")
    for cp in checkpoint_dirs:
        click.echo(f"   - {cp.name}")

    # Evaluate each checkpoint
    results = {
        "checkpoints_dir": str(checkpoints_dir),
        "train_set": str(train_set),
        "test_set": str(test_set),
        "eval_mode": eval_mode,
        "apply_chat_template": not no_chat_template,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "checkpoints": [],
    }

    apply_chat_template = not no_chat_template

    for i, checkpoint_path in enumerate(checkpoint_dirs):
        checkpoint_name = checkpoint_path.name if checkpoint_path != checkpoints_path else "final"
        click.echo(f"\n🔄 [{i + 1}/{len(checkpoint_dirs)}] Evaluating {checkpoint_name}...")

        try:
            # Evaluate on train set
            click.echo("   Computing train perplexity...")
            train_profile = compute_perplexity_profile(
                train_texts,
                model_name=str(checkpoint_path),
                eval_mode=eval_mode,
                show_progress=True,
                use_cache=False,
                apply_chat_template=apply_chat_template,
                return_perplexities=True,
            )

            # Compute bootstrap CI for train
            train_mean, train_ci_lower, train_ci_upper = bootstrap_confidence_interval(train_profile["perplexities"])

            # Evaluate on test set
            click.echo("   Computing test perplexity...")
            test_profile = compute_perplexity_profile(
                test_texts,
                model_name=str(checkpoint_path),
                eval_mode=eval_mode,
                show_progress=True,
                use_cache=False,
                apply_chat_template=apply_chat_template,
                return_perplexities=True,
            )

            # Compute bootstrap CI for test
            test_mean, test_ci_lower, test_ci_upper = bootstrap_confidence_interval(test_profile["perplexities"])

            # Compute generalization gap
            gap_absolute = test_mean - train_mean
            gap_relative = gap_absolute / train_mean if train_mean > 0 else 0

            checkpoint_result = {
                "name": checkpoint_name,
                "path": str(checkpoint_path),
                "train": {
                    "mean_perplexity": train_mean,
                    "ci_95_lower": train_ci_lower,
                    "ci_95_upper": train_ci_upper,
                    "std_perplexity": train_profile["std_perplexity"],
                    "median_perplexity": train_profile["median_perplexity"],
                },
                "test": {
                    "mean_perplexity": test_mean,
                    "ci_95_lower": test_ci_lower,
                    "ci_95_upper": test_ci_upper,
                    "std_perplexity": test_profile["std_perplexity"],
                    "median_perplexity": test_profile["median_perplexity"],
                },
                "gap": {
                    "absolute": gap_absolute,
                    "relative": gap_relative,
                    "relative_pct": f"{gap_relative * 100:+.1f}%",
                },
            }

            results["checkpoints"].append(checkpoint_result)

            # Print summary for this checkpoint
            click.echo(f"   Train PPL: {train_mean:.2f} [{train_ci_lower:.2f}, {train_ci_upper:.2f}]")
            click.echo(f"   Test PPL:  {test_mean:.2f} [{test_ci_lower:.2f}, {test_ci_upper:.2f}]")
            click.echo(f"   Gap: {gap_relative * 100:+.1f}%")

            # Clear GPU memory
            torch.cuda.empty_cache()

        except Exception as e:
            click.secho(f"   Error evaluating {checkpoint_name}: {e}", fg="yellow")
            logger.exception(f"Error evaluating checkpoint {checkpoint_path}")
            continue

    # Evaluate baseline model if provided
    if baseline_model:
        click.echo(f"\n🔄 Evaluating baseline model: {baseline_model}...")
        try:
            # Test set only for baseline (most important comparison)
            baseline_profile = compute_perplexity_profile(
                test_texts,
                model_name=baseline_model,
                eval_mode=eval_mode,
                show_progress=True,
                use_cache=False,
                apply_chat_template=apply_chat_template,
                return_perplexities=True,
            )

            baseline_mean, baseline_ci_lower, baseline_ci_upper = bootstrap_confidence_interval(
                baseline_profile["perplexities"]
            )

            results["baseline"] = {
                "model": baseline_model,
                "test": {
                    "mean_perplexity": baseline_mean,
                    "ci_95_lower": baseline_ci_lower,
                    "ci_95_upper": baseline_ci_upper,
                    "std_perplexity": baseline_profile["std_perplexity"],
                },
            }

            click.echo(f"   Test PPL: {baseline_mean:.2f} [{baseline_ci_lower:.2f}, {baseline_ci_upper:.2f}]")

            torch.cuda.empty_cache()

        except Exception as e:
            click.secho(f"   Error evaluating baseline: {e}", fg="yellow")
            logger.exception(f"Error evaluating baseline model {baseline_model}")

    # Find best checkpoint (lowest test perplexity)
    if results["checkpoints"]:
        best = min(results["checkpoints"], key=lambda x: x["test"]["mean_perplexity"])
        results["best_checkpoint"] = {
            "name": best["name"],
            "test_perplexity": best["test"]["mean_perplexity"],
        }

    # Print summary table
    click.echo("\n" + "=" * 70)
    click.echo("CHECKPOINT EVALUATION SUMMARY")
    click.echo("=" * 70)
    click.echo(f"{'Checkpoint':<20} {'Train PPL':>12} {'Test PPL':>12} {'Gap':>10}")
    click.echo("-" * 70)

    for cp in results["checkpoints"]:
        name = cp["name"][:20]
        train_ppl = cp["train"]["mean_perplexity"]
        test_ppl = cp["test"]["mean_perplexity"]
        gap = cp["gap"]["relative_pct"]
        marker = " *" if cp["name"] == results.get("best_checkpoint", {}).get("name") else ""
        click.echo(f"{name:<20} {train_ppl:>12.2f} {test_ppl:>12.2f} {gap:>10}{marker}")

    if "baseline" in results:
        click.echo("-" * 70)
        click.echo(f"{'Baseline':<20} {'N/A':>12} {results['baseline']['test']['mean_perplexity']:>12.2f} {'N/A':>10}")

    click.echo("-" * 70)
    if "best_checkpoint" in results:
        click.echo(f"* Best checkpoint: {results['best_checkpoint']['name']}")

    # ADR-008 interpretation
    if results["checkpoints"] and "baseline" in results:
        best_test = results["best_checkpoint"]["test_perplexity"]
        baseline_test = results["baseline"]["test"]["mean_perplexity"]
        improvement = (baseline_test - best_test) / baseline_test

        click.echo("\n📊 ADR-008 Interpretation:")
        if improvement > 0.1:
            click.echo(f"   ✅ Significant improvement: {improvement * 100:.1f}% reduction in test perplexity")
        elif improvement > 0.05:
            click.echo(f"   📈 Modest improvement: {improvement * 100:.1f}% reduction in test perplexity")
        elif improvement > 0:
            click.echo(f"   ➡️ Marginal improvement: {improvement * 100:.1f}% reduction in test perplexity")
        else:
            click.echo(f"   ⚠️ No improvement: {abs(improvement) * 100:.1f}% increase in test perplexity")

    # Save results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\n💾 Results saved to: {output}")

    click.secho("\n✅ Checkpoint evaluation complete!", fg="green")


@experiment.command("compare")
@click.option(
    "--baseline-results",
    "-b",
    type=click.Path(exists=True),
    required=True,
    help="JSON results file from baseline evaluation",
)
@click.option(
    "--augmented-results",
    "-a",
    type=click.Path(exists=True),
    required=True,
    help="JSON results file from augmented evaluation",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file for comparison report (HTML or JSON)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "json", "text"]),
    default="text",
    help="Output format for comparison report",
)
def compare(
    baseline_results: str,
    augmented_results: str,
    output: str | None,
    output_format: str,
) -> None:
    """
    Compare baseline vs augmented experiment results.

    Generates a comparison report following ADR-008 methodology,
    including statistical significance and interpretation guidance.

    Examples:

        # Text comparison
        irca experiment compare \\
          -b results/baseline.json \\
          -a results/augmented.json

        # HTML report
        irca experiment compare \\
          -b results/baseline.json \\
          -a results/augmented.json \\
          -o results/comparison.html \\
          --format html
    """
    from src.diversity.report import generate_experiment_comparison_report

    click.echo("📊 Experiment Comparison (ADR-008)")
    click.echo(f"   Baseline: {baseline_results}")
    click.echo(f"   Augmented: {augmented_results}")

    # Load results
    with open(baseline_results) as f:
        baseline = json.load(f)
    with open(augmented_results) as f:
        augmented = json.load(f)

    # Generate comparison
    report = generate_experiment_comparison_report(baseline, augmented)

    if output_format == "text" or output is None:
        # Print text summary
        click.echo("\n" + "=" * 70)
        click.echo("EXPERIMENT COMPARISON REPORT")
        click.echo("=" * 70)

        click.echo("\n📈 Best Checkpoint Comparison:")
        click.echo(f"{'Metric':<25} {'Baseline':>15} {'Augmented':>15} {'Delta':>12}")
        click.echo("-" * 70)

        baseline_best = report["baseline"]["best_test_ppl"]
        augmented_best = report["augmented"]["best_test_ppl"]
        delta = augmented_best - baseline_best
        delta_pct = delta / baseline_best * 100 if baseline_best > 0 else 0

        click.echo(f"{'Test Perplexity':<25} {baseline_best:>15.2f} {augmented_best:>15.2f} {delta_pct:>+11.1f}%")

        baseline_gap = report["baseline"]["best_gap"]
        augmented_gap = report["augmented"]["best_gap"]
        click.echo(f"{'Generalization Gap':<25} {baseline_gap * 100:>14.1f}% {augmented_gap * 100:>14.1f}%")

        click.echo("\n" + "=" * 70)
        click.echo("INTERPRETATION (ADR-008)")
        click.echo("=" * 70)
        click.echo(report["interpretation"])

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
        elif output_format == "html":
            html_content = _generate_html_report(report, baseline, augmented)
            with open(output_path, "w") as f:
                f.write(html_content)
        else:
            with open(output_path, "w") as f:
                f.write(report["text_summary"])

        click.echo(f"\n💾 Report saved to: {output}")

    click.secho("\n✅ Comparison complete!", fg="green")


def _generate_html_report(report: dict, baseline: dict, augmented: dict) -> str:
    """Generate an HTML comparison report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>ADR-008 Experiment Comparison Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        tr:nth-child(even) {{ background-color: #fafafa; }}
        .metric-good {{ color: #28a745; font-weight: bold; }}
        .metric-bad {{ color: #dc3545; font-weight: bold; }}
        .metric-neutral {{ color: #6c757d; }}
        .interpretation {{ background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .summary-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>ADR-008 Experiment Comparison Report</h1>
    <p><strong>Generated:</strong> {timestamp}</p>

    <div class="summary-box">
        <h3>Experiment Configuration</h3>
        <ul>
            <li><strong>Baseline:</strong> {baseline_train_samples} train samples, {baseline_checkpoints} checkpoints</li>
            <li><strong>Augmented:</strong> {augmented_train_samples} train samples, {augmented_checkpoints} checkpoints</li>
            <li><strong>Test samples:</strong> {test_samples}</li>
            <li><strong>Eval mode:</strong> {eval_mode}</li>
        </ul>
    </div>

    <h2>Best Checkpoint Comparison</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Baseline</th>
            <th>Augmented</th>
            <th>Delta</th>
        </tr>
        <tr>
            <td>Best Checkpoint</td>
            <td>{baseline_best_name}</td>
            <td>{augmented_best_name}</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Test Perplexity</td>
            <td>{baseline_test_ppl:.2f}</td>
            <td>{augmented_test_ppl:.2f}</td>
            <td class="{delta_class}">{delta_pct:+.1f}%</td>
        </tr>
        <tr>
            <td>Train Perplexity</td>
            <td>{baseline_train_ppl:.2f}</td>
            <td>{augmented_train_ppl:.2f}</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Generalization Gap</td>
            <td>{baseline_gap:.1f}%</td>
            <td>{augmented_gap:.1f}%</td>
            <td>-</td>
        </tr>
    </table>

    <h2>Checkpoint Learning Curves</h2>
    <h3>Baseline Checkpoints</h3>
    <table>
        <tr><th>Checkpoint</th><th>Train PPL</th><th>Test PPL</th><th>Gap</th></tr>
        {baseline_rows}
    </table>

    <h3>Augmented Checkpoints</h3>
    <table>
        <tr><th>Checkpoint</th><th>Train PPL</th><th>Test PPL</th><th>Gap</th></tr>
        {augmented_rows}
    </table>

    <div class="interpretation">
        <h2>ADR-008 Interpretation</h2>
        <p>{interpretation}</p>
    </div>

    <h2>Statistical Notes</h2>
    <ul>
        <li>All perplexity values include 95% bootstrap confidence intervals</li>
        <li>Effect size > 10% is considered significant</li>
        <li>Effect size 5-10% is considered modest</li>
        <li>Effect size < 5% may be noise given small test set</li>
    </ul>
</body>
</html>"""

    from datetime import datetime

    # Extract data
    baseline_best = report["baseline"]
    augmented_best = report["augmented"]

    delta_pct = (
        (augmented_best["best_test_ppl"] - baseline_best["best_test_ppl"]) / baseline_best["best_test_ppl"] * 100
    )
    delta_class = "metric-good" if delta_pct < -5 else ("metric-bad" if delta_pct > 5 else "metric-neutral")

    # Generate checkpoint rows
    baseline_rows = ""
    for cp in baseline.get("checkpoints", []):
        baseline_rows += f"<tr><td>{cp['name']}</td><td>{cp['train']['mean_perplexity']:.2f}</td><td>{cp['test']['mean_perplexity']:.2f}</td><td>{cp['gap']['relative_pct']}</td></tr>\n"

    augmented_rows = ""
    for cp in augmented.get("checkpoints", []):
        augmented_rows += f"<tr><td>{cp['name']}</td><td>{cp['train']['mean_perplexity']:.2f}</td><td>{cp['test']['mean_perplexity']:.2f}</td><td>{cp['gap']['relative_pct']}</td></tr>\n"

    return html.format(
        timestamp=datetime.now().isoformat(),
        baseline_train_samples=baseline.get("train_samples", "N/A"),
        baseline_checkpoints=len(baseline.get("checkpoints", [])),
        augmented_train_samples=augmented.get("train_samples", "N/A"),
        augmented_checkpoints=len(augmented.get("checkpoints", [])),
        test_samples=baseline.get("test_samples", "N/A"),
        eval_mode=baseline.get("eval_mode", "completion"),
        baseline_best_name=baseline_best.get("best_name", "N/A"),
        augmented_best_name=augmented_best.get("best_name", "N/A"),
        baseline_test_ppl=baseline_best["best_test_ppl"],
        augmented_test_ppl=augmented_best["best_test_ppl"],
        baseline_train_ppl=baseline_best.get("best_train_ppl", 0),
        augmented_train_ppl=augmented_best.get("best_train_ppl", 0),
        baseline_gap=baseline_best["best_gap"] * 100,
        augmented_gap=augmented_best["best_gap"] * 100,
        delta_pct=delta_pct,
        delta_class=delta_class,
        baseline_rows=baseline_rows,
        augmented_rows=augmented_rows,
        interpretation=report["interpretation"].replace("\n", "<br>"),
    )
