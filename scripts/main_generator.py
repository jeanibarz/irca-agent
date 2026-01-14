"""
Main Trace Generator Script

Generates agent traces from user queries for dataset creation.

Usage:
    python scripts/main_generator.py
"""

import argparse
import logging
import sys

from datasets import load_from_disk

from config import get_settings
from core.trace_generator import GuidedTraceGenerator
from core.utils import shuffle_json_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate agent traces from user queries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="irca_user_query_dataset_v5-6",
        help="Name of the source dataset (in datasets directory)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index in the dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of traces to generate",
    )
    return parser.parse_args()


def main():
    """Main function to generate traces."""
    args = parse_arguments()
    settings = get_settings()

    # Determine model path
    if args.model:
        model_path = args.model
    else:
        # Default: use latest finetuned model
        model_path = str(
            settings.finetuned_models_path
            / "Mistral-7B-Instruct-v0.2-with-data-augmentation_irca_agent_v5-6.gguf"
            / "checkpoint-97"
        )
        logger.info(f"Using default model: {model_path}")

    # Load source dataset
    dataset_path = settings.datasets_path / args.dataset
    logger.info(f"Loading dataset from: {dataset_path}")

    try:
        src_ds = load_from_disk(str(dataset_path))
        logger.info(f"Loaded {len(src_ds)} records from dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Initialize trace generator
    logger.info(f"Initializing trace generator with model: {model_path}")
    trace_generator = GuidedTraceGenerator(model_name_or_path=model_path)

    # Process records
    records = src_ds
    start_idx = args.start
    end_idx = len(records) if args.limit is None else min(start_idx + args.limit, len(records))

    logger.info(f"Generating traces for records {start_idx} to {end_idx - 1}")

    for i in range(start_idx, end_idx):
        logger.info(f"Generation {i}/{end_idx - 1}")

        try:
            available_functions = records[i]["available_functions"]
            user_query = records[i]["corrected_user_query"][0]["value"]
        except (KeyError, IndexError) as e:
            logger.warning(f"Skipping record {i}: {e}")
            continue

        # Shuffle functions to reduce positional bias
        shuffled_available_functions = shuffle_json_functions(
            available_functions=available_functions
        )

        # Generate trace
        traces = [
            trace_generator.generate_single_trace(
                available_functions=shuffled_available_functions,
                user_query=user_query,
            )
        ]

        logger.info(f"Trace {i} generated successfully")

        # TODO: Add option to save traces to dataset
        # For now, traces are just generated and logged

    logger.info("Trace generation completed")


if __name__ == "__main__":
    main()

