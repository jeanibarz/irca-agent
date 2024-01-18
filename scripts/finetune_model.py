import os
import argparse
import json

from dotenv import load_dotenv

from model_training import finetune_model
from config import defaults

load_dotenv()

# Default path to the defaults.json file (assuming it's in the same directory as the script)
default_json_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "defaults.json"
)


def load_config(model_type):
    model_config = defaults.get(model_type)
    if model_config is None:
        raise ValueError(f"No configuration found for model type: {model_type}")

    # Constructing the model_name
    model_name_prefix = model_config["model_name"].split("_")[0]
    version_suffix = f"_irca_agent_{defaults['version']}.gguf"
    model_name = model_name_prefix + version_suffix

    return {
        "dataset": defaults["dataset"],
        "auth_token": defaults["auth_token"],
        "push_to_hub": defaults["push_to_hub"],
        "lora_r": defaults["lora_r"],
        "lora_dropout": defaults["lora_dropout"],
        "lora_alpha": defaults["lora_alpha"],
        "num_train_epochs": defaults["num_train_epochs"],
        "learning_rate": defaults["learning_rate"],
        "base_model": model_config["base_model"],
        "model_name": model_name,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Fine-tuning Script")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of the model to train (e.g., 'mistral', 'tinyllama').",
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        default=None,
        help="HF authentication token, only used if downloading a private dataset.",
    )
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    # Load configuration based on the model type
    config = load_config(args.model_type)

    # Override auth_token from command line if provided
    if args.auth_token is not None:
        config["auth_token"] = args.auth_token

    finetune_model(config)


if __name__ == "__main__":
    main()
