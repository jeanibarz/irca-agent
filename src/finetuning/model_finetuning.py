# model_finetuning.py

import os
import argparse
import torch
import logging
import sys
import trl
import datasets
import transformers
import peft
import huggingface_hub
import dotenv
import core.utils as utils
import core.prompt_builder as prompt_builder

from defaults.v1.training_args import training_args

import json
import jsonschema

# Constants
DOTENV_PATH = "/workspace/.env"
WORKSPACE_DIR = "/workspace"
MODELS_DIR = "models"
FINETUNED_MODELS_DIR = "finetuned_models"
LOG_LEVEL = logging.DEBUG
VERSION = "v1"
SCHEMA_PATH = f"/workspace/src/core/schemas/{VERSION}/training_args.json"

# Configure logger
logger = logging.getLogger(__name__)

logger.debug("Loading environment variables from .env file.")
dotenv.load_dotenv(DOTENV_PATH)


def load_and_validate_schema(schema_file, data):
    with open(schema_file, "r") as file:
        schema = json.load(file)
    jsonschema.validate(instance=data, schema=schema)


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(LOG_LEVEL)
    datasets.utils.logging.set_verbosity(LOG_LEVEL)
    transformers.utils.logging.set_verbosity(LOG_LEVEL)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return logger


def setup_peft_config(training_args):
    # LoRA config based on QLoRA paper
    peft_config = peft.LoraConfig(
        r=training_args["lora_r"],
        lora_alpha=training_args["lora_alpha"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=training_args["lora_dropout"],
        task_type="CAUSAL_LM",
    )
    return peft_config


def load_and_prepare_model(config, peft_config):
    huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
    model_id = config["base_model"]

    # BitsAndBytesConfig to quantize the model int-4 config
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pretraining_tp = 1

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # prepare model for training
    model = peft.prepare_model_for_kbit_training(model)
    model = peft.get_peft_model(model, peft_config)

    return model, tokenizer


def setup_training(model, tokenizer, dataset, training_args, peft_config=None):
    model_args = transformers.TrainingArguments(
        output_dir=os.path.join(WORKSPACE_DIR, MODELS_DIR, FINETUNED_MODELS_DIR, training_args["model_name"]),
        num_train_epochs=training_args["num_train_epochs"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        gradient_checkpointing=False,
        optim="adamw_8bit",
        logging_steps=2,
        save_strategy="epoch",
        learning_rate=training_args["learning_rate"],
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,
    )

    train_dataset = dataset["train"]
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=4096,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=prompt_builder.format_instruction,
        args=model_args,
    )
    return trainer


def load_config(model_type):
    model_config = training_args["model_settings"].get(model_type)
    if model_config is None:
        logger.critical(f"No configuration found for model type: {model_type}")
        sys.exit(1)

    model_name_prefix = model_config["model_name"].split("_")[0]
    version_suffix = f"_irca_agent_{training_args['version']}.gguf"
    model_name = model_name_prefix + version_suffix

    return {
        "dataset": training_args["dataset"],
        "push_to_hub": training_args["push_to_hub"],
        "lora_r": training_args["lora_r"],
        "lora_dropout": training_args["lora_dropout"],
        "lora_alpha": training_args["lora_alpha"],
        "num_train_epochs": training_args["num_train_epochs"],
        "learning_rate": training_args["learning_rate"],
        "base_model": model_config["base_model"],
        "model_name": model_name,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Fine-tuning Script")
    parser.add_argument(
        "--model_type",
        type=str,
        default="mistral",
        help="Type of the model to train (e.g., 'mistral', 'tinyllama').",
    )
    return parser


def main():
    logger.info("Setting up logging configuration.")
    setup_logging()

    # Load and validate training_args against JSON schema
    try:
        load_and_validate_schema(SCHEMA_PATH, training_args)
        logger.info("training_args validation successful.")
    except jsonschema.exceptions.ValidationError as e:
        logger.critical(f"training_args validation failed: {e}")
        sys.exit(1)

    parser = parse_arguments()
    args = parser.parse_args()
    logger.info("Command-line arguments parsed successfully.")

    logger.info(f"Model type selected: {args.model_type}")
    config = load_config(args.model_type)
    logger.debug(f"Loaded model configuration: {config}")

    peft_config = setup_peft_config(config)
    logger.info("Starting model loading and preparation.")
    model, tokenizer = load_and_prepare_model(config, peft_config)
    logger.info("Model loading and preparation completed.")

    utils.print_trainable_parameters(model)

    try:
        logger.debug(f"Dataset loading: {config['dataset']}")
        dataset = datasets.load_dataset(config["dataset"])
        logger.debug("Dataset loading successfully completed.")
    except Exception as e:
        logger.critical(f"Failed to load dataset: {e}")
        sys.exit(1)

    trainer = setup_training(model, tokenizer, dataset, config, peft_config)

    logger.info("Starting the model training process...")
    trainer.train()

    logger.info("Model training completed. Saving the model.")
    trainer.save_model()
    logger.info(f"Model {config['model_name']} saved successfully.")

    if config["push_to_hub"]:
        logger.info(f"Pushing model {config['model_name']} to Hugging Face Hub.")
        trainer.model.push_to_hub(config["model_name"])

    torch.cuda.empty_cache()

    logger.info("Model fine-tuning script execution completed.")


if __name__ == "__main__":
    main()

    logger.info("Model fine-tuning script executed successfully. Exiting.")
