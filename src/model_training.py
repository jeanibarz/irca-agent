import os
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from huggingface_hub import login

from dotenv import load_dotenv

from utils import print_trainable_parameters
from prompt_builder import format_instruction

from dotenv import load_dotenv

load_dotenv("/workspace/.env")


def finetune_model(config):
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    # dataset = load_dataset(args.dataset)
    dataset = load_dataset(config["dataset"])
    # dataset = dataset.filter(lambda x: len(x["corrected_user_trace"]) > 0)

    # base model to finetune
    model_id = config["base_model"]

    # BitsAndBytesConfig to quantize the model int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, use_cache=False, device_map="auto"
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
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
        lora_dropout=config["lora_dropout"],
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # print the number of trainable model params
    print_trainable_parameters(model)

    model_args = TrainingArguments(
        output_dir=os.path.join(
            "/workspace", "models", "finetuned_models", config["model_name"]
        ),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=3,
        gradient_checkpointing=False,
        optim="adamw_8bit",
        logging_steps=2,
        save_strategy="epoch",
        # save_steps=10,
        learning_rate=config["learning_rate"],
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,
    )

    train_dataset = dataset["train"]
    trainer = SFTTrainer(
        model=model,
        # train_dataset=dataset["train"].shard(num_shards=20, index=0),
        train_dataset=train_dataset,  # .select(range(0, 18)),
        # eval_dataset=dataset["train"].select(range(18, 20)),
        peft_config=peft_config,
        max_seq_length=4096,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=model_args,
    )

    # train
    trainer.train()

    # save model
    trainer.save_model()

    if config["push_to_hub"]:
        trainer.model.push_to_hub(config["model_name"])

    torch.cuda.empty_cache()
