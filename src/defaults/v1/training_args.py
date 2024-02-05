# there arguments are used in src/model_finetuning.py
training_args = {
    "dataset": "JeanIbarz/irca_agent_dataset_v5-5acc",
    "push_to_hub": False,
    "version": "v5-6",
    "lora_r": 32,
    "lora_dropout": 0.05,
    "lora_alpha": 64,
    "num_train_epochs": 5,
    "learning_rate": 5e-5,
    "model_settings": {
        "mistral": {
            "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
            "model_name": "Mistral-7B-Instruct-v0.2-with-data-augmentation",
        },
        "tinyllama": {
            "base_model": "Doctor-Shotgun/TinyLlama-1.1B-32k",
            "model_name": "TinyLlama-7B-v0.1",
        },
        "eagle": {
            "base_model": "RWKV/v5-Eagle-7B",
            "model_name": "Eagle-7B",
        },
        "snorkel": {
            "base_model": "snorkelai/Snorkel-Mistral-PairRM-DPO",
            "model_name": "Snorkel-Mistral-7B-no-data-augmentation",
        },
        "eval": {
            "base_model": "/workspace/models/finetuned_models/Snorkel-Mistral-7B_irca_agent_v5-6.gguf/checkpoint-122",
            "model_name": "Snorkel-Mistral-7B-irca_agent_v5-6-checkpoint-122",
        },
    },
}
