defaults = {
    "dataset": "JeanIbarz/irca_agent_dataset_v5-3acc",
    "auth_token": None,
    "push_to_hub": False,
    "version": "v5-5-3",
    "lora_r": 32,
    "lora_dropout": 0.05,
    "lora_alpha": 64,
    "num_train_epochs": 5,
    "learning_rate": 5e-5,
    "mistral": {
        # "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "base_model": "/workspace/models/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24",
        "model_name": "Mistral-7B-Instruct-v0.1",
    },
    "tinyllama": {
        "base_model": "Doctor-Shotgun/TinyLlama-1.1B-32k",
        "model_name": "TinyLlama-7B-v0.1",
    },
}
