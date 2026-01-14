import json
import os

import argilla as rg
from dotenv import load_dotenv
from guided_generation_gpt4 import GuidedPromptGenerator

load_dotenv()

rg.init(
    api_url=os.getenv("ARGILLA_API_URL"),
    api_key=os.getenv("ARGILLA_API_KEY"),
    # extra_headers={"Authorization": f"Bearer {"HF_TOKEN"}"}
)


def main():
    # 1-5  : checkpoint-10
    # 6-10 : checkpoint-20
    # 11-15: checkpoint-30
    # 16-20: checkpoint-40
    # 21-25: checkpoint-50
    # Define a configuration for the PromptGenerator

    # to copy a model from host to docker volume:
    #  docker cp C:\Users\ibarz\.cache\lm-studio\models\TheBloke\Mistral-7B-Instruct-v0.2-DARE-GGUF dummy:/root/TheBloke/Mistral-7B-Instruct-v0.2-DARE-GGUF

    config = {
        "records_nbr_to_generate": 20,
        # "model_name_or_path": "/workspace/models/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
        # "model_name_or_path": "/workspace/models/TheBloke/Mixtral_11Bx2_MoE_19B-GGUF/mixtral_11bx2_moe_19b.Q5_K_M.gguf",
        # "model_name_or_path": "/workspace/models/finetuned_models/Mistral-7B-v0.1_irca_agent_v1.gguf/checkpoint-30",
        # "model_name_or_path": "/workspace/models/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24",
        "model_name_or_path": "/workspace/models/finetuned_models/Mistral-7B-v0.1_user_query_generator_v1.gguf/checkpoint-10",
        "min_funcs": 1,
        "max_funcs": 20,
        "user_request_satisfiable": True,
        "rg": rg,
        "argilla": {
            "name": "irca_user_query_dataset_v3-1",
            "workspace": "irca_agent",
        },
        "use_gpt4": False,
    }

    # Initialize the PromptGenerator with the given configuration
    prompt_generator = GuidedPromptGenerator(config)

    for i in range(config["records_nbr_to_generate"]):
        print(f"Generation n°{i}")

        # Generate a random subset of functions and print the number of functions selected
        available_functions_dict = prompt_generator.generate_random_subset()
        available_functions_json = json.dumps(available_functions_dict)

        print("Selected functions:", available_functions_dict)
        print("Number of selected functions:", len(available_functions_dict))

        prompt_generator.generate_user_query(available_functions=available_functions_json)


if __name__ == "__main__":
    main()
