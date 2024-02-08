import os

from dotenv import load_dotenv

import argilla as rg

from core.trace_generator import GuidedTraceGenerator
from core.utils import shuffle_json_functions

load_dotenv()

# rg.init(
#     api_url=os.getenv("ARGILLA_API_URL"),
#     api_key=os.getenv("ARGILLA_API_KEY"),
#     # extra_headers={"Authorization": f"Bearer {"HF_TOKEN"}"}
# )


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
        "records_nbr_to_generate": 5,
        # "model_name_or_path": "/workspace/models/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
        # "model_name_or_path": "/workspace/models/TheBloke/Mixtral_11Bx2_MoE_19B-GGUF/mixtral_11bx2_moe_19b.Q5_K_M.gguf",
        # "model_name_or_path": "/workspace/models/finetuned_models/Mistral-7B-v0.1_irca_agent_v1.gguf/checkpoint-30",
        # "model_name_or_path": "/workspace/models/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24",
        # "model_name_or_path": "/workspace/models/finetuned_models/Mistral-7B-v0.1_irca_agent_v5-5-1.gguf/checkpoint-31",
        # "model_name_or_path": "/workspace/models/finetuned_models/Starling-LM-7B-alpha-irca_agent_v5-5-1.gguf/checkpoint-31",
        # "model_name_or_path": "/workspace/models/finetuned_models/TinyLlama-7B-v0.1_irca_agent_v5-5-3.gguf/checkpoint-84",
        "model_name_or_path": "/workspace/models/finetuned_models/Mistral-7B-Instruct-v0.2-with-data-augmentation_irca_agent_v5-6.gguf/checkpoint-97",
        "min_funcs": 1,
        "max_funcs": 20,
        "user_request_satisfiable": True,
        "rg": rg,
        "argilla_source": {
            "name": "irca_user_query_dataset_v4",
            "workspace": "irca_agent",
        },
        "argilla_target": {
            "name": "irca_agent_dataset_v5-5",
            "workspace": "irca_agent",
        },
        "use_gpt4": False,
    }

    # # Get argilla remote dataset
    # ds = rg.FeedbackDataset.from_argilla(
    #     name=config["argilla_target"]["name"],
    #     workspace=config["argilla_target"]["workspace"],
    # )

    # Initialize the PromptGenerator with the given configuration
    trace_generator = GuidedTraceGenerator(model_name_or_path=config["model_name_or_path"])

    # src_ds = rg.FeedbackDataset.from_argilla(**config["argilla_source"]).pull()
    # src_ds = rg.FeedbackDataset.from_huggingface("JeanIbarz/irca_user_query_dataset_v4")
    from datasets import load_dataset, load_from_disk

    # src_ds = load_dataset("JeanIbarz/irca_user_query_dataset_v4")
    src_ds = load_from_disk("/workspace/datasets/irca_user_query_dataset_v5-6")

    dataset_type = "disk"  # can be 'argilla', 'huggingface', or 'disk'
    if dataset_type == "disk":
        records = src_ds
    elif dataset_type == "huggingface":
        records = src_ds["train"]
    else:
        records = src_ds.records

    # for i in range(66, len(src_ds)):
    start = 0
    # need to resume at 75 !!!!
    # n = 40
    for i in range(start, len(records)):
        print(f"Generation n°{i}")

        # Generate a random subset of functions and print the number of functions selected
        # available_functions_dict = prompt_generator.generate_random_subset()
        # available_functions_json = json.dumps(available_functions_dict)
        try:
            if dataset_type in ["huggingface", "disk"]:
                available_functions = records[i]["available_functions"]
                user_query = records[i]["corrected_user_query"][0]["value"]
            else:
                available_functions = records[i].fields["available_functions"]
                user_query = src_ds.records[i].responses[0].values["corrected_user_query"].value
        except IndexError:
            print("Skipping generation...")
            continue

        shuffled_available_functions = shuffle_json_functions(available_functions=available_functions)

        # print("Selected functions:", available_functions_dict)
        # print("Number of selected functions:", len(available_functions_dict))

        # traces = trace_generator.generate_traces(
        #     available_functions=available_functions,
        #     user_query=user_query,
        # )
        traces = [
            trace_generator.generate_single_trace(
                available_functions=shuffled_available_functions,
                user_query=user_query,
            )
        ]

        print("trace generated")

        # # Create Argilla records
        # records = [
        #     trace_generator.trace_to_argilla_record(
        #         available_functions=available_functions,
        #         user_query=user_query,
        #         trace=trace,
        #     )
        #     for trace in traces
        # ]

        # # Push records to remote dataset
        # ds.add_records(records)
        # print(f"{len(records)} records pushed to argilla")


if __name__ == "__main__":
    main()
