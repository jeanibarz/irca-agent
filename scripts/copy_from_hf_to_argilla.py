from datasets import load_dataset

config = dict(
    huggingface_dataset="JeanIbarz/irca_user_query_dataset_v4",
    argilla_dataset=dict(
        name="irca_user_query_dataset_v4",
        workspace="irca_agent",
    ),
)

ds_hf = load_dataset(config["huggingface_dataset"])

import argilla as rg
from dotenv import load_dotenv

load_dotenv()

import os

rg.init(
    api_url=os.getenv("ARGILLA_API_URL"),
    api_key=os.getenv("ARGILLA_API_KEY"),
    # extra_headers={"Authorization": f"Bearer {"HF_TOKEN"}"}
)

ds_rg = rg.FeedbackDataset.from_argilla(
    name=config["argilla_dataset"]["name"],
    workspace=config["argilla_dataset"]["workspace"],
)

records = []
for sample in ds_hf["train"]:
    available_functions = sample["available_functions"]
    user_query = sample["corrected_user_query"][0]["value"]
    record = rg.FeedbackRecord(
        fields={
            "available_functions": available_functions,
            "user_query": user_query,
        },
        responses=[
            {
                "values": {
                    "corrected_user_query": {
                        "value": user_query,
                    },
                },
            },
        ],
    )
    records.append(record)

ds_rg.add_records(records)
