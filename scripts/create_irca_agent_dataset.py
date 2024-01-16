import os
from datetime import datetime
import json

import argilla as rg
from dotenv import load_dotenv

load_dotenv()

WORKSPACE = "irca_agent"
DATASET_NAME = "irca_agent_dataset_v5-5"

rg.init(
    api_url=os.getenv("ARGILLA_API_URL"),
    api_key=os.getenv("ARGILLA_API_KEY"),
    # extra_headers={"Authorization": f"Bearer {"HUGGINGFACE_TOKEN"}"}
)

DEFAULT_WORKSPACE = "irca_agent"
workspace = rg.Workspace.from_name(DEFAULT_WORKSPACE)

user = rg.User.me()
print(user)


def create_and_push_dataset(name, workspace=DEFAULT_WORKSPACE):
    dataset = rg.FeedbackDataset(
        fields=[
            # rg.TextField(name="fct_schemas", title="Functions schemas:"),
            # rg.TextField(name="fct_names", title="Functions names:"),
            rg.TextField(name="agent_trace", title="Agent Trace:"),
            rg.TextField(name="user_query", title="User Query:"),
            rg.TextField(name="available_functions", title="Available Functions:"),
            # rg.TextField(name="fct_used", title="Functions called:"),
        ],
        questions=[
            # rg.TextQuestion(
            #     name="corrected_user_query",
            #     title="Corrected User Query:",
            #     required=True,
            # ),
            rg.TextQuestion(
                name="corrected_agent_trace",
                title="Corrected Agent Trace:",
                required=True,
            ),
        ],
    )
    dataset.push_to_argilla(name=name, workspace=workspace)


create_and_push_dataset(name=DATASET_NAME, workspace=WORKSPACE)
