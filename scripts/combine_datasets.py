import os
from datetime import datetime
import json

import argilla as rg
from dotenv import load_dotenv

load_dotenv()

rg.init(
    api_url=os.getenv("ARGILLA_API_URL"),
    api_key=os.getenv("ARGILLA_API_KEY"),
    # extra_headers={"Authorization": f"Bearer {"HF_TOKEN"}"}
)

# Create a local copy of the data (don't remove the .pull() /!\)
ds1 = rg.FeedbackDataset.from_argilla(
    name="irca_agent_dataset_v5-2acc", workspace="irca_agent"
).pull()
ds2 = rg.FeedbackDataset.from_argilla(
    name="irca_agent_dataset_v5-3", workspace="irca_agent"
).pull()
# ds3 = rg.FeedbackDataset.from_argilla(
#     name="irca_agent_dataset_v4-4", workspace="irca_agent"
# ).pull()

# Remove non filtered records
new_records = []
for record in ds1.records + ds2.records:
    if not record.responses or record.responses[0].status != "submitted":
        continue
    try:
        available_functions = (
            record.fields["available_functions"].replace("\r", "").strip("\n")
        )
        user_query = record.fields["user_query"].replace("\r", "").strip("\n")
        # user_query = record.responses[0].values["corrected_user_query"].value
        try:
            agent_trace = record.responses[0].values["corrected_agent_trace"].value
        except KeyError:
            try:
                agent_trace = (
                    record.responses[0].values["corrected_irca_agent_trace"].value
                )
            except KeyError:
                agent_trace = record.responses[0].values["corrected_user_trace"].value
        agent_trace = agent_trace.replace("\r", "").strip("\n")
        new_record = rg.FeedbackRecord(
            fields={
                "available_functions": available_functions,
                "user_query": user_query,
                "agent_trace": agent_trace,
            },
            responses=[
                {
                    "values": {
                        "corrected_agent_trace": {
                            "value": agent_trace,
                        },
                    },
                },
            ],
        )
        new_records.append(new_record)
    except KeyError:
        print("skipping")
        pass

new_ds = rg.FeedbackDataset.from_argilla(
    name="irca_agent_dataset_v5-3acc", workspace="irca_agent"
)
new_ds.add_records(new_records)
