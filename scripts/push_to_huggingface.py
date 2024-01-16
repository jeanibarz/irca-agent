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
config = dict(
    name="irca_agent_dataset_v5-3acc",
    workspace="irca_agent",
)
ds = rg.FeedbackDataset.from_argilla(
    name=config["name"], workspace=config["workspace"]
).pull()

# Remove non filtered records
records_to_remove = []
for idx, record in enumerate(ds.records):
    if record.responses[0].status != "submitted":
        records_to_remove.append(record)

for record in records_to_remove:
    ds.records.remove(record)

ds.push_to_huggingface(
    repo_id=f"JeanIbarz/{config['name']}",
    private=True,
    token=os.getenv("HUGGINGFACE_TOKEN"),
)
