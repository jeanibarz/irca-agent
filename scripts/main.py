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

DEFAULT_WORKSPACE = "function_calling"
workspace = rg.Workspace.from_name(DEFAULT_WORKSPACE)

user = rg.User.me()
print(user)


def create_and_push_dataset(name, workspace=DEFAULT_WORKSPACE):
    dataset = rg.FeedbackDataset(
        fields=[
            rg.TextField(name="available_functions", title="Available functions:"),
            rg.TextField(name="user_query", title="User query:"),
            rg.TextField(name="agent_scratchpad", title="Agent scratchpad:"),
            rg.TextField(name="original_completion", title="LLM completion:"),
        ],
        questions=[
            rg.TextQuestion(
                name="corrected_completion",
                title="Provide the ideal completion:",
                required=True,
            ),
        ],
    )
    dataset.push_to_argilla(name=name, workspace=workspace)


def create_and_push_user_query_dataset(name, workspace=DEFAULT_WORKSPACE):
    dataset = rg.FeedbackDataset(
        fields=[
            rg.TextField(name="available_functions", title="Available functions:"),
            rg.TextField(name="original_user_query", title="User query:"),
        ],
        questions=[
            rg.LabelQuestion(
                name="is_request_feasible",
                title="Is the user query feasible given the available functions ?",
                labels={"YES": True, "NO": False},
                required=True,
            ),
            rg.TextQuestion(
                name="corrected_user_query",
                title="Provide the corrected user query:",
                required=True,
            ),
        ],
    )
    dataset.push_to_argilla(name=name, workspace=workspace)


# ds = rg.FeedbackDataset.from_huggingface("argilla/mistral-vs-llama")


def generate_completion(available_functions, user_query, agent_scratchpad):
    from openai import OpenAI
    from prompt.function_calling_oneshot import prompt_template

    client = OpenAI()

    prompt = prompt_template.format(
        available_functions=available_functions,
        user_query=user_query,
        agent_scratchpad=agent_scratchpad,
    )

    print("prompt after instantiation:")
    print(prompt)

    response = client.chat.completions.create(
        # model="gpt-3.5-turbo-instruct",
        model="gpt-4-0613",
        messages=[{"role": "user", "content": prompt}],
        # prompt=prompt,
        max_tokens=1024,
        stop=["<wait_for_output>"],
    )
    assert len(response.choices) == 1
    first_choice = response.choices[0]
    assert first_choice.finish_reason == "stop"
    return first_choice.text


def create_record(
    available_functions, user_query, agent_scratchpad, original_completion=None
):
    if not original_completion:
        # Run the LLM to generate a completion
        original_completion = generate_completion(
            available_functions, user_query, agent_scratchpad
        )

    corrected_completion = original_completion

    # Create record
    record = rg.FeedbackRecord(
        fields={
            "available_functions": available_functions,
            "user_query": user_query,
            "original_completion": original_completion,
        },
        responses=[
            {
                "values": {
                    "corrected_completion": {
                        "value": corrected_completion,
                    },
                },
            },
        ],
    )
    return record


create_and_push_user_query_dataset(name="user_query_ds", workspace="function_calling")

# test_create_record = {
#     "available_functions": json.dumps(
#         {
#             "name": "get_news_headlines",
#             "description": "Get the latest news headlines",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "country": {
#                         "type": "string",
#                         "description": "The country for which to fetch news",
#                     }
#                 },
#                 "required": ["country"],
#             },
#         }
#     ),
#     "user_query": "Can you tell me the latest news headlines for the United States?",
#     "agent_scratchpad": "",
# }
# test_record = create_record(**test_create_record)

# ds = rg.FeedbackDataset.from_argilla(
#     name="thinktrack_function_calling_v1-2", workspace=DEFAULT_WORKSPACE
# )

# ds.add_records([test_record])

# remote_dataset = rg.FeedbackDataset.from_argilla(
#     name="thinktrack_function_calling_v1", workspace="lucygpt"
# )

# remote_dataset.add_records(records)
