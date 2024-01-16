import json
import random


from utils import extract_and_remove

# CHAT TEMPLATE EXAMPLE:
# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# hearties. None at all.</s>


def build_full_prompt(sample):
    # Extracting parts from the sample
    system_instructions = sample.get("system_instructions", "")
    example = sample.get("example", "")
    available_functions_json = sample.get("available_functions_json", [])
    user_query = sample.get("user_query", "")
    assistant_completion = sample.get("assistant_completion", "")

    # Constructing the full prompt
    full_prompt = f"""
### INSTRUCTIONS
{system_instructions}

EXAMPLE:
{example}

The functions available to you are described below.

### FUNCTIONS AVAILABLE
{available_functions_json}

Note: ensure you only use information provided in the context above or below. Do not make up information you are missing. If some information is missing and can't be gathered by using functions available to you, just admit you don't know and ask the user for clarification or more information.

### USER QUERY
{user_query}

### ITERATIVE RESOLUTION CYCLE
{assistant_completion}
"""
    return full_prompt


def parse_corrected_agent_trace(full_prompt):
    # Extracting and removing parts from the full_prompt
    system_instructions, full_prompt = extract_and_remove(
        start_marker="### INSTRUCTIONS",
        end_marker="EXAMPLE:",
        full_prompt=full_prompt,
        include_start_marker=False,
        include_end_marker=False,
    )
    example, full_prompt = extract_and_remove(
        "EXAMPLE:",
        "<|wait|>",
        full_prompt,
        include_start_marker=False,
        include_end_marker=True,
    )
    available_functions_json, full_prompt = extract_and_remove(
        "### FUNCTIONS AVAILABLE",
        "\n\n",
        full_prompt,
        include_start_marker=False,
        include_end_marker=False,
    )
    user_query, full_prompt = extract_and_remove(
        "### USER QUERY",
        "### ITERATIVE RESOLUTION CYCLE",
        full_prompt,
        include_start_marker=False,
        include_end_marker=False,
    )
    assistant_completion, full_prompt = extract_and_remove(
        "### ITERATIVE RESOLUTION CYCLE",
        None,
        full_prompt,
        include_start_marker=False,
        include_end_marker=False,
    )

    # Update the sample with parsed data
    parsed_data = {
        "system_instructions": system_instructions,
        "example": example,
        "available_functions_json": available_functions_json,
        "user_query": user_query,
        "assistant_completion": assistant_completion,
    }
    return parsed_data


iteration_nbr = 0


def randomize_newline_characters(text):
    newline_choice = random.choice(["\n", "\r\n"])
    return text.replace("\n", newline_choice)


def randomize_system_instructions_formatting(system_instructions):
    # Randomly choose a formatting style
    format_style = random.choice([1, 2])

    if format_style == 2:
        # Replace markers and add random newlines
        replacements = {
            "### INSTRUCTIONS": "",
            "### FUNCTIONS AVAILABLE": "<|FUNCTIONS AVAILABLE|>",
            "### USER QUERY": "<|USER QUERY|>",
        }
        for old, new in replacements.items():
            newline_count = random.choice(["", "\n", "\n\n"])
            system_instructions = system_instructions.replace(old, newline_count + new)

    # Randomize the newline characters in the system instructions
    system_instructions = randomize_newline_characters(system_instructions)

    return system_instructions


def format_instruction(sample):
    global iteration_nbr
    print(f"format {iteration_nbr}")
    iteration_nbr += 1

    full_prompt = sample["corrected_agent_trace"][0]["value"]
    full_prompt = full_prompt.replace("\r\n", "\n")

    # Split the full prompt into its constituent parts:
    # system_instructions,
    # example,
    # available_functions_json,
    # user_query,
    # assistant_completion

    parsed_data = parse_corrected_agent_trace(full_prompt)

    # Randomize the system_instructions formatting
    parsed_data["system_instructions"] = randomize_system_instructions_formatting(
        parsed_data["system_instructions"]
    )

    available_functions_json = parsed_data.get("available_functions_json", [])
    if available_functions_json:
        available_functions = json.loads(available_functions_json)
        random.shuffle(available_functions)
        parsed_data["available_functions_json"] = [
            json.dumps(available_functions, indent=random.choice([None, 3, 4]))
        ]
    formatted_sample = build_full_prompt(parsed_data)
    return formatted_sample
