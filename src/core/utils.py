import random
import json


def extract_and_remove(
    start_marker,
    end_marker,
    full_prompt,
    include_start_marker=False,
    include_end_marker=False,
):
    # Find the start and end of the section
    if start_marker is None:
        start_idx = 0
    else:
        start_idx = full_prompt.find(start_marker)
        if start_idx == -1:
            return "", full_prompt

    if include_start_marker or start_marker is None:
        start_extract_idx = start_idx
    else:
        start_extract_idx = start_idx + len(start_marker)

    if end_marker is None:
        end_idx = len(full_prompt)
    else:
        end_idx = full_prompt.find(end_marker, start_idx)
        if end_idx == -1:
            extracted_section = full_prompt[start_extract_idx:].strip()
            full_prompt = full_prompt[:start_idx].strip()
            return extracted_section, full_prompt

    if include_end_marker and end_marker is not None:
        end_extract_idx = end_idx + len(end_marker)
    else:
        end_extract_idx = end_idx

    extracted_section = full_prompt[start_extract_idx:end_extract_idx].strip()
    full_prompt = full_prompt[:start_idx].strip() + full_prompt[end_extract_idx:].strip()

    return extracted_section, full_prompt


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
    )


def format_generate_user_query(available_functions, user_query=None):
    text = (
        "The following functions are available to an AI ASSISTANT:\n"
        + available_functions
        + "\n\n"
        + "Please generate a potential user query for the AI ASSISTANT. Be very creative ! The query should be natural, and look like spontaneous. Feel free to imagine any kind of interesting scenarios you can think of. You can also add some typos some times or make some small grammar errors.\nSure ! Here's a potential query that a user could make to the AI ASSISTANT given the available functions described above: "
    )
    if user_query:
        # for training purposes, we fill the user query
        # when doing inference, we let the LLM generate the user query
        text += user_query + "</s>"
    return text


def shuffle_json_functions(available_functions):
    # Shuffle the list of functions
    functions = json.loads(available_functions)
    random.shuffle(functions)

    available_functions = json.dumps(functions)
    return available_functions
