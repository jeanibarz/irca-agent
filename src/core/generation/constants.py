"""
Prompt Constants and Templates

Centralized location for all prompt-related constants.
"""

# Step prompt prefixes
THOUGHT_PROMPT = "Thought: "
ACTION_CHOICE_PROMPT = "Action choice: "
CALL_FUNCTION_PROMPT = 'Call function: {"name": "'
FUNCTION_OUTPUT_PROMPT = "Output[{shortuuid}]: "
FINAL_ANSWER_PROMPT = "\n\n### FINAL ANSWER\n"

# Default values
DEFAULT_WORKSPACE = "function_calling"
DEFAULT_DATASET = "user_query_ds"

# Stop sequences
STOP_SEQUENCES = ["\n", "<|wait|>"]
FINAL_ANSWER_STOP = ["### INSTRUCTIONS", "### USER QUERY", "<|wait|>"]

# Action choices
ACTION_CALL_FUNCTION = "call function"
ACTION_FINAL_ANSWER = "final answer"
