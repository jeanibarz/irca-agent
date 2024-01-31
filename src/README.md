## src/utils.py

`utils.py` is a utility module in the IRCA-Agent project that provides essential helper functions used across various parts of the project. This file includes functions for text processing, model parameter analysis, and data manipulation. Here's an overview of the key functions and their purposes:

### extract_and_remove
- **Purpose**: To extract a specific section from a text prompt based on given start and end markers, and optionally include these markers in the extracted section. It also removes the extracted section from the original prompt.
- **Use Case**: This function is particularly useful in processing text inputs or outputs where certain parts of the text need to be isolated or removed based on specific markers.

### print_trainable_parameters
- **Purpose**: To print the number of trainable parameters in a given model. This includes a breakdown of all parameters versus trainable parameters, along with the percentage of trainable parameters.
- **Use Case**: Vital for model analysis, especially during the development and fine-tuning stages of the LLM, allowing for a better understanding of the model's complexity and capacity for learning.

### format_generate_user_query
- **Purpose**: To format a text prompt for generating potential user queries for an AI assistant, including instructions for creativity and spontaneity in the queries.
- **Use Case**: Useful in simulating and generating natural user queries for training or testing the AI assistant's response capabilities.

### shuffle_json_functions
- **Purpose**: To shuffle a list of functions (in JSON format) for randomization purposes.
- **Use Case**: This function can be used in scenarios where a randomized order of functions is necessary, such as in dataset generation or in presenting varied use-case scenarios for the AI assistant.
