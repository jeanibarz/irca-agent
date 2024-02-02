## src/generation.py

`generation.py` is a key module in the IRCA-Agent project, focusing on generating user queries based on a set of available functions. This file includes a class that facilitates the generation of function subsets and user queries using distribution functions and OpenAI's GPT models. Key functionalities of this module are as follows:

### Class: PromptGenerator
- **Purpose**: To generate random subsets of functions and user queries based on the provided configuration parameters.
- **Key Methods**:
  - `generate_random_subset`: Generates a random subset of functions using a specified distribution function, with an option to return the subset in JSON format.
  - `generate_user_query`: Generates a user query based on the available functions and a specified satisfiability condition using OpenAI's GPT models.

#### Details:
- **Initialization**: Sets default configuration values for the number of functions to sample and distribution parameters, and updates them with any provided configuration.
- **Distribution Functions**: Supports both uniform and beta distributions for generating function subsets, offering flexibility in sampling.
- **User Query Generation**: Uses a template and OpenAI's GPT models to generate a user query that corresponds to the available functions.

### User Query Generation Process:
1. **Template Formatting**: Formats a prompt template with the available functions and satisfiability condition.
2. **Query Generation**: Utilizes OpenAI's GPT models to generate a user query based on the formatted prompt.
3. **Record Creation and Pushing**: Optionally, the module can create and push records to an Argilla dataset for further analysis or training, capturing details like the available functions and the original user query.

This module plays a significant role in the project by automating the generation of user queries, which are essential for training the LLM and testing its capabilities in understanding and responding to various user requests. The flexibility in distribution functions for function sampling also adds diversity to the generated queries.

## src/guided_generation.py

`guided_generation.py` is an essential module in the IRCA-Agent project, designed to handle the generation of guided prompts and agent traces for training and testing purposes. This file includes classes and methods that facilitate the creation of dynamic and contextually rich prompts based on a set of specified parameters and distribution functions. Here are the key components:

### Class: GuidedPromptGenerator
- **Purpose**: To generate guided prompts based on a configurable set of parameters. This class includes methods for random subset generation, prompt generation, and workflow creation.
- **Key Methods**:
  - `generate_random_subset`: Generates a random subset of functions using a specified distribution function, supporting JSON format output.
  - `generate_irca_agent_trace`: Creates a complete IRCA agent trace, including user queries, agent thoughts, function calls, and final answers.

#### Details:
- **Model Initialization**: The class initializes a model (either `LlamaCpp` or `Transformers` model) based on the provided configuration, which includes model path, minimum and maximum function counts, distribution parameters, etc.
- **Distribution Functions**: Supports uniform and beta distributions for sampling functions, allowing for a diverse range of prompt structures.
- **Workflow Creation**: Generates a detailed workflow that simulates a user query, agent response, and iterative process of function calls and agent thoughts.

### Workflow Generation Process:
1. **User Query Generation**: Utilizes templates and the model to generate a user query based on available functions.
2. **Agent Thought and Function Calls**: Iteratively generates agent thoughts and function calls to create a detailed trace of the agent's decision-making process.
3. **Synthetic Function Output Generation**: Includes the generation of synthetic function outputs to simulate realistic agent interactions.
4. **Final Answer Generation**: Concludes the workflow with the generation of a final answer by the agent.

### Integration with Argilla:
- **Record Creation and Pushing**: If configured, the module can create and push records to an Argilla dataset for further analysis or training.

## src/model_training.py

`model_training.py` is a crucial module in the IRCA-Agent project, dedicated to the fine-tuning and training of the Large Language Model (LLM). This file encompasses various functionalities ranging from model configuration to the actual training process. Below are the key components and their roles:

### finetune_model
- **Purpose**: To fine-tune the LLM based on specific configurations. This function handles everything from loading datasets, setting up model configurations, initializing the tokenizer, to executing the training process.
- **Use Case**: Central to the model development phase, allowing for tailored training of the LLM to meet the specific needs of the IRCA-Agent project.

#### Key Steps in finetune_model:
1. **Model and Tokenizer Initialization**: Loads the base model and tokenizer using configurations for quantization (BitsAndBytesConfig) and parameter-efficient fine-tuning (PEFT) techniques.
2. **Training Configuration**: Sets up training arguments such as output directory, number of epochs, batch size, learning rate, etc., using the `TrainingArguments` class from the transformers library.
3. **Dataset Preparation**: Loads and prepares the dataset for training.
4. **Model Preparation for Training**: Applies PEFT techniques (such as LoRA) to the model and prepares it for training with reduced memory footprint.
5. **Training Execution**: Utilizes the `SFTTrainer` for the training process, which includes custom formatting functions for the training data.
6. **Model Saving and Uploading**: After training, the model is saved locally and optionally pushed to the Hugging Face Hub.

### Additional Utility Functions
- **print_trainable_parameters**: Imported from `utils.py`, this function is used to print the number of trainable parameters in the model, providing insights into the model’s complexity.
- **format_instruction**: Imported from `prompt_builder.py`, this function is used for formatting the training data, ensuring it is in the correct structure for model training.

## src/prompt_builder.py

`prompt_builder.py` is a critical module in the IRCA-Agent project that focuses on constructing and formatting prompts for the Large Language Model (LLM). This file contains functions for building prompts, parsing data, and applying various formatting techniques. The key functionalities are as follows:

### build_full_prompt
- **Purpose**: To construct a comprehensive prompt that includes system instructions, examples, available functions, user queries, and assistant completions.
- **Use Case**: Essential for creating structured and detailed prompts needed for training or querying the LLM.

### parse_corrected_agent_trace
- **Purpose**: To extract and separate different parts of a given full prompt, such as system instructions, examples, available functions, user queries, and assistant completions.
- **Use Case**: Useful for parsing and analyzing the structure of prompts, which can be critical for both training the model and understanding its responses.

### randomize_newline_characters
- **Purpose**: To randomize newline characters in a text, choosing between `\n` and `\r\n`.
- **Use Case**: This function aids in ensuring that the text formatting remains consistent and compatible across different operating systems and platforms.

### randomize_system_instructions_formatting
- **Purpose**: To randomly choose and apply different formatting styles to system instructions within a prompt.
- **Use Case**: Enhances the diversity in prompt presentation, which can be beneficial for training the model to understand and respond to varied text formats.

### format_instruction
- **Purpose**: To format a given sample by parsing its corrected agent trace and applying randomization in newline characters and system instructions formatting.
- **Use Case**: Utilized in the preprocessing stage of data for training, ensuring that the LLM is exposed to a variety of prompt formats.

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
