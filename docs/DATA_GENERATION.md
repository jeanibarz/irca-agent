# Data Generation

This document details the process of generating synthetic agent traces using `irca-agent`.

## The Generation Pipeline

The generation pipeline is designed to create high-quality, structured data for training function-calling agents. It uses [Microsoft Guidance](https://github.com/guidance-ai/guidance) to enforce strict output schemas, ensuring that generated traces are always syntactically valid.

### 1. Source Data
Generation starts with a **User Query Dataset**. This dataset contains:
- `user_query`: The question or instruction.
- `sql_query` (optional): Ground truth if deriving from text-to-SQL tasks.
- `context`: Additional context if needed.

### 2. The Iterative Resolution Cycle
For each user query, the `TraceGenerator` simulates an agent reasoning process:

1. **Thought Generation**: The model generates a chain-of-thought reasoning step.
2. **Action Choice**: The model explicitly chooses between `call_function` and `final_answer`.
3. **Function Execution**:
   - If `call_function` is chosen, the model is constrained to generate valid JSON matching a function schema.
   - The system "executes" this function (using mock return values or actual logic).
   - The result is fed back into the context.
4. **Final Answer**: When the model has enough information, it generates a final answer.

### 3. Data Augmentation
To prevent overfitting and improve generalization, we apply several augmentation techniques:

#### Function Removal (Negative Examples)
We randomly remove the function that would solve the user's query from the available tools.
- **Goal**: Teach the model to recognize when it *cannot* solve a problem.
- **Expected Outcome**: The model should incorrectly try to answer or (ideally, with training) state that the function is missing.

#### Function Shuffling
The list of available functions in the system prompt is shuffled.
- **Goal**: Prevent the model from memorizing function positions (e.g., "always call the first function").

#### Prompt Style Randomization
We vary the formatting of the prompt slightly (e.g., different newline styles, spacing).
- **Goal**: Make the model robust to subtle prompt variations.

## Usage

### Generating Traces via CLI

```bash
# Basic generation
irca generate traces

# Specify a model
irca generate traces --model mistralai/Mistral-7B-Instruct-v0.2

# Limit number of samples
irca generate traces --limit 100

# Resume from a specific index
irca generate traces --start 50
```

### Dataset Output Format
Generated datasets can be pushed to HuggingFace. They typically contain:
- `id`: Unique identifier.
- `user_query`: The original query.
- `trace`: The JSON-structured trace.
- `prompt`: The full prompt used (for debugging).
- `completion`: The target completion text for training.

## Extending Generation

### Adding New Functions
Function schemas are defined in `src/dataset_generation/function_variants`. To add a new domain:
1. Create a definition file (e.g., `crypto_tools.json`).
2. Implement the mock logic in `functions_factory.py`.
3. Register it in the generator configuration.

### Modifying the Agent Loop
The core loop is in `src/core/generation/trace_generator.py`. You can subclass `TraceGenerator` to implement different reasoning patterns (e.g., ReAct, Plan-and-Solve).
