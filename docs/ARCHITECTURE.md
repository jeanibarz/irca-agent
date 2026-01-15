# Architecture

This document describes the high-level architecture of `irca-agent`.

## Overview

IRCA-Agent is designed as a modular toolkit for generating synthetic agent traces and fine-tuning language models. The architecture separates domain modeling, trace generation logic, and model training.

```mermaid
graph TD
    CLI[CLI (src/cli)] --> TG[TraceGenerator (src/core/generation)]
    CLI --> FT[Finetuning (src/finetuning)]
    
    TG --> SG[Step Generators (src/core/generation)]
    TG --> DM[Domain Models (src/core/domain)]
    TG --> PB[Prompt Builder (src/core/prompt_builder.py)]
    TG --> DF[Dataset Factory (src/dataset_generation)]
    
    SG --> GL[Guidance Library]
    
    FT --> HF[HuggingFace / PEFT]
```

## Core Components

### 1. Domain Models (`src/core/domain`)
The domain layer defines the data structures for agent interactions.
- `Trace`: Represents a complete interaction cycle (User Query -> Final Answer).
- `Step`: Represents a single atomic action (Thought, Action Choice, Function Call, etc.).
- `StepType`: Enum for discriminating step types.

### 2. Generation Engine (`src/core/generation`)
This is the heart of the synthetic data creation.
- **TraceGenerator**: The main orchestrator class. It manages the state of the trace and calls appropriate step generators.
- **Step Generators**: Independent functions (`generate_thought`, `generate_function_call`) that use the `Guidance` library to constrain model output to specific formats (e.g., valid JSON for function calls).

### 3. Prompt Engineering (`src/core/prompt`)
- **PromptBuilder**: Responible for constructing the context window for the model. It stitches together system instructions, available functions (schema), and the conversation history.
- **Templates**: Stored in `src/core/prompt/templates`, defining the structure of inputs.

### 4. Dataset Generation (`src/dataset_generation`)
- **Function Factory**: Generates synthetic function definitions and their Python implementations (mock execution).
- **Variants**: Logic to create variations of functions to test model robustness (e.g., parameter shuffling).

### 5. Finetuning (`src/finetuning`)
- **ModelFinetuning**: Handles the training loop using HuggingFace `transformers` and `peft` (LoRA).
- **Data Formatting**: Converts generated `Trace` objects into training samples (prompt + completion) compatible with the model's chat template.

## Data Flow

### Trace Generation
1. **User Query**: A query is selected from the input dataset.
2. **Context Build**: `TraceGenerator` retrieves available functions.
3. **Loop**:
   - `generate_thought()`: Model reflects on the state.
   - `generate_action_choice()`: Model decides to call a function or answer.
   - **If Function Call**:
     - `generate_function_call()`: Model outputs JSON.
     - Mock execution simulates a return value.
     - `generate_function_output()`: Output is added to trace.
   - **If Final Answer**:
     - `generate_final_answer()`: Model responds to user.
     - Loop terminates.
4. **Validation**: The complete trace is validated against schema.

### Data Augmentation
To improve model robustness, three key augmentation strategies are applied during generation:
1. **Function Removal**: Intentionally removing the "correct" function to train the model to say "I cannot answer".
2. **Function Shuffling**: Randomizing the order of function definitions in the prompt.
3. **Prompt Randomization**: Varying the syntax of the prompt (e.g., different delimiters) to prevent overfitting to specific formatting.

## Technology Stack
- **Language**: Python 3.10+
- **LLM Control**: [Microsoft Guidance](https://github.com/guidance-ai/guidance)
- **Training**: PyTorch, HuggingFace Transformers, PEFT (LoRA)
- **CLI**: Click
- **Config**: Pydantic Settings
