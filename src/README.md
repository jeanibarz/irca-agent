# Source Code Documentation

This directory contains the main source code for IRCA-Agent.

## Module Overview

```
src/
├── __init__.py              # Package root with version
├── cli/                     # Command-line interface
├── config/                  # Configuration management
├── core/                    # Core trace generation
├── dataset_generation/      # Function schemas
└── finetuning/              # Model finetuning
```

## Modules

### `cli/` - Command Line Interface

Modern CLI built with [Click](https://click.palletsprojects.com/).

```python
from cli import cli

# Available commands:
# irca generate traces    - Generate agent traces
# irca finetune run       - Run model finetuning
# irca dataset push       - Push to HuggingFace Hub
```

**Files:**
- `__init__.py` - Main CLI entry point
- `commands/generate.py` - Trace generation commands
- `commands/finetune.py` - Model finetuning commands
- `commands/dataset.py` - Dataset management commands

### `config/` - Configuration

Centralized configuration using [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

```python
from config import get_settings

settings = get_settings()
print(settings.models_path)           # /workspace/models
print(settings.huggingface_token)     # From .env
config = settings.get_training_config("mistral")
```

**Files:**
- `settings.py` - Settings class with model presets and training config

### `core/` - Core Functionality

Heart of the trace generation system.

```python
from core import TraceGenerator, Trace, StepType, create_step

# Generate traces
generator = TraceGenerator(model_name_or_path="mistralai/Mistral-7B-v0.1")
trace = generator.generate_single_trace(
    available_functions='[{"name": "get_weather", ...}]',
    user_query="What's the weather in Paris?",
)

# Work with steps
step = create_step(
    step_type=StepType.THOUGHT,
    thought="I should check the weather",
    diff="Thought: I should check the weather",
)
```

**Submodules:**

- `domain/` - Data models
  - `steps.py` - Step types (ThoughtStep, FunctionCallStep, etc.)
  - `trace.py` - Trace model (collection of steps)

- `generation/` - Trace generation
  - `trace_generator.py` - TraceGenerator class
  - `step_generators.py` - Individual step functions
  - `constants.py` - Prompt constants
  - `argilla.py` - Argilla integration

- `prompt/` - Prompt templates
  - `function_calling_oneshot.py` - Main prompt template

**Other files:**
- `prompt_builder.py` - Prompt construction and parsing
- `utils.py` - Utility functions
- `step_factory.py` - Backwards compatibility re-exports
- `trace_generator.py` - Backwards compatibility re-exports

### `dataset_generation/` - Function Schemas

Function definitions used by the agent.

```python
from dataset_generation.functions_factory import FunctionsFactory

functions = FunctionsFactory.load_function_variants(version="v1")
```

**Files:**
- `functions_factory.py` - Factory for loading function variants
- `function_variants/` - Different versions of function schemas

### `finetuning/` - Model Finetuning

LoRA/QLoRA finetuning with PEFT and TRL.

```python
# Run via CLI
# irca finetune run --model-type mistral --epochs 5

# Or programmatically
from config import get_settings
settings = get_settings()
config = settings.get_training_config("mistral")
```

**Files:**
- `model_finetuning.py` - Main finetuning script

## Key Concepts

### Step Types

An agent trace consists of ordered steps:

| Step Type | Description | Key Fields |
|-----------|-------------|------------|
| `InitialPromptStep` | Starting prompt | `diff` |
| `ThoughtStep` | Agent reasoning | `thought`, `diff` |
| `ActionChoiceStep` | Call function or final answer | `action_choice`, `diff` |
| `FunctionCallStep` | Function invocation | `fct_name`, `fct_parameters`, `diff` |
| `FunctionOutputStep` | Function result | `shortuuid`, `function_output`, `diff` |
| `FinalAnswerStep` | Agent's response | `final_answer`, `diff` |

### Trace

A `Trace` is a container for steps with helpful methods:

```python
trace = Trace()
trace.append(step)           # Add a step
trace.to_string()            # Convert to text
len(trace)                   # Number of steps
trace.last_step              # Most recent step
for step in trace: ...       # Iterate
```

### Settings

Configuration is loaded from environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKSPACE_DIR` | Base workspace path | `/workspace` |
| `HUGGINGFACE_TOKEN` | HF Hub access | None |
| `ARGILLA_API_URL` | Argilla server | None |
| `LORA_R` | LoRA rank | 128 |
| `NUM_TRAIN_EPOCHS` | Training epochs | 5 |

See `.env.example` for complete list.

## Import Examples

```python
# Domain models
from core.domain import StepType, Trace, ThoughtStep, create_step

# Generation (requires full dependencies)
from core.generation import TraceGenerator

# Configuration
from config import Settings, get_settings

# Prompt building
from core.prompt_builder import build_full_prompt, format_instruction

# Utilities
from core.utils import shuffle_json_functions, extract_and_remove
```

## Backwards Compatibility

Old imports continue to work:

```python
# These still work (re-exported)
from core.step_factory import create_step_model, StepType
from core.trace_generator import GuidedTraceGenerator
```

Prefer new imports for new code:

```python
# Preferred
from core.domain import create_step, StepType
from core.generation import TraceGenerator
```
