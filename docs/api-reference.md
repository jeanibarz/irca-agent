# API Reference

This document provides API documentation for the main IRCA-Agent modules.

## Configuration

### `config.Settings`

Central configuration class using Pydantic Settings.

```python
from config import Settings, get_settings

# Get singleton instance
settings = get_settings()

# Or create custom instance
custom_settings = Settings(
    workspace_dir=Path("/my/workspace"),
    lora_r=256,
    learning_rate=2e-4,
)
```

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_dir` | `Path` | `/workspace` | Base workspace directory |
| `models_dir` | `Path` | `models` | Models subdirectory |
| `datasets_dir` | `Path` | `datasets` | Datasets subdirectory |
| `finetuned_models_dir` | `Path` | `finetuned_models` | Finetuned models subdirectory |
| `huggingface_token` | `str | None` | `None` | HuggingFace Hub token |
| `argilla_api_url` | `str | None` | `None` | Argilla API URL |
| `argilla_api_key` | `str | None` | `None` | Argilla API key |
| `dataset` | `str` | `JeanIbarz/irca_agent_...` | Default dataset |
| `lora_r` | `int` | `128` | LoRA rank |
| `lora_alpha` | `int` | `64` | LoRA alpha |
| `lora_dropout` | `float` | `0.05` | LoRA dropout |
| `num_train_epochs` | `int` | `5` | Training epochs |
| `learning_rate` | `float` | `1e-3` | Learning rate |
| `max_seq_length` | `int` | `4096` | Max sequence length |

#### Computed Properties

| Property | Type | Description |
|----------|------|-------------|
| `models_path` | `Path` | Full path to models directory |
| `datasets_path` | `Path` | Full path to datasets directory |
| `finetuned_models_path` | `Path` | Full path to finetuned models |

#### Methods

```python
def get_model_config(model_type: str | None = None) -> dict[str, Any]:
    """Get model-specific configuration."""

def get_training_config(model_type: str | None = None) -> dict[str, Any]:
    """Get complete training configuration."""
```

---

## Domain Models

### `core.domain.StepType`

Enum defining the types of steps in an agent trace.

```python
from core.domain import StepType

StepType.THOUGHT         # "thought"
StepType.ACTION_CHOICE   # "action_choice"
StepType.FUNCTION_CALL   # "function_call"
StepType.FUNCTION_OUTPUT # "function_output"
StepType.FINAL_ANSWER    # "final_answer"
StepType.INITIAL_PROMPT  # "initial_prompt"
```

### `core.domain.Trace`

Container for agent trace steps.

```python
from core.domain import Trace, ThoughtStep

trace = Trace()
trace.append(ThoughtStep(thought="...", diff="..."))
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `append(step)` | `None` | Add a step to the trace |
| `to_string()` | `str` | Convert trace to text |
| `__len__()` | `int` | Number of steps |
| `__getitem__(i)` | `Step` | Get step by index |
| `__iter__()` | `Iterator` | Iterate over steps |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `last_step` | `Step | None` | Most recent step |
| `steps` | `list[Step]` | All steps |

### Step Models

All step models inherit from `BaseStep` and have a `diff` field.

#### `ThoughtStep`

```python
step = ThoughtStep(thought="I should help", diff="Thought: I should help")
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `StepType` | Always `StepType.THOUGHT` |
| `thought` | `str` | The thought content |
| `diff` | `str` | Raw text for this step |

#### `ActionChoiceStep`

```python
step = ActionChoiceStep(action_choice="call function", diff="Action choice: call function")
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `StepType` | Always `StepType.ACTION_CHOICE` |
| `action_choice` | `str` | "call function" or "final answer" |
| `diff` | `str` | Raw text for this step |

#### `FunctionCallStep`

```python
step = FunctionCallStep(
    fct_name="get_weather",
    fct_parameters='{"location": "Paris"}',
    diff='Call function: {"name": "get_weather"}, "parameters": {...}'
)
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `StepType` | Always `StepType.FUNCTION_CALL` |
| `fct_name` | `str` | Function name |
| `fct_parameters` | `str` | JSON parameters |
| `diff` | `str` | Raw text for this step |

#### `FunctionOutputStep`

```python
step = FunctionOutputStep(
    shortuuid="abc123",
    function_output='{"temp": 20}',
    diff='Output[abc123]: {"temp": 20}'
)
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `StepType` | Always `StepType.FUNCTION_OUTPUT` |
| `shortuuid` | `str` | Unique output ID |
| `function_output` | `str` | Function result |
| `diff` | `str` | Raw text for this step |

#### `FinalAnswerStep`

```python
step = FinalAnswerStep(
    final_answer="The weather is sunny.",
    diff="\n\n### FINAL ANSWER\nThe weather is sunny."
)
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `StepType` | Always `StepType.FINAL_ANSWER` |
| `final_answer` | `str` | The answer content |
| `diff` | `str` | Raw text for this step |

### `create_step`

Factory function to create step models.

```python
from core.domain import create_step, StepType

step = create_step(
    step_type=StepType.THOUGHT,
    thought="Test thought",
    diff="Thought: Test thought",
)
```

---

## Generation

### `core.generation.TraceGenerator`

Main class for generating agent traces.

```python
from core.generation import TraceGenerator

generator = TraceGenerator(model_name_or_path="mistralai/Mistral-7B-v0.1")
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str | None` | `None` | Path or HuggingFace ID |
| `model` | `LanguageModel | None` | `None` | Pre-initialized model |
| `torch_dtype` | `dtype` | `bfloat16` | Model dtype |
| `device_map` | `dict | None` | `{"": 0}` | Device mapping |

#### Methods

```python
def generate_single_trace(
    available_functions: str,
    user_query: str,
    max_steps: int = 10,
) -> Trace:
    """Generate a single complete trace."""

def generate_trace_missing_function(
    available_functions: str,
    user_query: str,
    base_trace: Trace | None = None,
    next_thought_prefix: str = "",
) -> Trace:
    """Generate a trace where required function is missing."""

def generate_traces_with_augmentation(
    available_functions: str,
    user_query: str,
    max_steps: int = 5,
) -> list[Trace]:
    """Generate multiple traces with data augmentation."""
```

### Step Generators

Individual functions for generating each step type.

```python
from core.generation import (
    generate_thought,
    generate_action_choice,
    generate_function_call,
    generate_function_output,
    generate_final_answer,
)

# Each function has signature like:
def generate_thought(
    lm: Program,
    trace: Trace,
    prefix: str = "",
    suffix: str = "",
    temperature: float = 0.25,
    max_tokens: int = 500,
) -> Program:
    """Generate a thought step."""
```

---

## Utilities

### `core.utils`

```python
from core.utils import (
    extract_and_remove,
    shuffle_json_functions,
    format_generate_user_query,
    print_trainable_parameters,
    get_trainable_param_count,
)
```

#### `extract_and_remove`

Extract a section from text between markers.

```python
extracted, remaining = extract_and_remove(
    start_marker="### START",
    end_marker="### END",
    full_prompt=text,
    include_start_marker=False,
    include_end_marker=False,
)
```

#### `shuffle_json_functions`

Shuffle function order for data augmentation.

```python
shuffled = shuffle_json_functions('[{"name": "a"}, {"name": "b"}]')
# Returns JSON with shuffled order
```

---

## Prompt Building

### `core.prompt_builder`

```python
from core.prompt_builder import (
    build_full_prompt,
    parse_corrected_agent_trace,
    format_instruction,
    InstructionFormatter,
)
```

#### `build_full_prompt`

Build a complete prompt from components.

```python
prompt = build_full_prompt({
    "system_instructions": "You are an AI assistant.",
    "example": "...",
    "available_functions_json": "[...]",
    "user_query": "Help me",
    "assistant_completion": "Thought: ...",
})
```

#### `format_instruction`

Format a training sample with optional augmentation.

```python
formatted = format_instruction(
    sample={"corrected_agent_trace": [{"value": "..."}]},
    random_augmentation=True,
)
```
