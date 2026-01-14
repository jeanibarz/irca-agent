# IRCA-Agent

**Dataset Generation Toolkit for Finetuning Function-Calling Agents**

[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-Managed-blue.svg)](https://python-poetry.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

IRCA-Agent (Iterative Resolution Cycle Agent) is a toolkit for generating high-quality datasets to finetune language models for function-calling / tool-use capabilities. It implements data augmentation techniques to improve model robustness and truthfulness.

## ✨ Features

- **Guided Trace Generation**: Generate complete agent traces with thoughts, function calls, and final answers using [Microsoft Guidance](https://github.com/guidance-ai/guidance)
- **Data Augmentation**:
  - Function removal to train models to acknowledge limitations
  - Prompt style randomization to improve generalization
  - Function order shuffling to reduce positional bias
- **Multiple Model Support**: Works with Mistral, TinyLlama, Qwen, and more
- **Modern CLI**: Clean command-line interface with Click
- **HuggingFace Integration**: Easy dataset upload/download from HuggingFace Hub
- **Argilla Integration**: Optional human-in-the-loop data curation

## 📁 Project Structure

```
irca-agent/
├── src/
│   ├── cli/                       # Command-line interface
│   │   └── commands/              # CLI command groups
│   ├── config/                    # Configuration (Pydantic Settings)
│   │   └── settings.py            # Centralized settings
│   ├── core/
│   │   ├── domain/                # Data models (Step, Trace)
│   │   ├── generation/            # Trace generation logic
│   │   ├── prompt/                # Prompt templates
│   │   ├── prompt_builder.py      # Prompt construction
│   │   └── utils.py               # Utility functions
│   ├── dataset_generation/        # Function schemas & variants
│   └── finetuning/                # Model finetuning with PEFT/LoRA
├── scripts/                       # Legacy scripts (deprecated)
├── tests/                         # Unit & integration tests
├── datasets/                      # Generated datasets (gitignored)
└── models/                        # Downloaded/finetuned models (gitignored)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10-3.12
- CUDA-compatible GPU (recommended for finetuning)
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/JeanIbarz/irca-agent.git
cd irca-agent

# Install dependencies
poetry install

# For local inference with llama.cpp
poetry install --extras local-inference

# Copy and configure environment
cp .env.example .env
# Edit .env with your tokens (HuggingFace, Argilla, etc.)
```

### Using DevContainer (Recommended)

If you have Docker and VS Code with the DevContainers extension:

1. Open the project in VS Code
2. Click "Reopen in Container" when prompted
3. The environment will be set up automatically with all dependencies

## 💻 CLI Usage

IRCA-Agent provides a modern CLI for all operations:

```bash
# Activate environment
poetry shell

# Show available commands
irca --help

# Show version
irca --version
```

### Generate Traces

```bash
# Generate traces from a user query dataset
irca generate traces

# Generate with specific options
irca generate traces --model ./path/to/model --limit 10 --start 5

# Verbose mode for debugging
irca -v generate traces
```

### Finetune Models

```bash
# Finetune with default settings (Mistral)
irca finetune run

# Finetune TinyLlama for 3 epochs
irca finetune run --model-type tinyllama --epochs 3

# Show training configuration
irca finetune info --model-type mistral
```

### Manage Datasets

```bash
# Push dataset to HuggingFace Hub
irca dataset push --source irca_agent_v5 --target username/my-dataset

# Pull dataset from HuggingFace
irca dataset pull --source JeanIbarz/irca_agent_dataset_v5-5acc

# List Argilla datasets
irca dataset list --workspace irca_agent
```

## 📊 Datasets

The project uses an iterative dataset generation approach:

| Dataset | Description |
|---------|-------------|
| `irca_agent_dataset_v5-5acc` | **Best/latest** accumulated dataset |
| `irca_user_query_dataset_v*` | User query datasets for trace generation |

Datasets are available on [HuggingFace Hub](https://huggingface.co/JeanIbarz).

## ⚙️ Configuration

Configuration is managed through environment variables and `src/config/settings.py`:

### Environment Variables (.env)

```bash
# HuggingFace Hub access
HUGGINGFACE_TOKEN=hf_...

# Argilla (optional)
ARGILLA_API_URL=http://localhost:6900
ARGILLA_API_KEY=...

# Weights & Biases (optional)
WANDB_API_KEY=...

# Path overrides (optional)
WORKSPACE_DIR=/workspace
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 128 | LoRA rank |
| `lora_alpha` | 64 | LoRA alpha |
| `lora_dropout` | 0.05 | LoRA dropout |
| `num_train_epochs` | 5 | Training epochs |
| `learning_rate` | 1e-3 | Learning rate |
| `max_seq_length` | 4096 | Max sequence length |

### Supported Models

| Model Type | Base Model |
|------------|-----------|
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.2` |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `qwen-4b` | `Qwen/Qwen2.5-3B-Instruct` |
| `qwen-14b` | `Qwen/Qwen2-14B-Instruct` |

## 🧪 Development

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run linting
ruff check src/

# Run type checking
mypy src/

# Format code
ruff format src/

# Run all checks (pre-commit)
pre-commit run --all-files
```

## 📖 How It Works

### Iterative Resolution Cycle

The agent follows a structured approach to answer user queries:

```
User Query → Thought → Action Choice → [Function Call → Output]* → Final Answer
```

1. **Thought**: Reflect on necessary steps and available functions
2. **Action Choice**: Decide to `call function` or provide `final answer`
3. **Function Call**: Execute function with parameters
4. **Function Output**: Receive synthetic or real function response
5. Repeat steps 1-4 until ready to answer
6. **Final Answer**: Provide concise response to user

### Data Augmentation Techniques

| Technique | Purpose |
|-----------|---------|
| **Function Removal** | Remove the function the model would use, training it to acknowledge inability |
| **Prompt Randomization** | Vary formatting (newlines, markers) to reduce style overfitting |
| **Function Shuffling** | Randomize function order to reduce positional bias |

### Architecture

```
TraceGenerator (orchestration)
    │
    ├── Step Generators (individual steps)
    │   ├── generate_thought()
    │   ├── generate_action_choice()
    │   ├── generate_function_call()
    │   ├── generate_function_output()
    │   └── generate_final_answer()
    │
    ├── Domain Models
    │   ├── Step types (ThoughtStep, FunctionCallStep, etc.)
    │   └── Trace (collection of steps)
    │
    └── Guidance Library (constrained generation)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`pytest && ruff check src/`)
5. Commit with conventional commits (`feat: add amazing feature`)
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for Transformers, TRL, and the Hub
- [Microsoft Guidance](https://github.com/guidance-ai/guidance) for constrained generation
- [Argilla](https://argilla.io/) for data curation tools
- [Pydantic](https://docs.pydantic.dev/) for robust configuration management
