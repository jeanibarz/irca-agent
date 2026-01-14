# IRCA-Agent

**Dataset Generation Toolkit for Finetuning Function-Calling Agents**

IRCA-Agent (Iterative Resolution Cycle Agent) is a toolkit for generating high-quality datasets to finetune language models for function-calling / tool-use capabilities. It implements data augmentation techniques to improve model robustness and truthfulness.

## 🌟 Features

- **Guided Trace Generation**: Generate complete agent traces with thoughts, function calls, and final answers
- **Data Augmentation**: 
  - Function removal to train models to acknowledge limitations
  - Prompt style randomization to improve generalization
  - Function order shuffling to reduce positional bias
- **Multiple Model Support**: Works with various base models (Mistral, TinyLlama, Qwen, etc.)
- **HuggingFace Integration**: Easy dataset upload/download from HuggingFace Hub
- **Argilla Integration**: Optional integration for human-in-the-loop data curation

## 📁 Project Structure

```
irca-agent/
├── src/
│   ├── core/                      # Core trace generation logic
│   │   ├── trace_generator.py     # Main trace generation
│   │   ├── prompt_builder.py      # Prompt construction & formatting
│   │   ├── step_factory.py        # Step types (Pydantic models)
│   │   └── prompt/                # Prompt templates
│   ├── dataset_generation/        # Function definitions & variants
│   ├── finetuning/                # Model finetuning with PEFT/LoRA
│   └── defaults/                  # Default configurations
├── scripts/                       # Utility scripts
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
# Edit .env with your tokens
```

### Using DevContainer (Recommended)

If you have Docker and VS Code with the DevContainers extension:

1. Open the project in VS Code
2. Click "Reopen in Container" when prompted
3. The environment will be set up automatically

### Generate a Dataset

```bash
# Activate the virtual environment
poetry shell

# Run the trace generator
python scripts/main_generator.py
```

### Finetune a Model

```bash
# Using default configuration (Mistral-7B)
python -m src.finetuning.model_finetuning

# Specify model type
python -m src.finetuning.model_finetuning --model_type mistral
```

## 📊 Datasets

The project uses an iterative dataset generation approach:

| Dataset | Description |
|---------|-------------|
| `irca_agent_dataset_v5-5acc` | **Best/latest** accumulated dataset |
| `irca_user_query_dataset_v*` | User query datasets for trace generation |

Datasets are available on [HuggingFace Hub](https://huggingface.co/JeanIbarz).

## 🔧 Configuration

Configuration is managed through:
- `src/defaults/v1/training_args.py` - Training hyperparameters
- `.env` - Environment variables (API keys, paths)
- `pyproject.toml` - Dependencies and tool settings

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 128 | LoRA rank |
| `lora_alpha` | 64 | LoRA alpha |
| `num_train_epochs` | 5 | Training epochs |
| `learning_rate` | 1e-3 | Learning rate |

## 🧪 Development

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
pytest

# Run linting
ruff check src/

# Run type checking
mypy src/

# Format code
ruff format src/
```

## 📖 How It Works

### Iterative Resolution Cycle

The agent follows a structured approach to answer user queries:

1. **Thought**: Reflect on necessary steps and available functions
2. **Action Choice**: Decide to `call function` or provide `final answer`
3. **Function Call**: Execute function with parameters, wait for output
4. **Function Output**: Receive synthetic or real function response
5. Repeat steps 1-4 until ready to answer
6. **Final Answer**: Provide concise response to user

### Data Augmentation Techniques

1. **Function Removal**: Remove the function the model would use, forcing it to acknowledge inability to complete the task
2. **Prompt Randomization**: Vary formatting (newlines, markers) to reduce overfitting to style
3. **Function Shuffling**: Randomize function order in the prompt

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for transformers and TRL
- [Microsoft Guidance](https://github.com/guidance-ai/guidance) for constrained generation
- [Argilla](https://argilla.io/) for data curation tools
