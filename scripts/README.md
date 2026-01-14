# Scripts

> ⚠️ **DEPRECATED**: These scripts are being migrated to the CLI.
> Use `irca` command instead. See [CLI Usage](#cli-usage) below.

## CLI Usage

The recommended way to use IRCA-Agent is through the CLI:

```bash
# Install with poetry
poetry install

# Activate environment
poetry shell

# Show available commands
irca --help

# Generate traces
irca generate traces --help

# Finetune a model
irca finetune run --model-type mistral

# Manage datasets
irca dataset --help
```

## Legacy Scripts

The following scripts are being phased out. Use the CLI equivalents when possible.

### Dataset Generation

| Script | CLI Equivalent | Status |
|--------|---------------|--------|
| `main_generator.py` | `irca generate traces` | ⚠️ Migrated |
| `main_generator_user_query.py` | `irca generate user-queries` | ⏳ Pending |
| `create_irca_agent_dataset.py` | `irca dataset create` | ⏳ Pending |

### Data Management

| Script | CLI Equivalent | Status |
|--------|---------------|--------|
| `combine_datasets.py` | `irca dataset combine` | ⚠️ Migrated |
| `push_to_huggingface.py` | `irca dataset push` | ⚠️ Migrated |
| `copy_from_hf_to_argilla.py` | `irca dataset pull` | ⚠️ Migrated |

### Utilities

| Script | CLI Equivalent | Status |
|--------|---------------|--------|
| `download_model.py` | Manual download | - |
| `simple_llamacpp_python.py` | Example only | - |
| `main.py` | Deprecated | ❌ Remove |

## Running Legacy Scripts

If you need to run legacy scripts directly:

```bash
# From project root with PYTHONPATH set
PYTHONPATH=src python scripts/main_generator.py

# Or using poetry run
poetry run python scripts/main_generator.py
```

## Configuration

All scripts and CLI commands use environment variables from `.env` file:

- `HUGGINGFACE_TOKEN` - For HuggingFace Hub access
- `ARGILLA_API_URL` - Argilla server URL
- `ARGILLA_API_KEY` - Argilla API key

See `.env.example` for all available options.
