# Scripts

Utility scripts for dataset generation, model training, and data management.

## Dataset Generation

### `main_generator.py`
Main trace generation script. Generates agent traces from user queries.

```bash
python scripts/main_generator.py
```

### `main_generator_user_query.py`
Generates user queries based on available functions.

```bash
python scripts/main_generator_user_query.py
```

### `create_irca_agent_dataset.py`
Creates a new Argilla dataset with the IRCA Agent schema.

```bash
python scripts/create_irca_agent_dataset.py
```

## Data Management

### `combine_datasets.py`
Combines multiple Argilla datasets into a single accumulated dataset.

```bash
python scripts/combine_datasets.py
```

### `push_to_huggingface.py`
Pushes a dataset from Argilla to HuggingFace Hub.

```bash
python scripts/push_to_huggingface.py
```

### `copy_from_hf_to_argilla.py`
Copies a dataset from HuggingFace Hub to Argilla.

```bash
python scripts/copy_from_hf_to_argilla.py
```

## Utilities

### `main.py`
Legacy script for Argilla dataset creation and completions. 
*Note: Consider using `create_irca_agent_dataset.py` instead.*

### `download_model.py`
Downloads models from HuggingFace Hub.

```bash
python scripts/download_model.py
```

### `simple_llamacpp_python.py`
Simple example of using llama-cpp-python for local inference.

```bash
python scripts/simple_llamacpp_python.py
```

## Configuration

Scripts use environment variables from `.env` file. Required variables:
- `HUGGINGFACE_TOKEN` - For HuggingFace Hub access
- `ARGILLA_API_URL` - Argilla server URL
- `ARGILLA_API_KEY` - Argilla API key

See `.env.example` for all available options.