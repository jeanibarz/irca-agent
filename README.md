# IRCA-Agent

## Overview
IRCA-Agent is a comprehensive project aimed at generating datasets and fine-tuning a Large Language Model (LLM) from scratch. The project leverages the Guidance-AI library to constrain the LLM to generate specific outputs, enhancing the quality and relevance of the generated content. 

## Key Components
- **Guidance-AI Library**: Utilized for directing the LLM's output towards specific objectives or constraints.
- **Argilla**: An open-source framework employed for labeling datasets efficiently and accurately.
- **Transformers/PEFT/QLORA**: A set of tools and frameworks for the fine-tuning process of the LLM.
- **FastAPI and OAuth2**: These technologies are integrated to deploy a public API, enabling web-based inferences and interactions with the LLM.

## Project Structure
The `irca-agent` repository is organized as follows:

- `/datasets`: Contains scripts and tools for dataset generation and preprocessing.
- `/model_training`: Includes the necessary scripts and configuration files for LLM training and fine-tuning.
- `/api`: Houses the FastAPI implementation for
