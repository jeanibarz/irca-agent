# Requirements Definition

## Functional Requirements

### Core Generation
- **FR-GEN-01 (Trace Generation)**: The system shall generate synthetic agent traces starting from a user query, simulating a "Thought -> Action Choice -> Function Call -> Observation -> Final Answer" loop.
- **FR-GEN-02 (Guidance-Constrained)**: The system shall use [Microsoft Guidance](https://github.com/guidance-ai/guidance) to enforce strict schema compliance for all model outputs (e.g., JSON validation).
- **FR-GEN-03 (Function Augmentation)**: The system shall support "Function Removal" (negative examples) where the required tool is intentionally omitted to train the model to acknowledge limitations.
- **FR-GEN-04 (Function Shuffling)**: The system shall support randomizing the order of available functions in the prompt to prevent positional bias.
- **FR-GEN-05 (Prompt Randomization)**: The system shall support varying prompt syntax (delimiters, spacing) to improve model robustness.

### CLI & Usability
- **FR-CLI-01 (Command Interface)**: The system shall provide a Command Line Interface (CLI) for generating traces, finetuning models, and managing datasets.
- **FR-CLI-02 (Verbose Mode)**: The CLI shall support a verbose mode (`-v`) to display generation steps in real-time for debugging.
- **FR-CLI-03 (Resume Capability)**: The generation command shall allow starting from a specific index (`--start`) to resume interrupted jobs.

### Finetuning
- **FR-FT-01 (Q-LoRA Training)**: The system shall support finetuning models using Quantized Low-Rank Adaptation (Q-LoRA) to minimize resource usage.
- **FR-FT-02 (Model Support)**: The system shall explicitly support Mistral-7B, TinyLlama-1.1B, and Qwen models via configuration presets.
- **FR-FT-03 (Hyperparameters)**: The user shall be able to configure training parameters (epochs, learning rate, LoRA rank) via environment variables or settings.

### Dataset Management
- **FR-DATA-01 (HuggingFace Integration)**: The system shall allow pushing generated datasets to HuggingFace Hub and pulling them for training.
- **FR-DATA-02 (Argilla Integration)**: The system shall support uploading datasets to Argilla for human-in-the-loop review (optional).

## Non-Functional Requirements

### Performance & Scalability
- **NFR-PERF-01 (Local Execution)**: The system shall be capable of running generation and training on consumer-grade GPUs (e.g., RTX 3090/4090) or local CPU (slow but functional).
- **NFR-PERF-02 (Efficiency)**: Trace generation should use token fast-forwarding (via Guidance) to minimize generation latency.

### Maintainability & Code Quality
- **NFR-CODE-01 (Type Safety)**: All Python code shall use type hints and pass `mypy` strict checking.
- **NFR-CODE-02 (Configuration)**: All configuration shall be centralized in a Pydantic Settings class, loadable from `.env` files.
- **NFR-CODE-03 (Modularity)**: The core generation logic must be decoupled from specific model implementations where possible.

### Usability
- **NFR-USE-01 (Documentation)**: The system shall provide clear architecture diagrams and "Getting Started" guides (fulfilled by `docs/`).
