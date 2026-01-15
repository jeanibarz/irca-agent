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
- **FR-DATA-03 (Online Multilingual Diversification)**: The system shall support diversifying the training dataset on-the-fly by translating text fields (`user_query`, `final_answer`) to random target languages during the training process, while strictly preserving English for reasoning traces (`<thought>`, tool calls). This ensures that every epoch presents uniquely diversified samples to the model.
- **FR-DATA-04 (Stochastic Configuration)**: The system shall support a configurable total probability (ratio) for diversification, with parameters for translation models and random language selection applied per individual sample retrieval.
- **FR-DATA-05 (Feature Flipping)**: The diversification logic shall be toggleable via a master flag in the configuration/CLI.
- **FR-DATA-06 (Parallel Execution)**: The diversification process shall be implemented using asynchronous/parallel data loading (e.g., Iterable Datasets) to ensure that translation does not become a bottleneck for the training loop.

### Playground & Evaluation
- **FR-PLAY-01 (Interactive Chat)**: The system shall provide a web interface (`ui/`) to interact with models, supporting parameter adjustment (temp, top_p) and visualization of agent traces (thoughts, tool calls).
- **FR-PLAY-02 (Model Management)**: The system shall allow users to list, load, and eject models/adapters from GPU memory via the UI to manage resources.
- **FR-PLAY-03 (Synthetic Red-Teaming)**: The system shall provide "Magic" buttons to generate synthetic user queries (both feasible and infeasible) to stress-test model instruction following and hallucination resistance.
- **FR-PLAY-04 (Robustness)**: The backend shall handle concurrent loading requests gracefully and report current model status to prevents state inconsistencies.
- **FR-PLAY-05 (Conversation History)**: The system shall persist conversation history to disk (`data/sessions/`) and allow users to create new chats or resume previous ones via the sidebar.
- **FR-PLAY-06 (Output Sanitization)**: The UI shall automatically strip internal control tokens (e.g., `<|wait|>`) from the assistant's response before displaying it to the user.
- **FR-PLAY-08 (Reference Highlighting)**: The Final Answer display shall support parsing markdown-style references (e.g., `[label](Output[ID])`) which, on hover, highlight the corresponding Tool Output block in the trace.
- **FR-PLAY-09 (Conversation Deletion)**: The system shall allow users to delete conversations from the history list, requiring a confirmation dialog to prevent accidental deletion.

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
- **NFR-USE-02 (Feedback)**: Long-running operations, especially model loading, shall provide real-time, granular progress feedback to the user (e.g., percentage loaded) to prevent perception of freezing.
