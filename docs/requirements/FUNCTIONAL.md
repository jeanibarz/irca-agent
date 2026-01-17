# Functional Requirements

## Core Generation
- **FR-GEN-01 (Trace Generation)**: The system shall generate synthetic agent traces starting from a user query, simulating a "Thought -> Action Choice -> Function Call -> Observation -> Final Answer" loop.
- **FR-GEN-02 (Guidance-Constrained)**: The system shall use [Microsoft Guidance](https://github.com/guidance-ai/guidance) to enforce strict schema compliance for all model outputs (e.g., JSON validation).
- **FR-GEN-03 (Function Augmentation)**: The system shall support "Function Removal" (negative examples) where the required tool is intentionally omitted to train the model to acknowledge limitations.
- **FR-GEN-04 (Function Shuffling)**: The system shall support randomizing the order of available functions in the prompt to prevent positional bias.
- **FR-GEN-05 (Prompt Randomization)**: The system shall support varying prompt syntax (delimiters, spacing) to improve model robustness.

## CLI & Usability
- **FR-CLI-01 (Command Interface)**: The system shall provide a Command Line Interface (CLI) for generating traces, finetuning models, and managing datasets.
- **FR-CLI-02 (Verbose Mode)**: The CLI shall support a verbose mode (`-v`) to display generation steps in real-time for debugging.
- **FR-CLI-03 (Resume Capability)**: The generation command shall allow starting from a specific index (`--start`) to resume interrupted jobs.

## Finetuning
- **FR-FT-01 (Q-LoRA Training)**: The system shall support finetuning models using Quantized Low-Rank Adaptation (Q-LoRA) to minimize resource usage.
- **FR-FT-02 (Model Support)**: The system shall explicitly support Mistral-7B, TinyLlama-1.1B, and Qwen models via configuration presets.
- **FR-FT-03 (Hyperparameters)**: The user shall be able to configure training parameters (epochs, learning rate, LoRA rank) via environment variables or settings.
- **FR-FT-W01 (W&B Explicit Init)**: Finetuning shall explicitly initialize W&B before training to ensure reliable metric streaming.
- **FR-FT-W02 (W&B Run Naming)**: W&B run names shall include model type, dataset name, and timestamp for easy identification (e.g., `qwen3-4b-dataset-20260117-143052`).
- **FR-FT-W03 (W&B Config Logging)**: Training configuration (model_type, backend, epochs, learning_rate, lora_r, batch_size, etc.) shall be logged to W&B config.
- **FR-FT-W04 (W&B Backend Support)**: W&B integration shall work with both Unsloth and TRL backends.
- **FR-FT-W05 (W&B Finalization)**: W&B shall be properly finalized after training completes, logging final metrics (train_loss, train_runtime, samples_per_second).
- **FR-FT-W06 (W&B Optional)**: W&B integration shall be optional via `--no-wandb` flag, allowing training without tracking.

## Dataset Management
- **FR-DATA-01 (HuggingFace Integration)**: The system shall allow pushing generated datasets to HuggingFace Hub and pulling them for training.
- **FR-DATA-02 (Argilla Integration)**: The system shall support uploading datasets to Argilla for human-in-the-loop review (optional).
- **FR-DATA-03 (On-the-fly Multilingual Diversification)**: The system shall support diversifying the training dataset by translating text fields (`user_query`, `final_answer`) to target languages while strictly preserving English for reasoning traces (`<thought>`, tool calls). This can be performed either "Offline" (pre-calculated before training to save GPU memory) or "Online" (dynamically during training).
- **FR-DATA-04 (Stochastic Configuration)**: The system shall support a configurable total probability (ratio) for diversification, with parameters for translation models and random language selection applied per individual sample retrieval.
- **FR-DATA-05 (Feature Flipping)**: The diversification logic shall be toggleable via a master flag in the configuration/CLI.
- **FR-DATA-06 (Parallel Execution)**: The diversification process shall be implemented using parallel processing (CPU workers) to ensure that translation does not become a bottleneck. When using Offline diversification, the process shall complete before model loading to optimize GPU VRAM availability and avoid multiprocessing conflicts with CUDA.
- **FR-DATA-07 (Model-Aware Diversification)**: The system shall support model-specific language configuration, automatically filtering the requested diversification languages based on the known linguistic compatibility of the target base model (e.g., filtering for Mistral while allowing broader sets for Qwen).
- **FR-DATA-08 (Separated Augmentation Pipeline)**: The system shall provide a dedicated CLI command (`irca dataset augment`) to augment datasets independently from finetuning. The augmented dataset shall be saved to disk with a `text` column ready for training.
- **FR-DATA-09 (Markdown Link Preservation)**: During multilingual translation, the system shall preserve markdown link URLs intact (e.g., `Output[uuid]` references) while allowing link text to be translated. This ensures reference integrity in final answers.
- **FR-DATA-10 (Dataset Inspection)**: The system shall provide a CLI command (`irca dataset inspect`) to display dataset structure, column names, and sample entries for verification before finetuning.
- **FR-DATA-11 (Dataset Size Multiplication)**: The augmentation system shall support multiplying the dataset size by a configurable factor (e.g., `multiply: 3` produces 3x samples from the input dataset, with the original sample always preserved as variant 0).
- **FR-DATA-12 (Pipeline Configuration)**: The augmentation system shall accept a JSON configuration file defining an ordered pipeline of transformation steps. Each step shall have a `type`, `probability` (0.0-1.0), and optional `params`. The configuration shall include `input`, `output`, `multiply`, `seed`, and `pipeline` fields.
- **FR-DATA-13 (Feature Combination)**: The augmentation system shall support applying multiple transformation features to the same variant by walking through the pipeline sequentially and independently rolling against each step's probability.
- **FR-DATA-14 (Post-Generation Deduplication)**: The augmentation system shall support optional deduplication (`deduplicate: true`) to remove variants with identical text content within each original sample's variants after generation.
- **FR-DATA-15 (Augmentation Metadata)**: The augmentation system shall generate a metadata sidecar file (`_augmentation_metadata.json`) alongside the output dataset, containing: git commit hash, full configuration used, input/output paths and sizes, augmentation statistics, and environment details (Python version, package versions).
- **FR-DATA-16 (W&B Augmentation Tracking)**: The augmentation system shall optionally log augmented datasets as W&B Artifacts with full metadata when `--track-wandb` flag is provided, enabling experiment tracking and lineage to finetuning runs.
- **FR-DATA-17 (Config-Driven Augmentation CLI)**: The augmentation CLI shall require a JSON configuration file (`--config`), with optional CLI overrides for `--input` and `--output` paths to enable config reuse across datasets.
- **FR-DATA-18 (Perplexity Diversity Metrics)**: The system shall compute perplexity-based diversity metrics for datasets using a specified model, including mean, std, min, max, median, and percentile statistics.
- **FR-DATA-19 (Diversity CLI)**: The system shall provide an `irca dataset diversity` command supporting: single dataset analysis, original vs augmented comparison, model specification for perplexity metrics, and quick mode for model-free analysis.
- **FR-DATA-20 (Diversity in Augmentation Metadata)**: The augmentation system shall optionally record diversity metrics (perplexity, Vendi Score, n-gram) in the metadata file when a diversity model is specified via `--diversity-model`.
- **FR-DATA-21 (Diversity Comparison)**: The diversity CLI shall support comparing two datasets (original vs augmented) and report the delta in all metrics, including percentage change.
- **FR-DATA-22 (Augmentation Effectiveness Evaluation)**: The system shall support evaluating augmentation effectiveness by comparing test set perplexity after finetuning on original vs augmented datasets, providing a definitive measure of whether augmentation improved generalization.
- **FR-DATA-23 (Quick Diversity Mode)**: The diversity CLI shall support a `--quick` mode that computes only model-free metrics (Vendi Score, Distinct-n) without requiring model loading.
- **FR-DATA-24 (Dataset Format Command)**: The system shall provide an `irca dataset format` command to convert raw structured datasets (with columns like `corrected_agent_trace`) to training-ready datasets with a `text` column.
- **FR-DATA-25 (Staged Augmentation)**: The format command shall process augmentations in three stages: structural (before formatting, e.g., translation), formatting (during formatting, e.g., function shuffling), and presentation (after formatting, e.g., newline variation).
- **FR-DATA-26 (Format Baseline)**: The format command shall support a `--no-augment` flag to generate a formatted baseline without any augmentations, enabling diversity comparison between original and augmented datasets.
- **FR-DATA-27 (Format Config)**: The format command shall accept a JSON configuration file defining staged augmentation pipelines with `structural`, `formatting`, and `presentation` sections, each containing steps with `type`, `probability`, and optional `params`.
- **FR-DATA-28 (Diversity Evaluation Workflow)**: The system shall enable comparing diversity between formatted baseline and augmented datasets by: (1) formatting raw data without augmentation, (2) formatting with augmentation config, (3) running diversity comparison on both formatted datasets.

## Diversity Reporting
- **FR-DIV-10 (HTML Report Generation)**: The system shall generate self-contained HTML reports from diversity metrics, embedding all CSS and JavaScript inline for offline viewing and sharing.
- **FR-DIV-11 (Report Visualizations)**: The HTML report shall include interactive Chart.js visualizations for: distinct-n comparison bar charts, perplexity statistics comparison, and augmentation contribution horizontal bar charts.
- **FR-DIV-12 (Report CLI Flag)**: The diversity CLI command shall support a `--report` / `-r` flag to specify an output path for generating an HTML report alongside the standard CLI output.
- **FR-DIV-13 (Report Metadata)**: The HTML report shall include metadata showing: generation timestamp, dataset paths, model used (if any), git commit (if available), and IRCA version.
- **FR-DIV-14 (Report Comparison Mode)**: The HTML report shall support comparison mode displaying original vs augmented metrics side-by-side with delta calculations and color-coded indicators (green for positive change, red for negative).
- **FR-DIV-15 (Report Single Dataset Mode)**: The HTML report shall support single dataset mode displaying metrics for a single dataset analysis without comparison.

## Model Evaluation
- **FR-EVAL-01 (Completion-Only Perplexity)**: The system shall support computing perplexity on completion tokens only (excluding prompt tokens), which is the recommended metric for evaluating SFT model quality.
- **FR-EVAL-02 (Baseline Model Comparison)**: The system shall support comparing a finetuned model against a baseline model on the same dataset, computing improvement percentages for perplexity metrics.
- **FR-EVAL-03 (Generalization Gap)**: The system shall support computing the generalization gap by comparing perplexity on training vs held-out test sets, which indicates potential overfitting.
- **FR-EVAL-04 (Evaluation Mode Flag)**: The diversity CLI shall support an `--eval-mode` flag with options "full" (entire sample) and "completion" (response tokens only).
- **FR-EVAL-05 (Robustness Report)**: The system shall generate evaluation reports showing model comparison results including perplexity deltas, generalization gaps, and interpretation summaries.

## Playground & Evaluation
- **FR-PLAY-01 (Interactive Chat)**: The system shall provide a web interface (`ui/`) to interact with models, supporting parameter adjustment (temp, top_p) and visualization of agent traces (thoughts, tool calls).
- **FR-PLAY-02 (Model Management)**: The system shall allow users to list, load, and eject models/adapters from GPU memory via the UI to manage resources.
- **FR-PLAY-03 (Synthetic Red-Teaming)**: The system shall provide "Magic" buttons to generate synthetic user queries (both feasible and infeasible) to stress-test model instruction following and hallucination resistance.
- **FR-PLAY-04 (Robustness)**: The backend shall handle concurrent loading requests gracefully and report current model status to prevents state inconsistencies.
- **FR-PLAY-05 (Conversation History)**: The system shall persist conversation history to disk (`data/sessions/`) and allow users to create new chats or resume previous ones via the sidebar.
- **FR-PLAY-06 (Output Sanitization)**: The UI shall automatically strip internal control tokens (e.g., `<|wait|>`) from the assistant's response before displaying it to the user.
- **FR-PLAY-08 (Reference Highlighting)**: The Final Answer display shall support parsing markdown-style references (e.g., `[label](Output[ID])`) which, on hover, highlight the corresponding Tool Output block in the trace.
- **FR-PLAY-09 (Conversation Deletion)**: The system shall allow users to delete conversations from the history list, requiring a confirmation dialog to prevent accidental deletion.

## Security (Added from FM Audit 2025-01-17)
- **FR-SEC-01 (Safe Expression Evaluation)**: The system shall use AST-based safe evaluation for mathematical expressions, rejecting any code execution attempts including imports, function calls, attribute access, and lambda expressions.
- **FR-SEC-02 (Input Validation)**: The API shall validate all input parameters with bounded ranges: max_tokens (1-32768), temperature (0.0-2.0), top_p (0.0-1.0), and message content length (max 100000 chars).
- **FR-SEC-03 (CORS Configuration)**: The server shall use environment-configurable CORS origins (via `IRCA_CORS_ORIGINS`) instead of allowing all origins.
- **FR-SEC-04 (Generation Timeout)**: The system shall enforce a configurable timeout (default 120 seconds, range 10-600) on model generation calls to prevent infinite hangs from stuck models.
- **FR-SEC-05 (Conversation ID Validation)**: The system shall validate conversation IDs as valid UUIDs to prevent path traversal attacks (FM-34).
- **FR-SEC-06 (Adapter ID Validation)**: The system shall validate adapter IDs to prevent path traversal attacks, only allowing loading from the finetuned models directory or valid HuggingFace Hub IDs (FM-36).
- **FR-SEC-07 (Model ID Validation)**: The system shall validate base model IDs against a whitelist of known presets or valid HuggingFace Hub ID format to prevent loading arbitrary models (FM-42).

---

**Navigation:**
- [Non-Functional Requirements](NON_FUNCTIONAL.md)
- [Traceability Matrix](../testing/TRACEABILITY_MATRIX.md)
- [Test Strategy](../testing/TEST_STRATEGY.md)
