# Traceability Matrix

This matrix maps Functional Requirements (FR) to their Implementation components and verifying Test cases.

| Req ID | Requirement Description | Implementation Component(s) | Verifying Test(s) |
|--------|--------------------------|----------------------------|-------------------|
| **FR-GEN-01** | Trace Generation Loop | `src/core/generation/trace_generator.py`<br>`src/core/domain/trace.py` | `tests/unit/test_trace_generator.py`*<br>`tests/integration/test_full_loop.py`* |
| **FR-GEN-02** | Guidance Constraints | `src/core/generation/step_generators.py` | `tests/unit/test_step_generators.py`* |
| **FR-GEN-03** | Function Removal | `src/core/generation/trace_generator.py` (augment logic)<br>`src/core/utils.py` | `tests/unit/test_function_augmentation.py` |
| **FR-GEN-04** | Function Shuffling | `src/core/utils.py` -> `shuffle_json_functions` | `tests/unit/test_utils.py` |
| **FR-GEN-05** | Prompt Randomization | `src/core/prompt_builder.py` | `tests/unit/test_prompt_builder.py` |
| **FR-CLI-01** | Command Interface | `src/cli/main.py`<br>`src/cli/commands/*` | Manual Verification / CLI Smoke Tests |
| **FR-CLI-02** | Verbose Mode | `src/cli/utils.py` (logging setup) | Manual Verification |
| **FR-FT-01** | Q-LoRA Training | `src/finetuning/model_finetuning.py` | `tests/integration/test_finetuning.py`* |
| **FR-FT-02** | Model Presets | `src/config/settings.py` -> `get_model_config` | `tests/unit/test_settings.py`* |
| **FR-FT-03** | Configurable Params | `src/config/settings.py` | `tests/unit/test_settings.py`* |
| **FR-FT-W01** | W&B Explicit Init | `src/cli/commands/finetune.py` (wandb.init call) | Manual Verification (W&B Dashboard) |
| **FR-FT-W02** | W&B Run Naming | `src/cli/commands/finetune.py` (run_name construction) | Manual Verification (W&B Dashboard) |
| **FR-FT-W03** | W&B Config Logging | `src/cli/commands/finetune.py` (wandb.init config dict) | Manual Verification (W&B Dashboard) |
| **FR-FT-W04** | W&B Backend Support | `src/cli/commands/finetune.py` (_train_with_unsloth, _train_with_trl) | Manual Verification (Both backends) |
| **FR-FT-W05** | W&B Finalization | `src/cli/commands/finetune.py` (wandb.finish calls) | Manual Verification (W&B Dashboard) |
| **FR-FT-W06** | W&B Optional | `src/cli/commands/finetune.py` (--no-wandb flag) | Manual Verification (CLI) |
| **FR-DATA-01** | HF Integration | `src/dataset_generation/hf_utils.py`*<br>`src/cli/commands/dataset.py` | Manual Integration Test |
| **FR-DATA-02** | Argilla Integration | `src/dataset_generation/hf_utils.py` | Manual Integration Test |
| **FR-DATA-03** | Online Diversification | `src/finetuning/model_finetuning.py` | `tests/unit/test_augmentation.py` |
| **FR-DATA-04** | Stochastic Config | `src/config/settings.py` | `tests/unit/test_augmentation.py` |
| **FR-DATA-05** | Feature Flipping | `src/finetuning/model_finetuning.py` | `tests/unit/test_augmentation.py` |
| **FR-DATA-06** | Parallel Execution | `src/finetuning/model_finetuning.py` | Integration Training Test |
| **FR-DATA-07** | Model-Aware Diversification | `src/config/settings.py` | Unit Test (Settings) |
| **FR-DATA-08** | Separated Augmentation Pipeline | `src/cli/commands/dataset.py` | `tests/integration/test_dataset_augmentation.py::TestDatasetAugmentationCLI` |
| **FR-DATA-09** | Markdown Link Preservation | `src/cli/commands/dataset.py::translate_preserving_links` | `tests/integration/test_dataset_augmentation.py::TestMarkdownLinkPreservation` |
| **FR-DATA-10** | Dataset Inspection | `src/cli/commands/dataset.py::inspect` | `tests/integration/test_dataset_augmentation.py::TestDatasetInspect` |
| **FR-DATA-11** | Dataset Size Multiplication | `src/augmentation/pipeline.py`* | `tests/integration/test_augmentation_pipeline.py::TestDatasetMultiplication`* |
| **FR-DATA-12** | Pipeline Configuration | `src/augmentation/pipeline.py`*<br>`src/augmentation/config.py`* | `tests/unit/test_augmentation_config.py::TestConfigValidation`* |
| **FR-DATA-13** | Feature Combination | `src/augmentation/pipeline.py`* | `tests/integration/test_augmentation_pipeline.py::TestFeatureCombination`* |
| **FR-DATA-14** | Post-Generation Deduplication | `src/augmentation/pipeline.py`* | `tests/unit/test_augmentation_pipeline.py::TestDeduplication`* |
| **FR-DATA-15** | Augmentation Metadata | `src/augmentation/metadata.py`* | `tests/unit/test_augmentation_metadata.py`* |
| **FR-DATA-16** | W&B Augmentation Tracking | `src/augmentation/tracking.py`* | `tests/integration/test_augmentation_pipeline.py::TestWandbTracking`* |
| **FR-DATA-17** | Config-Driven CLI | `src/cli/commands/dataset.py` | `tests/integration/test_augmentation_pipeline.py::TestConfigDrivenCLI`* |
| **FR-DATA-18** | Perplexity Diversity Metrics | `src/diversity/perplexity.py`* | `tests/unit/test_diversity_perplexity.py`* |
| **FR-DATA-19** | Diversity CLI | `src/cli/commands/dataset.py` | `tests/integration/test_diversity_cli.py`* |
| **FR-DATA-20** | Diversity in Metadata | `src/augmentation/metadata.py`<br>`src/diversity/perplexity.py`* | `tests/integration/test_augmentation_pipeline.py::TestDiversityMetadata`* |
| **FR-DATA-21** | Diversity Comparison | `src/diversity/perplexity.py`* | `tests/unit/test_diversity_perplexity.py::TestComparison`* |
| **FR-DATA-22** | Augmentation Effectiveness | `src/diversity/effectiveness.py`* | `tests/integration/test_diversity_effectiveness.py`* |
| **FR-DATA-23** | Quick Diversity Mode | `src/diversity/lexical.py`*<br>`src/diversity/semantic.py`* | `tests/unit/test_diversity_lexical.py`*<br>`tests/unit/test_diversity_semantic.py`* |
| **FR-DATA-24** | Dataset Format Command | `src/formatting/formatter.py`<br>`src/cli/commands/dataset.py` | `tests/unit/test_formatting_formatter.py`<br>`tests/integration/test_format_cli.py` |
| **FR-DATA-25** | Staged Augmentation | `src/formatting/steps.py`<br>`src/formatting/formatter.py` | `tests/unit/test_formatting_steps.py`<br>`tests/unit/test_formatting_formatter.py` |
| **FR-DATA-26** | Format Baseline | `src/formatting/formatter.py`<br>`src/formatting/config.py` | `tests/unit/test_formatting_formatter.py::TestDatasetFormatter::test_format_baseline`<br>`tests/integration/test_format_cli.py::TestFormatCLIBaseline` |
| **FR-DATA-27** | Format Config | `src/formatting/config.py` | `tests/unit/test_formatting_config.py` |
| **FR-DATA-28** | Diversity Evaluation Workflow | `src/cli/commands/dataset.py`<br>`src/formatting/formatter.py` | `tests/integration/test_format_cli.py::TestFormatDiversityWorkflow` |
| **FR-DIV-10** | HTML Report Generation | `src/diversity/report.py` | `tests/unit/test_diversity_report.py::TestGenerateReport` |
| **FR-DIV-11** | Report Visualizations | `src/diversity/report.py` | `tests/unit/test_diversity_report.py::TestGenerateReport::test_generate_report_contains_chartjs` |
| **FR-DIV-12** | Report CLI Flag | `src/cli/commands/dataset.py` | `tests/integration/test_diversity_cli.py::TestDiversityCLI::test_report_flag` |
| **FR-DIV-13** | Report Metadata | `src/diversity/report.py` | `tests/unit/test_diversity_report.py::TestPrepareReportData::test_prepare_includes_metadata` |
| **FR-DIV-14** | Report Comparison Mode | `src/diversity/report.py` | `tests/unit/test_diversity_report.py::TestPrepareReportData::test_prepare_comparison_mode`<br>`tests/integration/test_diversity_cli.py::TestDiversityCLI::test_report_comparison_mode` |
| **FR-DIV-15** | Report Single Dataset Mode | `src/diversity/report.py` | `tests/unit/test_diversity_report.py::TestPrepareReportData::test_prepare_single_mode`<br>`tests/integration/test_diversity_cli.py::TestDiversityCLI::test_report_single_mode` |
| **NFR-DIV-01** | Report Portability | `src/diversity/report.py` | `tests/unit/test_diversity_report.py::TestGenerateReport::test_generate_report_self_contained` |
| **NFR-DIV-02** | Report File Size | `src/diversity/report.py` | `tests/unit/test_diversity_report.py::TestReportIntegration::test_report_with_full_metrics` |
| **NFR-DIV-03** | Report Accessibility | `src/diversity/report.py` | `tests/unit/test_diversity_report.py::TestGenerateReport::test_generate_report_contains_styles` |
| **NFR-DIV-04** | Report Browser Compatibility | `src/diversity/report.py` | Manual Verification (Browser) |
| **FR-EVAL-01** | Completion-Only Perplexity | `src/diversity/perplexity.py::compute_completion_perplexity`<br>`src/diversity/utils.py::extract_completion` | `tests/unit/test_diversity_perplexity.py::TestCompletionPerplexity`* |
| **FR-EVAL-02** | Baseline Model Comparison | `src/diversity/evaluation.py::evaluate_model_robustness` | `tests/unit/test_diversity_evaluation.py::TestModelRobustness`* |
| **FR-EVAL-03** | Generalization Gap | `src/diversity/evaluation.py::evaluate_generalization` | `tests/unit/test_diversity_evaluation.py::TestGeneralization`* |
| **FR-EVAL-04** | Evaluation Mode Flag | `src/cli/commands/dataset.py` | `tests/integration/test_diversity_cli.py::TestDiversityCLI::test_eval_mode_flag`* |
| **FR-EVAL-05** | Robustness Report | `src/diversity/evaluation.py`<br>`src/diversity/report.py` | `tests/integration/test_diversity_cli.py::TestDiversityCLI::test_robustness_report`* |
| **NFR-EVAL-01** | Memory Efficiency | `src/diversity/evaluation.py` | Manual Verification (Single GPU) |
| **FR-PLAY-01** | Interactive Chat | `ui/src/App.tsx`<br>`ui/src/components/ChatInterface.tsx`<br>`src/server/routers/generation.py` | Manual Verification (Browser) |
| **FR-PLAY-02** | Model Management | `src/server/model_manager.py`<br>`src/server/routers/models.py`<br>`ui/src/components/Sidebar.tsx` | Manual Verification (UI Buttons) |
| **FR-PLAY-03** | Synthetic Red-Teaming | `src/server/routers/synthetic.py`<br>`ui/src/components/ChatInterface.tsx` | Manual Verification (Magic Buttons) |
| **FR-PLAY-04** | Robustness | `src/server/model_manager.py` (Async locks) | Manual Verification (Concurrent clicking) |
| **FR-PLAY-05** | Conversation History | `src/server/routers/conversations.py`<br>`ui/src/App.tsx` | Manual Verification (Refresh page) |
| **FR-PLAY-06** | Output Sanitization | `ui/src/App.tsx` (regex replace) | Manual Verification (Generate trace) |
| **FR-PLAY-09** | Conversation Deletion | `src/server/routers/conversations.py`<br>`ui/src/components/Sidebar.tsx` | `tests/unit/test_conversations.py` |
| **FR-PLAY-08** | Reference Highlighting | `ui/src/components/TraceRenderers.tsx` | Manual Verification |
| **NFR-USE-02** | Loading Feedback | `src/server/events.py`<br>`src/server/model_manager.py`<br>`ui/src/App.tsx` | Manual Verification (Logs/UI) |

*> Symbol denotes components/tests that are logically inferred to exist or should exist based on the architecture analysis. Files marked with `*` in the Test column indicate recommended coverage gaps if not already present.

## Gap Analysis

Based on the current analysis of `tests/` (Updated 2025-01-17):

1.  **Unit Tests**: Good coverage exists for:
    *   Utilities (`test_utils.py`), prompt building (`test_prompt_builder.py`)
    *   Trace generation logic (`test_trace_generator.py`) - covers **FR-GEN-01**
    *   Function augmentation (`test_function_augmentation.py`) - covers **FR-GEN-03**
    *   Settings and configuration (`test_settings.py`) - covers **FR-FT-02**, **FR-FT-03**
    *   Security (safe math eval, input validation) - covers **FR-SEC-01**, **FR-SEC-02**
    *   Server model manager (`test_idle_ejection.py`, `test_generation_utils.py`)
    *   Conversations API (`test_conversations.py`) - covers **FR-PLAY-09**
2.  **Remaining gaps** (lower priority):
    *   Finetuning logic tests (currently manual scripts)
    *   Full CLI command tests (using `click.testing.CliRunner`) for **FR-CLI-01**
    *   Model loading/ejection integration tests

## Security Requirements (Added from FM Audit 2025-01-17)

| Req ID | Requirement Description | Implementation Component(s) | Verifying Test(s) |
|--------|--------------------------|----------------------------|-------------------|
| **FR-SEC-01** | Safe Expression Evaluation | `src/server/routers/generation.py::safe_math_eval` | `tests/unit/test_security_fixes.py::TestSafeMathEvaluator`<br>`tests/unit/test_generation_utils.py::TestSafeMathEval` |
| **FR-SEC-02** | Input Validation | `src/server/schemas.py::GenerationRequest` | `tests/unit/test_security_fixes.py::TestInputValidation` |
| **FR-SEC-03** | CORS Configuration | `src/server/main.py` (CORS middleware) | Manual Verification |
| **FR-SEC-04** | Generation Timeout | `src/server/model_manager.py::generate`<br>`src/config/settings.py::generation_timeout_seconds` | `tests/unit/test_security_fixes.py::TestGenerationTimeout` |
| **FR-SEC-05** | Conversation ID Validation | `src/server/routers/conversations.py::_validate_conv_id` | Manual Verification |
| **FR-SEC-06** | Adapter ID Validation | `src/server/routers/models.py::_validate_adapter_id` | Manual Verification |
| **FR-SEC-07** | Model ID Validation | `src/server/routers/models.py::_validate_base_model_id` | Manual Verification |
| **NFR-REL-01** | Thread-Safe Singletons | `src/config/settings.py::get_settings`<br>`src/server/events.py::EventBroadcaster`<br>`src/server/model_manager.py::ModelManager` | `tests/unit/test_security_fixes.py::TestThreadSafeSingletons`<br>`tests/unit/test_generation_utils.py::TestModelManagerSingleton` |
| **NFR-REL-02** | Bounded SSE Queues | `src/server/events.py::MAX_SUBSCRIBER_QUEUE_SIZE`<br>`src/server/events.py::subscribe` | Manual Verification |
| **NFR-REL-04** | Thread-Safe Set Operations | `src/server/events.py::_subscribers_lock`<br>`src/server/events.py::publish` | Manual Verification |
| **NFR-REL-05** | TOCTOU Prevention | `src/server/model_manager.py::_detect_base_model_from_adapter` | Manual Verification |
| **NFR-REL-06** | Atomic Lock Acquisition | `src/server/model_manager.py::load_model` | Manual Verification |
| **NFR-REL-07** | Directory Creation | `src/server/routers/conversations.py::create_conversation` | Manual Verification |

## Traceability to Files

- **Logic**: `src/core/generation/` satisfies GEN requirements.
- **Config**: `src/config/settings.py` satisfies FT-02, FT-03 and NFR-CODE-02.
- **CLI**: `src/cli/commands/` satisfies CLI requirements and DATA-01.
- **Security**: `src/server/routers/generation.py`, `src/server/schemas.py` satisfies SEC requirements.

---

**Navigation:**
- [Test Strategy](TEST_STRATEGY.md)
- [Functional Requirements](../requirements/FUNCTIONAL.md)
- [Non-Functional Requirements](../requirements/NON_FUNCTIONAL.md)
