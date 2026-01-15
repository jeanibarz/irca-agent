# Documentation

Welcome to the `irca-agent` documentation.

## User Guides

- [**Data Generation**](DATA_GENERATION.md): Learn how the agent trace generation pipeline works, including the core loop, step generation, and data augmentation techniques.
- [**Finetuning**](FINETUNING.md): Instructions on how to fine-tune supported models (Mistral, TinyLlama, etc.) using the generated datasets and Q-LoRA.

## Technical Documentation

- [**Architecture**](ARCHITECTURE.md): High-level overview of the system components, including Domain Models, Generation Engine, and Finetuning.
- [**API Reference**](api-reference.md): Detailed documentation of the core Python classes and functions.
- [**Development**](DEVELOPMENT.md): Guide for initializing the dev environment, running tests, and debugging.
- [**Guidance Library**](guidance-library.md): Reference for the underlying [Guidance](https://github.com/guidance-ai/guidance) pattern used for logic control.

## Templates

See the [template/](template/) directory for standardization templates:
- `RFC_TEMPLATE.md`: Request for Comments.
- `ADR_TEMPLATE.md`: Architectural Decision Records.
- `PLAN_TEMPLATE.md`: Planning template.
- And more.
