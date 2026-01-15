
# RFC: Multilingual Dataset Augmentation

**Version:** 1.0.0
**Date:** 2026-01-15
**Status:** Draft
**Author:** Antigravity

---

## Executive Summary

This RFC proposes adding a feature to the dataset generation pipeline to automatically augment datasets with multilingual examples. Specifically, it targets the translation of **User Queries** and **Final Answers** from English to a target language (e.g., French), while preserving the **Reasoning Traces** (Thoughts, Tool Calls) in English. This approach enforces "Language-Agnostic Reasoning" patterns in the model, teaching it to reason in its strongest language (English) regardless of the input language, and to correctly switch back to the user's language for the final response.

## 1. Motivation

### 1.1 The "Franglais" Problem
When training agentic models on non-English data, a common failure mode is "Franglais" reasoning, where the model mixes English tool definitions with French thought processes, leading to hallucinated tool names (e.g., `obtenir_meteo` instead of `get_weather`) and syntax errors.

### 1.2 The Goal
We want an agent that:
1.  **Understands** queries in French (or any target language).
2.  **Reasons** and calls tools in rigorous English (matching the Python API).
3.  **Responds** in French.

By explicitly training on examples that follow this pattern ($Input_{FR} \rightarrow Trace_{EN} \rightarrow Output_{FR}$), we anchor this behavior.

---

## 2. Proposed Solution

### 2.1 Training-Time Augmentation

Instead of creating a separate static dataset, we will integrate augmentation directly into the training pipeline (`src/finetuning/model_finetuning.py`). This allows for:
-   **Dynamic Configuration**: Users can choose the augmentation language and ratio at runtime via CLI arguments.
-   **Simpler Workflow**: No intermediate files to manage. "Just Train".

The system will perform the following steps during the "Dataset Loading" phase:
1.  Load original English dataset.
2.  Check for `--augment_lang` argument.
3.  If present, initialize local NMT model (e.g., `opus-mt-en-fr`).
4.  Apply translation to a subset (`--augment_ratio`) of the dataset using `dataset.map()`.
5.  Proceed to training with the mixed-language dataset.

### 2.2 Translation Strategy

We will use **Helsinki-NLP Opus-MT models** (e.g., `Helsinki-NLP/opus-mt-en-fr`) via the `transformers` library on the CPU (or GPU if available/free).

**Fields to Translate:**
-   `user_query`: English $\rightarrow$ Target.
-   `final_answer`: English $\rightarrow$ Target.
-   `trace` (Thought/Action): **KEEP ENGLISH**.

### 2.3 Example Transformation

**Original (English):**
```json
{
  "query": "What is the stock price of Apple?",
  "trace": [
    {"thought": "I need to check the stock price."},
    {"tool": "get_stock_price", "args": {"symbol": "AAPL"}},
    {"observation": "150.00"}
  ],
  "answer": "The current price of Apple is $150.00."
}
```

**Augmented (French):**
```json
{
  "query": "Quel est le cours de l'action Apple ?",
  "trace": [
    {"thought": "I need to check the stock price."},
    {"tool": "get_stock_price", "args": {"symbol": "AAPL"}},
    {"observation": "150.00"}
  ],
  "answer": "Le prix actuel d'Apple est de 150,00 $."
}
```

---

## 3. Detailed Design

### 3.1 New Components

#### `TranslationService` Class
Located in `src/dataset_generation/translator.py`.
-   Manages loading of `AutoModelForSeq2SeqLM` and `AutoTokenizer`.
-   Implements caching to avoid reloading models.
-   Provides a `translate_batch(texts: list[str]) -> list[str]` method.

#### `model_finetuning.py` Update
-   Add `--augment_lang` (str) and `--augment_ratio` (float) arguments.
-   Inject `augment_dataset_with_translation(dataset, lang, ratio)` step before Trainer initialization.

### 3.2 Dependencies

New dependencies required in `pyproject.toml`:
-   `sentencepiece` (Required for Opus-MT tokenizers).
-   `sacremoses` (Required for Opus-MT tokenizers).


---

## 4. Risks & Mitigations

### 4.1 Translation Quality
**Risk**: NMT models might mistranslate technical terms or change entities (e.g., "Apple" key -> "Pomme").
**Mitigation**:
-   Opus-MT is generally robust for major languages.
-   Since we *don't* translate the reasoning/tool args (which contain the strict entities like "AAPL"), the risk of functional breakage is low. The prompt is fuzzy, so minor translation errors in the query are acceptable (and arguably improve robustness).

### 4.2 Performance
**Risk**: Translation is slow (model inference).
**Mitigation**:
-   Use batching.
-   Allow GPU acceleration if available.
-   Show progress bar (using our existing `tqdm` patterns).

---
