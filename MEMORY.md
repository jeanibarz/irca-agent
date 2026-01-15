
# Project Session Memory

**Last Updated:** 2026-01-15
**Last Commit:** `fix(core): ensure prompt is stripped from generation output` - Fixed prompt echoing.
**Branch:** main
**Task ID:** playground-v1-refinement

---

## Current Task

**Goal:** Implement Interactive Playground and Evaluation Suite (RFC-001)
**Status:** ✅ Completed (Phase 1 & Refinements)
**Started:** 2026-01-15
**Task Type:** feature

### What I Did

1.  **Frontend Polish**:
    *   Migrated to Tailwind CSS v4.
    *   Added **Available Tools** to Sidebar.
    *   Implemented **Conversation History** list in Sidebar.
    *   Implemented **"New Chat"** functionality.
    *   **Output Sanitization**: Automatically strips `<|wait|>` and other control tokens from the display.

2.  **Synthetic Red-Teaming**:
    *   Implemented `POST /v1/synthetic/query` endpoint with "Feasible" and "Infeasible" modes.
    *   **Fixed Generation**: Updated `ModelManager` to slice output tokens strictly (removing prompt echo).

3.  **Robust Backend**:
    *   **Async/Locking**: Prevented race conditions in model loading.
    *   **Persistence**: Created `src/server/routers/conversations.py` to save/load chats from `data/sessions/`.
    *   **Ejection**: Allowed manual GPU memory clearing.

4.  **Documentation**:
    *   Updated `docs/REQUIREMENTS.md` with Playgroung requirements including `FR-PLAY-07` (Dual Model support).
    *   Updated `docs/TRACEABILITY_MATRIX.md`.

### Key Findings

-   **Prompt Stripping**: String-based prompt stripping (`response.startswith(prompt)`) is unreliable because of tokenization artifacts (models adding spaces/newlines). Always use `input_token_len` to slice the output tensor directly: `outputs[0][input_len:]`.
-   **Multi-Model vs Single**: Synthetic Data Generation originally used a separate model, but we reverted to a **Single Model architecture** (alias `default`) to simplify the UX and resource management. The UI tabs were removed in favor of a unified model configuration.
-   **Model Loading Progress**: Capturing `tqdm` progress from `transformers` in a headless server environment is tricky. `sys.stderr` capture works for loading shards, but explicit `hf_logging.enable_progress_bar()` is needed. A hybrid approach (Real Progress + Fallback "Downloading..." heartbeat) works best.
-   **Cross-Lingual Training**: To train robust agents, augmenting the dataset with translated queries/answers while keeping reasoning traces in English is highly effective. On-the-fly augmentation during training (using `dataset.map`) avoids static file maintenance and allows for stochastic variety.

### Files Created/Modified

| File | Purpose |
|------|---------|
| `src/server/model_manager.py` | Progress capture & Rebalanced model alias logic. |
| `src/dataset_generation/translator.py` | `TranslationService` using Opus-MT models. |
| `src/finetuning/model_finetuning.py` | Integrated `augment_dataset` logic into training loop. |
| `src/config/settings.py` | Added `AugmentationConfig` fields for multilingual support. |
| `ui/src/components/Sidebar.tsx` | Reverted to single model UI (removed tabs). |
| `docs/rfcs/003-multilingual-augmentation.md` | RFC for training-time translation. |

---

## Session Progress

### Completed This Session
- ✅ Configured separate Frontend (Vite) and Backend (FastAPI).
- ✅ Implemented Tool visualization and Synthetic Red-Teaming.
- ✅ Implemented Robust Backend (Async, Eject, Lock).
- ✅ Implemented Persistent Conversation History.
- ✅ Implemented Output Token Sanitization.
- ✅ Fixed Generation Output (Prompt Echoing).
- ✅ **New**: Added Unit Tests for Function Augmentation (FR-GEN-03).
- ✅ **New**: Reverted to Single Model Architecture - Simplified Sidebar and Routers.
- ✅ **New**: Implemented Reference Highlighting (FR-PLAY-08) - Interactive tool citations in Final Answer.
- ✅ **New**: Implemented Model Loading Progress Bar (RFC-002) - Granular Tqdm capture + SSE.
- ✅ **New**: Implemented Online Multilingual Diversification (RFC-003) - Real-time, parallelized translation using Iterable Datasets.
- ✅ **New**: Optimized Translation Cache - Class-level model reuse for efficiency.
- ✅ **New**: Parallel Data Loading - Prefetching translations to avoid training bottlenecks.
- ✅ **New**: Requirement FR-DATA-06 - Formalized online diversification as the standard training method.

### Task Validation

-   **Command**: `npm run dev` + `python -m src.server.main`.
-   **Result**:
    -   Synthetic Queries now appear cleanly (without the prompt text).
    -   Chat history survives page reloads.
    -   "New Chat" clears context correctly.
    -   **Augmentation Tests**: `pytest tests/unit/test_augmentation.py` passes with coverage.

---

## Next Steps

1.  **RFC Phase 2: Function Execution Sandbox**:
    -   Current "Tools" are dummy definitions.
    -   Goal: Actually execute python code (e.g. `get_stock_price`) in a safe sandbox (Docker/gVisor).
2.  **LLM-as-a-Judge**:
    -   Automate the "synthetic query -> generation -> verify" loop using a stronger model.

---

## Files to Read After Memory Reset

1.  `MEMORY.md`
2.  `EXPERIENCES.md`
3.  `GEMINI.md`
