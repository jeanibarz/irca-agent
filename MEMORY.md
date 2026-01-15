
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
-   **Dual Model Use**: Synthetic Data Generation often requires a smarter/creative base model, while the model being tested might be a specialized narrow finetune. In the future, we should allow loading separate models for these tasks.

### Files Created/Modified

| File | Purpose |
|------|---------|
| `src/server/model_manager.py` | Core model generation logic (Prompt slicing). |
| `src/server/routers/conversations.py` | Chat persistence. |
| `ui/src/App.tsx` | Main state machine. |
| `docs/REQUIREMENTS.md` | New `FR-PLAY-07`. |
| `docs/TRACEABILITY_MATRIX.md` | Gap analysis. |

---

## Session Progress

### Completed This Session
- ✅ Configured separate Frontend (Vite) and Backend (FastAPI).
- ✅ Implemented Tool visualization and Synthetic Red-Teaming.
- ✅ Implemented Robust Backend (Async, Eject, Lock).
- ✅ Implemented Persistent Conversation History.
- ✅ Implemented Output Token Sanitization.
- ✅ Fixed Generation Output (Prompt Echoing).

### Task Validation

-   **Command**: `npm run dev` + `python -m src.server.main`.
-   **Result**:
    -   Synthetic Queries now appear cleanly (without the prompt text).
    -   Chat history survives page reloads.
    -   "New Chat" clears context correctly.

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
