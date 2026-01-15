
# Project Session Memory

**Last Updated:** 2026-01-15
**Last Commit:** `feat(ui): add conversation history and token sanitization` - Implemented persistent chat history and output cleaning.
**Branch:** main
**Task ID:** playground-v1-complete

---

## Current Task

**Goal:** Implement Interactive Playground and Evaluation Suite (RFC-001)
**Status:** ✅ Completed (Phase 1 & History)
**Started:** 2026-01-15
**Task Type:** feature

### What I Did

1.  **Frontend Polish**:
    *   Migrated to Tailwind CSS v4.
    *   Added **Available Tools** to Sidebar.
    *   Implemented **Conversation History** list in Sidebar.
    *   Implemented **"New Chat"** functionality.
    *   **Output Sanitization**: Automatically strips`<|wait|>` and other control tokens from the display while preserving them in the backend generation logic if needed (though currently we just clean the display).

2.  **Synthetic Red-Teaming**:
    *   Implemented `POST /v1/synthetic/query` endpoint with "Feasible" and "Infeasible" modes.
    *   Fixed bug in `synthetic.py` where `max_token` was passed instead of `max_new_tokens`.

3.  **Robust Backend**:
    *   **Async/Locking**: Prevented race conditions in model loading.
    *   **Persistence**: Created `src/server/routers/conversations.py` to save/load chats from `data/sessions/`.
    *   **Ejection**: Allowed manual GPU memory clearing.

4.  **Documentation**:
    *   Updated `docs/REQUIREMENTS.md` with `FR-PLAY-05` (History) and `FR-PLAY-06` (Sanitization).
    *   Updated `docs/TRACEABILITY_MATRIX.md`.

### Key Findings

-   **FastAPI Async Blocking**: Calling synchronous CPU/GPU heavy functions inside `async def` endpoints blocks the event loop. Use `asyncio.to_thread`.
-   **Token Leaks**: Models trained with specific control tokens (like `<|wait|>`) often output them. Frontend sanitization is necessary to provide a clean UX.
-   **Local Persistence**: Quick file-based JSON storage is an effective way to adding "Memory" to a local playground without setting up a full database.

### Files Created/Modified

| File | Purpose |
|------|---------|
| `src/server/routers/conversations.py` | Backend logic for chat persistence. |
| `ui/src/App.tsx` | Main state machine (History, Models, Chat). |
| `ui/src/components/Sidebar.tsx` | UI for History list and Model control. |
| `ui/src/lib/api.ts` | API client additions for History. |
| `docs/REQUIREMENTS.md` | New requirements logged. |

---

## Session Progress

### Completed This Session
- ✅ Configured separate Frontend (Vite) and Backend (FastAPI).
- ✅ Implemented Tool visualization and Synthetic Red-Teaming.
- ✅ Implemented Robust Backend (Async, Eject, Lock).
- ✅ Implemented Persistent Conversation History.
- ✅ Implemented Output Token Sanitization.

### Task Validation

-   **Command**: `npm run dev` + `python -m src.server.main`.
-   **Result**:
    -   Chat history survives page reloads.
    -   "New Chat" clears context correctly.
    -   `<|wait|>` is removed from "Assistant" messages in the UI.

---

## Next Steps

1.  **RFC Phase 2: Function Execution Sandbox**:
    -   Current "Tools" are dummy definitions.
    -   Goal: Actually execute python code (e.g. `get_stock_price`) in a safe sandbox (Docker/gVisor).
2.  **LLM-as-a-Judge**:
    -   Automate the "synthetic query -> generation -> verify" loop using a stronger model (or the same model) to grade the response.

---

## Files to Read After Memory Reset

1.  `MEMORY.md`
2.  `EXPERIENCES.md`
3.  `GEMINI.md`
