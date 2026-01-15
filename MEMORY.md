
# Project Session Memory

**Last Updated:** 2026-01-15
**Last Commit:** `feat(core): robustness for model loading, ejection, visibility and synthetic testing` - Implemented robust model management and synthetic query generation.
**Branch:** main
**Task ID:** playground-v1

---

## Current Task

**Goal:** Implement Interactive Playground and Evaluation Suite (RFC-001)
**Status:** ✅ Completed (Phase 1)
**Started:** 2026-01-15
**Task Type:** feature

### What I Did

1.  **Frontend Polish**:
    *   Migrated to Tailwind CSS v4 to fix build errors (`@import "tailwindcss"`).
    *   Improved UI design (dark/glassmorphism) and readability (dropdown contrast).
    *   Added **Available Tools** section to Sidebar to visualize capabilities.

2.  **Synthetic Red-Teaming**:
    *   Implemented `POST /v1/synthetic/query` endpoint.
    *   Added **"Generate Feasible Query"** and **"Generate Infeasible Query"** buttons to Chat Interface.
    *   This allows rapid stress-testing of the model's tool calling and refusal logic.

3.  **Robust Model Management**:
    *   Implemented proper locking in `ModelManager` (asyncio locks).
    *   Made `load_model` and `generate` async/non-blocking using `asyncio.to_thread`.
    *   Added `POST /v1/model/eject` and `GET /v1/model/current` endpoints.
    *   Updated UI to show loaded status badges ("LOADED") and allow ejection.

4.  **Documentation**:
    *   Updated `docs/REQUIREMENTS.md` with new FR-PLAY requirements.
    *   Updated `docs/TRACEABILITY_MATRIX.md` with manual verification links.

### Key Findings

-   **FastAPI Async Blocking**: Calling synchronous CPU/GPU heavy functions (like `transformers.generate`) inside an `async def` endpoint blocks the entire event loop, freezing other requests (like health checks or status polling). Always use `await asyncio.to_thread` for these operations.
-   **Tailwind v4 Migration**: The new Tailwind version requires specific CSS import syntax (`@import "tailwindcss"`) and no longer uses `@tailwind base/components/utilities` or `@layer`/`@apply` in the same way for basic setups without PostCSS plugins being correctly configured for legacy support.
-   **React Polling**: Simple `setInterval` polling in `useEffect` is sufficient for local development status updates (like checking if a model is loaded) without complex websocket setups.

### Files Created/Modified

| File | Purpose |
|------|---------|
| `ui/src/components/Sidebar.tsx` | UI component for model selection and tools display. |
| `ui/src/components/ChatInterface.tsx` | Main chat UI with synthetic query buttons. |
| `ui/src/App.tsx` | Main application state (models, messages, config). |
| `src/server/routers/synthetic.py` | Backend endpoint for synthetic query generation. |
| `src/server/model_manager.py` | Core model logic (Async updates, Locking). |
| `docs/REQUIREMENTS.md` | Requirement definitions. |

---

## Session Progress

### Completed This Session
- ✅ Configured separate Frontend (Vite) and Backend (FastAPI).
- ✅ Implemented Tool visualization.
- ✅ Implemented Synthetic Query Generation (Red-Teaming).
- ✅ Implemented Robust Backend (Async, Eject, Lock).
- ✅ Updated Documentation (Requirements, Traceability).

### Task Validation

-   **Command**: `npm run dev` (Frontend) + `python -m src.server.main` (Backend).
-   **Result**:
    -   UI Loads without errors.
    -   Models list correctly.
    -   Loading locks UI prevents concurrent loads.
    -   Eject works and clears memory.
    -   Synthetic queries generate feasible/infeasible prompts correctly.

---

## Key Decisions Made

1.  **Synthetic Query via Model**: Instead of hardcoding prompts, we use the *currently loaded model* to generate synthetic user queries. This acts as a self-check (can the model understand its own tools?) and ensures variety.
2.  **Explicit "Eject"**: Added explicit manual memory management because GPU memory is scarce (consumer hardware target), allowing users to switch models cleanly.
3.  **Asyncio for GPU Ops**: Wrap blocking GPU calls in threads to ensure the web server remains responsive (e.g., capable of accepting a "Cancel" or "Status" request, though cancellation isn't fully implemented yet, status is).

---

## Next Steps

1.  **Refactor Server**: Organize `src/server` structure if it grows (currently good).
2.  **Implement "Judge"**: Add the LLM-as-a-Judge automated evaluation (RFC Phase 2).
3.  **Function Execution Sandbox**: Actually execute the `get_weather` calls using a sandboxed environment (Python Docker sandbox).

---

## Files to Read After Memory Reset

1.  `MEMORY.md`
2.  `EXPERIENCES.md`
3.  `GEMINI.md`
