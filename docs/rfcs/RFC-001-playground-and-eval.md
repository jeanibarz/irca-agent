# RFC: IRCA Playground & Evaluation Suite

**Version:** 1.0.0
**Date:** 2026-01-15
**Status:** Draft
**Author:** Antigravity

---

## Executive Summary

This RFC proposes the development of the **IRCA Playground**, a web-based interface for interactive testing, showcasing, and evaluating function-calling models. Currently, testing is limited to CLI scripts and lacks a visual feedback loop for iterating on model quality. The proposed solution is a modern React/FastAPI web application that enables parameter configuration (including increasing max tokens to 4096), real-time chat with function execution simulation, and an integrated "Judge" system to score trace accuracy. This will significantly improve the developer experience and provide a "wow" factor for demonstrating capabilities.

**Current Behavior:**
```
Terminal -> irca CLI -> Model Inference -> Text Output
(No visual feedback, no parameter tuning UI, limited context)
```

**Proposed Behavior:**
```
Web UI (React) -> Backend API (FastAPI) -> Model Inference -> Rich Chat UI
                                      -> Function Sandbox (Live/Dummy)
                                      -> Judge LLM (Score & Feedback)
```

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Current System Analysis](#2-current-system-analysis)
3. [Proposed Solution](#3-proposed-solution)
4. [Detailed Design](#4-detailed-design)
5. [Implementation Plan](#5-implementation-plan)
6. [Migration & Compatibility](#6-migration--compatibility)
7. [Testing Strategy](#7-testing-strategy)
8. [Risks & Mitigations](#8-risks--mitigations)
9. [Appendix: Code References](#9-appendix-code-references)

---

## 1. Motivation

### 1.1 Current Limitations

1. **Lack of Interactive Testing**: Testing relies on modifying and running Python scripts (`test_inference.py`). Changing prompts, parameters (like `max_new_tokens`), or functions requires code changes.
2. **Limited Visibility**: Text CLI output truncates complex traces (e.g., current 256 token limit) and makes it hard to distinguish between thoughts, function calls, and responses visually.
3. **No Evaluation Loop**: There is no easy way to "score" a generation. Verifying correctness requires manual inspection of JSON strings.
4. **Poor Demonstration Value**: A terminal window is not an effective way to showcase "rich aesthetics" or "premium capabilities" to stakeholders.

### 1.2 Use Cases

| Use Case | Why This Feature Helps |
|----------|------------------------|
| **Demo / Showcase** | Provides a polished, reliable visual interface to demonstrate agent capabilities to users/investors. |
| **Model Debugging** | Allows rapid iteration on prompt engineering and hyperparameter tuning (e.g., Temperature, Top-P). |
| **Function Testing** | Enables testing scenarios with "real" function execution to verify end-to-end logic. |
| **Regression Testing** | The "Judge" system serves as a qualitative metric to catch regressions in logic reasoning. |

---

## 2. Current System Analysis

### 2.1 System Lifecycle / Flow

Currently, the inference logic is embedded in scripts:

```
┌─────────────────────────────────┐
│        CLI / SCRIPT FLOW        │
├─────────────────────────────────┤
│                                 │
│  Script (test_inference.py)     │
│       │                         │
│       ▼                         │
│  Load Model (Slow, ~30s)        │
│       │                         │
│       ▼                         │
│  Build Prompt (Hardcoded)       │
│       │                         │
│       ▼                         │
│  Generate (Limited tokens)      │
│       │                         │
│       ▼                         │
│  Print to Stdout                │
│                                 │
└─────────────────────────────────┘
```

### 2.2 Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **Prompt Builder** | `src/core/prompt_builder.py` | Constructs the text input for the model. |
| **Inference Script** | `scripts/test_inference.py` | ad-hoc script for running generation. |
| **Settings** | `src/config/settings.py` | Configuration constants. |

---

## 3. Proposed Solution

### 3.1 Architecture Overview

I propose a split-stack architecture to ensure scalability and separation of concerns:

1.  **Backend (FastAPI)**:
    *   Stateful model loader (keeps model in VRAM).
    *   Endpoints for `generate`, `list_functions`, `execute_function`.
    *   Integration with "Judge" LLM (e.g., via OpenAI API or another local model).
2.  **Frontend (React/Vite)**:
    *   Modern, "Dark Mode" aesthetic using TailwindCSS + Framer Motion.
    *   Chat interface with specialized rendering for "Thoughts" (collapsible), "Function Calls" (code blocks), and "Results".
    *   Settings sidebar for Model parameters (Max Tokens, Temp) and System Prompt customization.

### 3.2 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       IRCA PLAYGROUND FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     Start      ┌─────────────┐     Prompt      ┌─────────────┐
│  │  Browser    │ ─────────────► │   FastAPI   │ ─────────────► │  LLM Model  │
│  │  Interface  │ ◄───────────── │             │ ◄───────────── │             │
│  └──────┬──────┘     Stream     └──────┬──────┘     Token       └─────────────┘
│         │                              │ Stream                                │
│         ▼                              ▼                                       │
│  Render Rich UI                 Function Call?                                 │
│  (Thoughts/Code)        ┌──────────────┴──────────────┐                        │
│                         ▼                             ▼                        │
│                 ┌──────────────┐              ┌──────────────┐                 │
│                 │ Dummy Sandbox│              │ Real Sandbox │                 │
│                 └──────────────┘              └──────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Design

### 4.1 Backend API Design (FastAPI)

**Service**: `src/server/`

#### 4.1.1 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available finetuned models & adapters. |
| `/v1/model/load` | POST | Load a specific model adapter into GPU memory. |
| `/v1/chat/completions` | POST | OpenAI-compatible chat generation endpoint. **Default max_tokens: 4096**. |
| `/v1/judge` | POST | Submit a trace for evaluation by a judge LLM. |

#### 4.1.2 Data Schemas

```python
class GenerationRequest(BaseModel):
    query: str
    system_prompt: str
    functions: List[Dict]
    temperature: float = 0.7
    max_tokens: int = 4096  # UPDATED default
    mock_functions: bool = True
```

### 4.2 Frontend Design (React)

**App**: `ui/`

#### 4.2.1 Features

*   **Left Sidebar**:
    *   Model Selector (Dropdown).
    *   Connection Status indicator.
    *   Hyperparameter sliders (Temperature, Top P).
    *   "Max Tokens" input (Default 4096).
*   **Main Chat Area**:
    *   User messages (Right aligned).
    *   Assistant messages (Left aligned).
    *   **Trace Rendering**:
        *   `<Thought>` block: Rendered as a collapsible "Thinking..." accordion detailed view.
        *   `<Call>` block: Rendered as a Syntax-highlighted JSON code block.
        *   `<Output>` block: Rendered as a diff-style or data-table view.
*   **Judge Panel** (Overlay/Bottom):
    *   "Review this interaction" button.
    *   Displays Score (1-10) and Critique text from the Judge LLM.

### 4.3 Evaluation "Judge" System

To automate quality assurance, we will implement a `JudgeService`.

**Logic**:
1.  Receive full conversation history.
2.  Construct a prompt for a stronger model (e.g., GPT-4 or local large model).
3.  Prompt: "You are an expert evaluator. Rate the assistant's performance on: 1. Function Selection, 2. Argument Correctness, 3. Reasoning. Score 0-10."
4.  Return structured Feedback.

---

## 5. Implementation Plan

### Phase 1: Backend Foundation (Medium Risk)

*Goal: Expose the model via API and solve the token truncation issue.*

1.  Create `src/server/main.py`, `src/server/routers/`.
2.  Implement `ModelManager` class to handle loading/unloading adapters efficiently.
3.  Implement `/v1/chat/completions` generation endpoint with `max_tokens=4096` support.
4.  Add unit tests for API endpoints.

**Commit:** `feat(server): implement fastapi backend for model inference`

### Phase 2: React Frontend (High Risk - "Wow Factor")

*Goal: Build the "Premium" visual interface.*

1.  Initialize Vite project in `ui/`.
2.  Implement "Glassmorphism" layout with TailwindCSS.
3.  Build `ChatInterface` component with special rendering for thoughts/function calls.
4.  Connect to Backend API.

**Commit:** `feat(ui): implement premium playground interface`

### Phase 3: Judge Integration (Low Risk)

*Goal: Add validatior/scoring.*

1.  Implement `src/core/evaluation/judge.py`.
2.  Add `/v1/judge` endpoint.
3.  Add "Review" button to UI.

**Commit:** `feat(eval): implement llm-based judge system`

---

## 6. Migration & Compatibility

*   **Script Compatibility**: `scripts/test_inference.py` will remain as a lightweight test.
*   **Dependencies**: Need to add `fastapi`, `uvicorn`, `pydantic` (already in), `requests` to poetry. Frontend needs `npm` stack.

---

## 7. Testing Strategy

### 7.1 Backend Tests
| Test ID | Description |
|---------|-------------|
| UT-API-001 | Test model loading state management |
| UT-API-002 | Test generation with dummy functions |
| IT-API-001 | End-to-end generation request |

### 7.2 Frontend Tests
| Test ID | Description |
|---------|-------------|
| E2E-UI-001 | User can send message and receive response |
| E2E-UI-002 | Thoughts are collapsible |

---

## 8. Risks & Mitigations

### 8.1 Model Loading Latency
**Risk**: Loading Mistral-7B takes 30s+, causing API timeouts or bad UX.
**Mitigation**:
*   Async loading with WebSocket status updates to UI.
*   "Keep-alive" cache for the model.

### 8.2 Frontend Aesthetic Quality
**Risk**: UI looks generic or "bootstrap-like".
**Mitigation**:
*   Use `framer-motion` for smooth animations.
*   Use a curated color palette (Dark Slate + Neon accents).
*   Follow "Linear" or "Vercel" design language.

---
