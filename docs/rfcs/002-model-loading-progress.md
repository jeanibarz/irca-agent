
# RFC: Model Loading Progress Feedback

**Version:** 1.0.0
**Date:** 2026-01-15
**Status:** Draft
**Author:** Antigravity

---

## Executive Summary

This RFC proposes adding real-time feedback for model loading operations in the UI. Currently, loading a large LLM (e.g. 7GB+) freezes the UI state in a "Loading..." spinner for extended periods (30s+), leaving users unsure if the process has hung. The proposed solution is to implement Server-Sent Events (SSE) to stream granular status updates (e.g., "Downloading", "Loading Shards", "Moving to GPU") from the backend to the frontend, displayed via a progress bar or status text.

**Current Behavior:**
```
[User Clicks Load] -> [Spinner (Indeterminate)] -> (Wait ~30s) -> [Success]
```

**Proposed Behavior:**
```
[User Clicks Load] -> [Progress Bar: 0%] "Initializing..."
                   -> [Progress Bar: 20%] "Checking Cache..."
                   -> [Progress Bar: 40%] "Loading Model Shards..."
                   -> [Progress Bar: 80%] "Loading Tokenizer..."
                   -> [Progress Bar: 90%] "Moving to GPU..."
                   -> [Success]
```

---

## 1. Motivation

### 1.1 Current Limitations

1. **User Anxiety**: Large models take time. Without feedback, users often refresh the page or click "Load" again, potentially triggering race conditions or errors (though our new Lock system prevents the latter, the UX is still poor).
2. **Hidden Failures**: If the download stalls, the user has no visibility until the global timeout is reached.
3. **Opaque Process**: Users don't know if the system is downloading (network bound) or loading (disk/compute bound).

### 1.2 Use Cases

| Use Case | Why This Feature Helps |
|----------|------------------------|
| **Loading New Model** | User sees "Downloading..." and knows it will take longer due to network. |
| **Switching Adapters** | User sees rapid progress as base model is reused. |
| **Large Model Load** | User sees "Loading Shards" and understands the delay is checking integrity. |

---

## 2. Current System Analysis

### 2.1 System Lifecycle

Currently, `ModelManager.load_model` runs in a thread `asyncio.to_thread`. It calls `AutoModelForCausalLM.from_pretrained`. This function blocks until completion.

### 2.2 Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `ModelManager` | `src/server/model_manager.py` | Orchestrates loading. |
| `from_pretrained` | `transformers` library | Actual heavy lifting. |
| `API Client` | `ui/src/lib/api.ts` | Initiates XHR request. |

### 2.3 Current Data Flow

```
UI -> POST /v1/model/load -> ModelManager (Lock) -> worker_thread -> DONE
```

There is no side-channel for status.

---

## 3. Proposed Solution

### 3.1 Architecture Changes

1.  **Event Broadcasting**: Introduce a simple `EventBroadcaster` singleton in the backend that any component can publish messages to.
2.  **SSE Endpoint**: Create a `GET /v1/events` endpoint that keeps a connection open and streams JSON events from the broadcaster.
3.  **Granular Loading Steps**: Break `load_model` into logical chunks (where possible) and emit status events between them.

### 3.2 High-Level Flow

```
┌─────────────┐       ┌───────────────┐       ┌─────────────────┐
│     UI      │       │ Events Router │       │  ModelManager   │
└──────┬──────┘       └───────┬───────┘       └────────┬────────┘
       │ Subscribes (SSE)     │                        │
       ├─────────────────────►│                        │
       │                      │ Register Client        │
       │                      │                        │
       │ POST /model/load     │                        │
       ├──────────────────────────────────────────────►│
       │                      │                        │ Accepts Request
       │                      │                        │ Emits "Loading Base"
       │                      │◄───────────────────────┤
       │ "Status: Base..."    │                        │ calls from_pretrained()
       │◄─────────────────────┤                        │
       │                      │                        │ Emits "Loading Adapter"
       │                      │◄───────────────────────┤
       │ "Status: Adapter..." │                        │ calls PeftModel...
       │◄─────────────────────┤                        │
       │                      │                        │ Returns 200 OK
       │  Success             │                        │
       │◄──────────────────────────────────────────────┤
```

---

## 4. Detailed Design

### 4.1 Backend Changes

#### 4.1.1 Event Broadcaster (`src/server/events.py`)

A simple asyncio-based publisher-subscriber pattern.

```python
class EventBroadcaster:
    def __init__(self):
        self._queues = set()

    async def subscribe(self):
        q = asyncio.Queue()
        self._queues.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._queues.remove(q)

    def publish(self, event_type: str, data: dict):
        for q in self._queues:
            q.put_nowait({"type": event_type, "data": data})
```

#### 4.1.2 Model Manager Integration

Inject `EventBroadcaster` into `ModelManager`. Update `load_model`:

```python
await self.broadcaster.publish("model_progress", {"step": "init", "message": "Checking configuration..."})
# ...
await self.broadcaster.publish("model_progress", {"step": "download", "message": "Loading model files..."})
# ...
await self.broadcaster.publish("model_progress", {"step": "gpu", "message": "Moving to GPU..."})
```

**Note on Download Progress**: `transformers` uses `tqdm`. Hooking into `tqdm` to pipe output to our broadcaster is possible but brittle. For V1 of this RFC, we will stick to **Phase-based progress** (0%, 25%, 50%, 75%, 100%) which is safe and deterministic.

### 4.2 Frontend Changes

#### 4.2.1 EventHook

Create a global hook/provider `useServerEvents` in React that connects to `/v1/events` on mount and exposes the latest events.

#### 4.2.2 Sidebar Update

Update the "Load" button or area to show a `<ProgressBar value={progress} label={status} />` when `isLoading` is true.

### 4.3 API Changes

| Endpoint | Method | Change |
|----------|--------|--------|
| `/v1/events` | GET | **NEW**: Returns `text/event-stream` |

---

## 5. Implementation Plan

### Phase 1: Infrastructure

1. Create `src/server/events.py` (Broadcaster).
2. Add `GET /v1/events` to `src/server/main.py`.
3. Add basic frontend SSE consumption code.

**Commit:** `feat(server): add sse event broadcasting infrastructure`

### Phase 2: Instrumentation

1. Update `ModelManager` to use the broadcaster.
2. Add broadcast calls at key checkpoints in `load_model` and `eject_model`.

**Commit:** `feat(models): broadcast loading progress events`

### Phase 3: UI Feedback

1. Create `ProgressBar` component.
2. Update `Sidebar` to listen to "model_progress" events and update state.

**Commit:** `feat(ui): display model loading progress in sidebar`

### Phase 4: Verification

1. Manual test with a large model reset.

**Commit:** `docs: update manual test procedures`

---

## 6. Testing Strategy

### 6.1 Unit Tests
*   `test_events.py`: Verify subscribers receive published messages.
*   `test_model_manager.py`: Mock the broadcaster and verify `publish` is called during load sequence.

### 6.2 Manual Verification
*   **Slow Network Sim**: Use `tc` or developer tools to throttle network (if downloading) or just observe locally.
*   **Concurrency**: Open two tabs, ensure both receive events (SSE is unicast per connection, but broadcaster fans out).

---

## 7. Risks & Mitigations

### 7.1 Connection Limits
**Risk**: Browsers have a limit on simultaneous HTTP connections (SSE counts as one).
**Mitigation**: For a local single-user tool, this is negligible (limit is usually 6 per domain).

### 7.2 Event Ordering
**Risk**: "Success" HTTP response arrives before final "100%" SSE event due to race.
**Mitigation**: Frontend should clear progress bar on HTTP success/failure explicitly, regardless of SSE state.
