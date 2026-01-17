# Non-Functional Requirements

## Performance & Scalability
- **NFR-PERF-01 (Local Execution)**: The system shall be capable of running generation and training on consumer-grade GPUs (e.g., RTX 3090/4090) or local CPU (slow but functional).
- **NFR-PERF-02 (Efficiency)**: Trace generation should use token fast-forwarding (via Guidance) to minimize generation latency.

## Maintainability & Code Quality
- **NFR-CODE-01 (Type Safety)**: All Python code shall use type hints and pass `mypy` strict checking.
- **NFR-CODE-02 (Configuration)**: All configuration shall be centralized in a Pydantic Settings class, loadable from `.env` files.
- **NFR-CODE-03 (Modularity)**: The core generation logic must be decoupled from specific model implementations where possible.

## Usability
- **NFR-USE-01 (Documentation)**: The system shall provide clear architecture diagrams and "Getting Started" guides (fulfilled by `docs/`).
- **NFR-USE-02 (Feedback)**: Long-running operations, especially model loading, shall provide real-time, granular progress feedback to the user (e.g., percentage loaded) to prevent perception of freezing.

## Report Quality
- **NFR-DIV-01 (Report Portability)**: Generated HTML reports shall be self-contained single files that work offline without external dependencies (except CDN fallback for Chart.js).
- **NFR-DIV-02 (Report File Size)**: Generated HTML reports shall not exceed 500KB including all embedded data, styles, and JavaScript.
- **NFR-DIV-03 (Report Accessibility)**: Generated HTML reports shall support dark mode via CSS `prefers-color-scheme` media query and be responsive for mobile viewing.
- **NFR-DIV-04 (Report Browser Compatibility)**: Generated HTML reports shall render correctly in modern browsers (Chrome, Firefox, Safari, Edge) without requiring additional plugins.

## Evaluation Quality
- **NFR-EVAL-01 (Memory Efficiency)**: Model evaluation shall support sequential model loading to enable comparison on systems with limited GPU memory (one model at a time).

## Reliability & Thread Safety (Added from FM Audit 2025-01-17)
- **NFR-REL-01 (Thread-Safe Singletons)**: All singleton patterns (Settings, EventBroadcaster, ModelManager) shall use double-checked locking with explicit locks to prevent race conditions during initialization.
- **NFR-REL-02 (Bounded Queues)**: SSE subscriber queues shall have bounded sizes (max 100 messages) to prevent memory exhaustion from slow consumers. Messages shall be dropped with a warning when queues are full.
- **NFR-REL-03 (Graceful Degradation)**: GPU memory allocation failures shall trigger cleanup and clear error messages rather than leaving partial state.
- **NFR-REL-04 (Thread-Safe Set Operations)**: Collection iteration during concurrent modification shall be protected by taking snapshots under lock (FM-33).
- **NFR-REL-05 (TOCTOU Prevention)**: File existence checks shall use try/except instead of exists() checks followed by open() (FM-39).
- **NFR-REL-06 (Atomic Lock Acquisition)**: Concurrent operation guards shall use atomic lock acquisition with timeout instead of non-atomic locked() checks (FM-43).
- **NFR-REL-07 (Directory Creation)**: Write operations shall ensure parent directories exist before attempting file creation (FM-35).

---

**Navigation:**
- [Functional Requirements](FUNCTIONAL.md)
- [Traceability Matrix](../testing/TRACEABILITY_MATRIX.md)
- [Test Strategy](../testing/TEST_STRATEGY.md)
