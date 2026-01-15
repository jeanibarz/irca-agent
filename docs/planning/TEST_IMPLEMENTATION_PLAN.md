# Testing Implementation Plan

## 1. Goal
Implement missing tests identified in `TRACEABILITY_MATRIX.md` to ensure full coverage of Functional Requirements, specifically **FR-GEN-01 (Trace Generation)**, **FR-GEN-02 (Guidance Constraints)**, and **FR-GEN-03 (Function Removal)**.

## 2. Missing Tests to Implement

### A. `tests/unit/test_trace_generator.py` (Mocked)
**Goal:** Verify the `TraceGenerator` logic without running an actual LLM.
**Reqs:** FR-GEN-01, FR-GEN-03

**Strategy:**
1.  Mock the `guidance` model object.
2.  Mock the step generator functions (`generate_thought`, `generate_function_call`, etc.) to return pre-determined strings immediately.
3.  **Test Cases:**
    *   `test_generate_single_trace_success`: Verify standard loop (Thought -> Action: Call -> Output -> Action: Answer).
    *   `test_generate_trace_missing_function`: Verify that `generate_trace_missing_function` correctly modifies the available functions list before calling the generation loop.
    *   `test_max_steps_reached`: Verify loop terminates if `max_steps` is hit.

### B. `tests/unit/test_step_generators.py` (Mocked)
**Goal:** Verify that step generators construct the correct Guidance programs (structure only).
**Reqs:** FR-GEN-02

**Strategy:**
1.  This is tricky because `guidance` programs are functional.
2.  We can at least verify that the functions accept the expected arguments and return a `guidance` Program/Function object.
3.  We might skip deep validation of the guidance internal graph for now, as it's implementation detail of the library.

### C. `tests/integration/test_full_loop.py` (TinyLlama)
**Goal:** Verify the full pipeline actually runs from end-to-end.
**Reqs:** FR-GEN-01, System Level

**Strategy:**
1.  See if we can run a "dummy" model in Guidance (e.g. `Mock` model provided by library) instead of loading weights.
2.  If not, mark this test with `@pytest.mark.slow` or `@pytest.mark.gpu`.

## 3. Implementation Steps

1.  **Create Mocks**: Create `tests/mocks.py` containing a `MockGuidanceModel` helper.
2.  **Implement `test_trace_generator.py`**:
    *   Setup `TraceGenerator` with mock model.
    *   Patch `src.core.generation.trace_generator.step_generators` to avoid external dependencies.
    *   Write the test cases.
3.  **Run Tests**: Verify pass.

## 4. Immediate Action
I will now implement `tests/unit/test_trace_generator.py` as it is the most critical gap.
