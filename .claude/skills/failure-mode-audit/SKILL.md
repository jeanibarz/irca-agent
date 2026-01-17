---
name: failure-mode-audit
description: Perform comprehensive failure mode analysis and remediation. Use when auditing code for potential issues, finding edge cases, analyzing error handling gaps, or reviewing code safety.
---

# Failure Mode Audit & Remediation Workflow

## Overview

This skill performs a comprehensive failure mode analysis of a specified area of the codebase, generates a structured report, and then implements fixes for identified issues.

## Usage

```
/failure-mode-audit [target-area]
```

**Examples:**
- `/failure-mode-audit Unsloth integration`
- `/failure-mode-audit API authentication`
- `/failure-mode-audit Database connection handling`

---

## Workflow Steps

### Phase 1: Failure Mode Analysis

Launch the `failure-mode-analyst` agent with the following prompt template:

```
Perform a comprehensive failure mode analysis of [TARGET_AREA] in the codebase. Focus on:

1. **Code Review**: Analyze the relevant files for:
   - Import order issues
   - Error handling gaps
   - Race conditions
   - Resource leaks (memory, file handles, connections)
   - Configuration inconsistencies
   - Missing edge case handling
   - Thread safety issues

2. **Test Coverage Evaluation**:
   - Review existing tests for the target area
   - Identify missing test scenarios
   - Check for untested error paths
   - Verify edge cases are covered

3. **Documentation Consistency**:
   - Check that code behavior matches documentation
   - Identify undocumented behaviors
   - Flag stale documentation

4. **Assumptions Analysis**:
   - List explicit assumptions in the code
   - Identify implicit assumptions
   - Document environmental assumptions
   - Note human/operational assumptions

Provide a structured report with:

## Report Format

### Failure Modes Table
| ID | Issue | Severity | File:Line | Description |
|----|-------|----------|-----------|-------------|
| FM-XX | Brief title | CRITICAL/HIGH/MEDIUM/LOW | path/file.py:123 | Description |

### Severity Definitions
- **CRITICAL**: Can cause data loss, security breach, or system crash
- **HIGH**: Causes incorrect behavior or significant user impact
- **MEDIUM**: Causes degraded experience or requires workaround
- **LOW**: Minor issue or code quality concern

### For Each Failure Mode, Include:
1. **Description**: What the issue is
2. **Risk**: What could go wrong
3. **Current Mitigation**: Any existing safeguards (or "None")
4. **Recommendation**: Concrete fix with code snippet if applicable

### Verification Gaps (Missing Tests)
| ID | Missing Test | Covers FM | Priority |
|----|--------------|-----------|----------|
| VG-XX | Description | FM-XX | HIGH/MEDIUM/LOW |

### Worst-Case Scenarios
Describe 2-3 realistic scenarios where these failure modes could cause significant problems.
```

### Phase 2: Triage and Planning

After receiving the report:

1. **Create a todo list** with all identified issues, ordered by severity
2. **Group related issues** that can be fixed together
3. **Identify quick wins** (low effort, high impact fixes)

### Phase 3: Implementation

For each failure mode, starting with CRITICAL/HIGH severity:

1. **Read the relevant code** to understand current state
2. **Implement the fix** following the recommendation
3. **Add/update tests** for the verification gaps
4. **Mark todo as completed** before moving to next item

### Phase 4: Verification

After all fixes are implemented:

1. **Run the test suite** to verify no regressions
2. **Re-run the failure mode analyst** (optional) to verify fixes and catch any new issues introduced
3. **Document changes** in commit messages referencing FM-XX IDs

### Phase 5: Documentation Updates (CRITICAL/HIGH Issues)

**MANDATORY** for CRITICAL and HIGH severity issues. Update project documentation to maintain coherence:

#### 5.1 Requirements Updates (`docs/requirements/`)

For each CRITICAL/HIGH failure mode, determine if it reveals:

| Scenario | Action |
|----------|--------|
| Missing requirement | Add new FR-XX or NFR-XX to appropriate spec file |
| Incorrect requirement | Update existing requirement with correct behavior |
| Undocumented constraint | Add new requirement capturing the constraint |

**New Requirement Template:**
```markdown
### FR-{AREA}-XXX: {Title derived from FM-XX}

**Added:** YYYY-MM-DD (from FM-XX failure mode audit)

The system shall {behavior that prevents the failure mode}.

**Acceptance Criteria:**
- [ ] Given {precondition}, when {action}, then {expected outcome}

**Linked Tests:** UT-{AREA}-XXX, IT-{AREA}-XXX
```

#### 5.2 Test Specification Updates (`docs/testing/`)

For each new test added (from Verification Gaps):

1. **Add test definition** to `docs/testing/TEST_STRATEGY.md`:
   ```markdown
   ### UT-{AREA}-XXX: {Test description from VG-XX}

   **Covers:** FM-XX, FR-{AREA}-XXX
   **Type:** Unit | Integration | E2E
   **File:** tests/unit/test_{area}.py

   **Test Cases:**
   - {test case 1}
   - {test case 2}
   ```

2. **Link to requirements** - Ensure each test references its requirement ID

#### 5.3 Traceability Matrix Updates (`docs/testing/TRACEABILITY_MATRIX.md`)

Add entries linking new requirements to new tests:

```markdown
| Req ID | Requirement Title | Test IDs | Coverage |
|--------|-------------------|----------|----------|
| FR-{AREA}-XXX | {From FM-XX} | UT-{AREA}-XXX | Full |
```

And reverse mapping:

```markdown
| Test ID | Test Description | Req IDs | Type |
|---------|------------------|---------|------|
| UT-{AREA}-XXX | {From VG-XX} | FR-{AREA}-XXX | Unit |
```

#### 5.4 Documentation Checklist

Before completing the audit, verify:

- [ ] All CRITICAL/HIGH FMs have corresponding requirements (new or existing)
- [ ] All new tests are documented in docs/testing/TEST_STRATEGY.md
- [ ] Traceability matrix has bidirectional links (Req→Test and Test→Req)
- [ ] No orphaned tests (tests without requirements)
- [ ] No untested requirements (requirements without tests)

---

## Example Session

```
User: Run failure mode analysis on the Unsloth integration

Agent: I'll launch the failure-mode-analyst agent to audit the Unsloth integration.

[Launches Task with failure-mode-analyst subagent]

Agent: The failure mode analysis is complete. Here's the report:

## Failure Modes Table
| ID | Issue | Severity | File:Line | Description |
|----|-------|----------|-----------|-------------|
| FM-01 | Import order race | CRITICAL | model_manager.py:15 | TORCHDYNAMO_DISABLE set after torch import |
| FM-02 | Silent fallback | HIGH | model_manager.py:180 | No warning when Unsloth unavailable |
| FM-03 | Singleton not thread-safe | HIGH | model_manager.py:45 | Race condition in get_instance() |

## Verification Gaps
| ID | Missing Test | Covers FM | Priority |
|----|--------------|-----------|----------|
| VG-01 | Test import order | FM-01 | HIGH |
| VG-02 | Test thread safety | FM-03 | HIGH |

I'll create a todo list and start implementing fixes...

[Creates todo list with items]
[Implements FM-01 fix - moves env var to top of file]
[Implements FM-02 fix - adds broadcaster warning]
[Implements FM-03 fix - adds thread-safe singleton]
[Adds tests for VG-01, VG-02]
[Runs test suite - all pass]

Agent: All failure modes have been addressed:
- FM-01: Fixed import order (env var now at module top)
- FM-02: Added explicit warning via broadcaster
- FM-03: Added thread-safe singleton with double-checked locking
- Added 4 new tests covering the verification gaps

Now updating documentation for CRITICAL/HIGH issues...

[Updates docs/requirements/FUNCTIONAL.md]
Added:
- NFR-SRV-001: Environment variables shall be set before ML library imports
- NFR-SRV-002: ModelManager shall use thread-safe singleton pattern
- FR-SRV-010: System shall warn users when falling back to non-optimized backend

[Updates docs/testing/TEST_STRATEGY.md]
Added test definitions:
- UT-SRV-001: Verify TORCHDYNAMO_DISABLE set before torch import
- UT-SRV-002: Verify ModelManager singleton thread safety
- UT-SRV-003: Verify fallback warning broadcast

[Updates docs/testing/TRACEABILITY_MATRIX.md]
Added bidirectional links:
| NFR-SRV-001 | Env var import order | UT-SRV-001 | Full |
| NFR-SRV-002 | Thread-safe singleton | UT-SRV-002 | Full |
| FR-SRV-010 | Fallback warning | UT-SRV-003 | Full |

Documentation checklist complete:
- [x] All CRITICAL/HIGH FMs have corresponding requirements
- [x] All new tests documented in docs/testing/TEST_STRATEGY.md
- [x] Traceability matrix updated with bidirectional links
- [x] No orphaned tests
- [x] No untested requirements

All 27 tests pass. Would you like me to run another audit to verify the fixes?
```

---

## Quick Reference

### Launching the Analysis

```python
# In Claude Code, use the Task tool:
Task(
    subagent_type="failure-mode-analyst",
    prompt="Perform a comprehensive failure mode analysis of [TARGET_AREA]...",
    description="Audit [TARGET_AREA]"
)
```

### Common Target Areas

| Target Area | Key Files to Analyze |
|-------------|---------------------|
| Model loading | `model_manager.py`, `constants.py` |
| Training pipeline | `finetune.py`, `train_unsloth.py` |
| Dataset processing | `formatting/`, `diversity/` |
| API endpoints | `server/routers/` |
| Configuration | `settings.py`, `constants.py` |

### Severity Priority Order

1. **CRITICAL** - Fix immediately, blocks release, **requires doc updates**
2. **HIGH** - Fix before next release, **requires doc updates**
3. **MEDIUM** - Schedule for upcoming sprint
4. **LOW** - Add to backlog

### Documentation Files to Update (CRITICAL/HIGH)

| File | Purpose | When to Update |
|------|---------|----------------|
| `docs/requirements/FUNCTIONAL.md` | Add/update FR-XX requirements | Missing or incorrect behavior |
| `docs/requirements/NON_FUNCTIONAL.md` | Add NFR-XX for quality constraints | Performance, safety, reliability issues |
| `docs/testing/TEST_STRATEGY.md` | Document new tests (UT/IT/E2E) | Each VG-XX verification gap |
| `docs/testing/TRACEABILITY_MATRIX.md` | Link requirements ↔ tests | Every new requirement or test |

### Commit Message Format

```
fix(component): address FM-XX description

- Describe the fix
- Reference verification gap if applicable

Fixes: FM-XX
Tests: VG-XX
```

---

## Benefits

1. **Systematic coverage** - Ensures no failure modes are overlooked
2. **Prioritized fixes** - Severity-based ordering focuses effort
3. **Test-driven** - Verification gaps ensure coverage
4. **Documented** - FM-XX IDs provide traceability
5. **Iterative** - Can re-run to verify fixes and catch new issues
6. **Documentation coherence** - CRITICAL/HIGH issues trigger requirements and test spec updates
7. **Traceability maintained** - Bidirectional links between requirements, tests, and failure modes
