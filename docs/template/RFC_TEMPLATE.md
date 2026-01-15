# RFC: {Feature Name}

**Version:** 1.0.0
**Date:** {YYYY-MM-DD}
**Status:** Draft | Review | Approved | Implemented | Deprecated
**Author:** {Author Name}

---

## Executive Summary

{2-3 sentences describing:
- What this RFC proposes
- The core problem it solves
- The high-level approach}

**Current Behavior:**
```
{Diagram or description of current flow}
```

**Proposed Behavior:**
```
{Diagram or description of proposed flow}
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

{List specific problems with the current implementation:}

1. **{Problem 1 Title}**: {Description of the limitation}

2. **{Problem 2 Title}**: {Description of the limitation}

3. **{Problem 3 Title}**: {Description of the limitation}

### 1.2 Use Cases

{Describe scenarios where this feature is needed:}

| Use Case | Why This Feature Helps |
|----------|------------------------|
| {Use case 1} | {Benefit} |
| {Use case 2} | {Benefit} |
| {Use case 3} | {Benefit} |

### 1.3 Tradeoffs

{Compare current vs proposed approach:}

| Aspect | Current Approach | Proposed Approach |
|--------|------------------|-------------------|
| {Aspect 1} | {Current behavior} | {New behavior} |
| {Aspect 2} | {Current behavior} | {New behavior} |
| {Aspect 3} | {Current behavior} | {New behavior} |

---

## 2. Current System Analysis

### 2.1 System Lifecycle / Flow

{ASCII diagram showing current flow:}

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         {SYSTEM NAME} LIFECYCLE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  {Step 1}                                                                   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────┐                                                            │
│  │  {State 1}  │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │  {State 2}  │ ──► │  {State 3}  │ ──► │  {State 4}  │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

{Table of components involved:}

| Component | File | Responsibility |
|-----------|------|----------------|
| **{Component 1}** | `{path/to/file.ts}` | {What it does} |
| **{Component 2}** | `{path/to/file.ts}` | {What it does} |
| **{Component 3}** | `{path/to/file.ts}` | {What it does} |

### 2.3 Current Data Flow / State Machine

{Describe current state transitions:}

```
{state1} → {state2} → {state3} → {state4}
                 ↘              ↗
                   {error_state}
```

**Key Points:**
- {Important observation 1}
- {Important observation 2}
- {Important observation 3}

### 2.4 Existing Related Features

{Document any existing features that are related or can be leveraged:}

```typescript
// Example of existing code that's relevant
// From {filename}:{line}
{code snippet}
```

**{Feature name}**: {How it works and how it relates to this RFC}

---

## 3. Proposed Solution

### 3.1 New Configuration / Options

{Describe new fields, types, or options:}

```typescript
// New type definition
export type {NewType} = '{option1}' | '{option2}';

// Updated interface
export interface {InterfaceName} {
    // ... existing fields ...
    {newField}: {NewType};  // NEW - {description}
}
```

### 3.2 Behavior Changes

{Table comparing old vs new behavior:}

| Aspect | {Option 1} (current) | {Option 2} (new) |
|--------|---------------------|------------------|
| **{Behavior 1}** | {Current} | {New} |
| **{Behavior 2}** | {Current} | {New} |
| **{Behavior 3}** | {Current} | {New} |

### 3.3 High-Level Flow

{ASCII diagram of proposed flow:}

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    {FEATURE NAME} LIFECYCLE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │  {State 1}  │ {new_field}: '{value}'                                     │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │  {State 2}  │ {description of new behavior}                              │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │  {State 3}  │                                                            │
│  └─────────────┘                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Design

### 4.1 Schema Changes

#### 4.1.1 {Table Name} Table

```sql
-- Migration: {Description}
ALTER TABLE {table_name}
ADD COLUMN {column_name} {DATA_TYPE} {CONSTRAINTS};

-- Add constraint if needed
ALTER TABLE {table_name}
ADD CONSTRAINT {constraint_name}
CHECK ({column_name} IN ('{value1}', '{value2}'));
```

#### 4.1.2 {Another Table} Table

```sql
-- Migration: {Description}
ALTER TABLE {table_name}
ADD COLUMN IF NOT EXISTS {column_name} {DATA_TYPE};
```

### 4.2 Type Changes

#### 4.2.1 {package}/src/types/{file}.ts

```typescript
// New type
export type {NewType} = '{option1}' | '{option2}';

// Update existing interface
export interface {InterfaceName} {
    // ... existing fields ...
    {newField}: {NewType};  // NEW - {description}
}

// Update input interface
export interface {CreateInputName} {
    // ... existing fields ...
    {newField}?: {NewType};  // NEW - default: '{default}'
}
```

### 4.3 {Component 1} Changes

#### 4.3.1 {methodName}() - {Description}

```typescript
// {path/to/file.ts}

async {methodName}({params}): Promise<{ReturnType}> {
    // ... existing logic ...

    // NEW: {Description of new logic}
    const {newVariable} = {calculation};

    // NEW: {Another change}
    if ({condition}) {
        {new behavior}
    }

    // ... rest of method ...
}
```

#### 4.3.2 {anotherMethod}() - {Description}

```typescript
// NEW helper method
private {helperMethod}({params}): {ReturnType} {
    // {Implementation}
}
```

### 4.4 {Component 2} Changes

{Repeat pattern for each component that needs changes}

### 4.5 {Component 3} Changes

{Repeat pattern for each component that needs changes}

### 4.6 API Changes

{If API endpoints are affected:}

| Endpoint | Method | Change |
|----------|--------|--------|
| `/api/v1/{resource}` | POST | Accept `{newField}` in body |
| `/api/v1/{resource}` | GET | Return `{newField}` in response |

---

## 5. Implementation Plan

### Phase 1: {Phase Name} ({Risk Level} Risk)

{Description of what this phase accomplishes}

1. {Step 1}
2. {Step 2}
3. {Step 3}

**Commit:** `{type}({scope}): {description}`

### Phase 2: {Phase Name} ({Risk Level} Risk)

{Description}

1. {Step 1}
2. {Step 2}

**Commit:** `{type}({scope}): {description}`

### Phase 3: {Phase Name} ({Risk Level} Risk)

{Continue for all phases...}

**Commit:** `{type}({scope}): {description}`

### Phase N: Tests (Required)

1. Unit tests for {component}
2. Integration tests for {flow}
3. E2E tests for {user journey}

**Commit:** `test({scope}): {description}`

### Phase N+1: Documentation

1. Update FUNCTIONAL_SPECIFICATIONS.md
2. Update EXPERIENCES.md
3. Update MEMORY.md

**Commit:** `docs: {description}`

---

## 6. Migration & Compatibility

### 6.1 Backward Compatibility

- **Default value**: `{field}` defaults to `'{value}'`
- **Existing {resources}**: Continue to work unchanged
- **New {resources}**: Must explicitly opt-in to new behavior

### 6.2 Database Migration

```sql
-- Migration {number}_{name}.sql

-- Add {field} to {table} table
ALTER TABLE {table}
ADD COLUMN IF NOT EXISTS {field} {TYPE} NOT NULL DEFAULT '{default}';

-- Add constraint
ALTER TABLE {table}
ADD CONSTRAINT IF NOT EXISTS {constraint_name}
CHECK ({field} IN ('{value1}', '{value2}'));

-- Index if needed
CREATE INDEX IF NOT EXISTS idx_{table}_{field}
ON {table}({field})
WHERE {condition};
```

### 6.3 API Compatibility

- `POST /api/v1/{resource}`: Accept optional `{field}` field
- `GET /api/v1/{resource}`: Return `{field}` in response
- No breaking changes to existing clients

---

## 7. Testing Strategy

### 7.1 Unit Tests

| Test ID | Description | File |
|---------|-------------|------|
| UT-{FEATURE}-001 | {Test description} | {file}.test.ts |
| UT-{FEATURE}-002 | {Test description} | {file}.test.ts |
| UT-{FEATURE}-003 | {Test description} | {file}.test.ts |
| UT-{FEATURE}-004 | {Test description} | {file}.test.ts |

### 7.2 Integration Tests

| Test ID | Description | File |
|---------|-------------|------|
| IT-{FEATURE}-001 | {Test description} | {file}.integration.test.ts |
| IT-{FEATURE}-002 | {Test description} | {file}.integration.test.ts |
| IT-{FEATURE}-003 | {Test description} | {file}.integration.test.ts |

### 7.3 E2E Tests

| Test ID | Description | File |
|---------|-------------|------|
| E2E-{FEATURE}-001 | {Test description} | {file}.spec.ts |
| E2E-{FEATURE}-002 | {Test description} | {file}.spec.ts |

---

## 8. Risks & Mitigations

### 8.1 {Risk 1 Title}

**Risk:** {Description of what could go wrong}

**Mitigation:**
- {Mitigation strategy 1}
- {Mitigation strategy 2}
- Future: {Potential future improvement}

### 8.2 {Risk 2 Title}

**Risk:** {Description}

**Mitigation:**
- {Strategy}

### 8.3 {Risk 3 Title}

**Risk:** {Description}

**Mitigation:**
- {Strategy}

---

## 9. Appendix: Code References

### 9.1 Key Files to Modify

| File | Changes |
|------|---------|
| `{path/to/file1.ts}` | {Description of changes} |
| `{path/to/file2.ts}` | {Description of changes} |
| `{path/to/file3.ts}` | {Description of changes} |
| `migrations/{number}_{name}.sql` | Database migration |

### 9.2 Existing Relevant Code

```typescript
// Current {feature} handling in {file}:{line}
{code snippet showing current implementation}

// Current {another feature} in {file}:{line}
{code snippet}
```

### 9.3 New Requirements (to add to FUNCTIONAL_SPECIFICATIONS.md)

```markdown
#### FR-{CATEGORY}-{NUMBER}: {Requirement Title}

**Status:** Draft
**Priority:** P{1-3} - {Critical|High|Medium|Low}

**Requirement:** The system shall {requirement description using "shall"}.

**Acceptance Criteria:**
- [ ] {Criterion 1}
- [ ] {Criterion 2}
- [ ] {Criterion 3}

**Linked Tests:** UT-{FEATURE}-001 to UT-{FEATURE}-00N, IT-{FEATURE}-001 to IT-{FEATURE}-00N
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | {YYYY-MM-DD} | {Author} | Initial draft |
| 1.0.1 | {YYYY-MM-DD} | {Author} | {Changes made} |

---

## Template Usage Notes

{DELETE THIS SECTION WHEN USING THE TEMPLATE}

### When to Use This Template

Use this RFC template when proposing:
- New features that affect multiple components
- Architectural changes
- Changes to data models or APIs
- New workflow or lifecycle patterns
- Any change requiring migration

### How to Fill In

1. **Executive Summary**: Write LAST, after you understand the full design
2. **Motivation**: Start here - clearly articulate the problem
3. **Current System Analysis**: Read code, document what exists
4. **Proposed Solution**: High-level approach before diving into details
5. **Detailed Design**: Code-level changes with examples
6. **Implementation Plan**: Break into phases with commits
7. **Testing Strategy**: Define tests for each requirement
8. **Risks**: Think about what could go wrong

### Quality Checklist

- [ ] All placeholders replaced with actual content
- [ ] ASCII diagrams are clear and properly formatted
- [ ] Code examples are syntactically correct
- [ ] File paths are accurate (verified with Glob/Grep)
- [ ] Migration SQL is valid
- [ ] Test IDs follow naming convention
- [ ] Risks have concrete mitigations
- [ ] Implementation phases have commit messages
