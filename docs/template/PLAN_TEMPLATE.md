# Implementation Plan: <Feature Name>

**Version:** 1.0.0
**Date:** <Current Date>
**Status:** Ready for Implementation
**Based On:** <RFC or source document if applicable>

---

## Executive Summary

<2-3 sentences describing what this plan implements, the core problem it solves, and the approach taken.>

**Estimated Effort:** <N phases over ~X weeks/days>
**Risk Level:** <Low/Medium/High> (<brief justification>)

---

## Table of Contents

1. [Phase 1: <Name>](#phase-1-name)
2. [Phase 2: <Name>](#phase-2-name)
...
N. [Phase N: <Name>](#phase-n-name)

---

## Current State

### Files to Modify

| File | Purpose | Changes |
|------|---------|---------|
| `path/to/file.ts` | <Current purpose> | <What changes> |

### Files to Create

| File | Purpose |
|------|---------|
| `path/to/new-file.ts` | <Purpose of new file> |

---

## Phase 1: <Phase Name>

**Goal:** <One sentence describing what this phase achieves>

### 1.1 <Sub-task>

<Detailed description of sub-task with code examples if applicable>

```typescript
// Code example showing the implementation approach
```

### 1.2 <Sub-task>

<Description>

### 1.N Commit Checkpoint

```bash
git add <files>
git commit -m "<type>: <description>"
```

---

## Phase 2: <Phase Name>

**Goal:** <One sentence>

### 2.1 <Sub-task>
...

### 2.N Commit Checkpoint

```bash
git add <files>
git commit -m "<type>: <description>"
```

---

<Repeat for all phases...>

---

## Final Phase: Self-Reflection & Experience Capitalization

**Goal:** Capture learnings before task completion

### N.1 Review What Was Learned
- What mistakes were made and how were they resolved?
- What patterns worked well?
- What patterns failed?
- What codebase-specific knowledge was discovered?

### N.2 Update EXPERIENCES.md
- Add new entries to appropriate sections
- Add session summary to "Recently Added"

### N.3 Commit Checkpoint
```bash
git add EXPERIENCES.md
git commit -m "docs: update EXPERIENCES.md with session learnings"
```

---

## Commit Summary

| Phase | Commit Message |
|-------|----------------|
| 1 | `<type>: <description>` |
| 2 | `<type>: <description>` |
| ... | ... |

---

## Risk Mitigation

### Fallback Strategy

<Describe how to revert or fallback if implementation fails>

### Rollback Plan

```bash
# Commands to rollback if needed
```

---

## Success Criteria

- [ ] <Criterion 1>
- [ ] <Criterion 2>
- [ ] <Criterion N>

---

## Appendix: <Additional Info>

<Environment variables, configuration options, reference links, etc.>

---

## Plan Generation Guidelines

1. **Explore First**: Before writing the plan, use the Explore agent to understand:
   - Current implementation of related features
   - Files that need modification
   - Integration points with existing code
   - Test infrastructure available

2. **Phase Structure**:
   - **Phase 1**: Always start with specification/documentation updates
   - **Phase 2**: Test infrastructure (TDD approach)
   - **Middle Phases**: Core implementation, broken into logical units
   - **Second-to-Last Phase**: Documentation updates
   - **Last Phase**: Cleanup, finalization, version bump
   - **MANDATORY Final Step**: Self-reflection and EXPERIENCES.md update

3. **Commit Checkpoints**: Each phase ends with a commit checkpoint specifying:
   - Files to stage
   - Commit message following conventional commits format

4. **Code Examples**: Include TypeScript code examples for:
   - New interfaces/types
   - Key implementation patterns
   - Configuration structures

5. **Risk Assessment**: Always include:
   - Fallback strategy (how to disable/revert)
   - Rollback commands
   - Success criteria checklist
