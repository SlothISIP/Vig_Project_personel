---
name: Issue Fixer
description: Fix known issues identified in project analysis
tags: fix, debug, issues
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Task
---

# Issue Fixer Agent

## Task
Fix known issues identified in the project analysis.

## Known Issues Registry

### CRITICAL
1. **State Synchronization Missing**
   - Location: feedback_loop.py + digital_twin_env.py
   - Problem: No sync mechanism between Feedback Loop and RL Environment health scores
   - Fix: Add state synchronization callback

2. **External Lines Validation Missing**
   - Location: digital_twin_env.py:94-146
   - Problem: No type/structure validation for external_production_lines
   - Fix: Add validation in constructor

### HIGH
3. **Service Initialization Order**
   - Location: main_integrated.py:169-246
   - Problem: No dependency graph for service initialization
   - Fix: Implement dependency injection

4. **Job Status Error Handling**
   - Location: scheduler.py:246
   - Problem: Returns dict with "error" key, not None
   - Fix: Standardize return type

5. **WebSocket Race Condition**
   - Location: main_integrated.py:909-924
   - Problem: Client list modified during iteration
   - Fix: Use thread-safe collection

### MEDIUM
6. **Health Update Non-Atomic**
   - Location: feedback_loop.py:116-150
   - Problem: No locking on shared state
   - Fix: Add threading lock

7. **Dashboard Cache Unbounded**
   - Location: main_integrated.py:754-770
   - Problem: Cache can grow unbounded
   - Fix: Add max size limit

### LOW
8. **Type Hint Error**
   - Location: scheduler.py:25
   - Problem: `any` instead of `Any`
   - Fix: Import and use typing.Any

## Instructions

1. **If no arguments**: List all issues with status

2. **If issue number provided** (e.g., "1", "2"): Fix that specific issue
   - Read the relevant files
   - Implement the fix
   - Run related tests
   - Report changes made

3. **If "all" argument**: Fix all issues in priority order

4. **If "critical" argument**: Fix only CRITICAL issues

5. **If "test" argument**: Run tests for fixed issues

## Usage Examples
- `/fix-issue 1` - Fix state synchronization
- `/fix-issue critical` - Fix all critical issues
- `/fix-issue all` - Fix everything

Arguments: $ARGUMENTS
