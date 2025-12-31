---
name: Code Quality Checker
description: Run code quality checks including linting, type checking, and formatting
tags: quality, linting, testing, formatting
allowed-tools: Read, Grep, Glob, Bash, Task
---

# Code Quality Agent

## Task
Run comprehensive code quality checks on the codebase.

## Tools Available
- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **Mypy**: Static type checking
- **Flake8**: Code style checking
- **ESLint**: TypeScript/JavaScript linting

## Instructions

1. **If no arguments**: Run all quality checks

   **Python Backend:**
   ```bash
   cd /home/user/Vig_Project_personel
   python -m black --check src/
   python -m ruff check src/
   python -m mypy src/ --ignore-missing-imports
   ```

   **Frontend:**
   ```bash
   cd /home/user/Vig_Project_personel/frontend
   npm run lint
   ```

2. **If "fix" argument**: Auto-fix issues
   ```bash
   cd /home/user/Vig_Project_personel
   python -m black src/
   python -m ruff check src/ --fix
   ```

3. **If "type" argument**: Focus on type checking
   ```bash
   cd /home/user/Vig_Project_personel
   python -m mypy src/ --ignore-missing-imports --show-error-codes
   ```

4. **If "backend" argument**: Python checks only

5. **If "frontend" argument**: TypeScript/React checks only

## Known Type Issues
- `scheduler.py:25` - `any` should be `Any`
- Multiple `Dict[str, any]` occurrences need fixing

## Quality Standards
- No unused imports
- Proper type hints on all public functions
- Docstrings for modules and classes
- Maximum line length: 88 (Black default)

Arguments: $ARGUMENTS
