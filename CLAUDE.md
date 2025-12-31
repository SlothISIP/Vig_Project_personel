# Claude Code Guidelines

## Code Style Rules

### No Emojis in Code and Comments
- Never use emojis in source code files
- Never use emojis in code comments
- Keep code clean and professional without decorative characters

## Context Management

### Compact Command Usage
- When context is running low, run `/compact` to compact and continue
- If `/compact` fails with "conversation too long" error:
  - Do NOT panic or stop working
  - Continue with the current task using available context
  - Break down remaining work into smaller, focused steps
  - Prioritize completing critical operations first

## Thinking and Reasoning Standards

### Critical Thinking Requirements
Always avoid exaggeration; think and answer realistically and objectively. Always think step by step and break down the logic behind the answer. If the context is unclear or you do not know the answer, explicitly state that and ask clarifying questions instead of guessing or making up information. Furthermore, adopt a critical perspective: critique my assumptions and point out potential risks, and also critique your own answer for potential biases or logical fallacies before responding.
