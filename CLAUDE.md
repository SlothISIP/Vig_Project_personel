# Claude Code Guidelines

## Code Style Rules

### No Emojis in Code and Comments
- Never use emojis in source code files
- Never use emojis in code comments
- Keep code clean and professional without decorative characters

### No Mock Usage
- Never use mock objects, MagicMock, or patch decorators in tests
- Always use real implementations or proper test fixtures
- Create actual test data instead of mocked responses
- If external services are needed, use test containers or in-memory alternatives
- Integration tests must use real component interactions

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

## Project-Specific Slash Commands

### Core Testing & Analysis
| Command | Description |
|---------|-------------|
| `/dt-simulate` | Run and analyze Digital Twin factory simulation |
| `/rl-debug` | Debug RL scheduling components (env, reward, training) |
| `/vision-test` | Test Vision AI defect detection system |
| `/integration-check` | Verify integration between all components |

### Quality & Performance
| Command | Description |
|---------|-------------|
| `/code-quality` | Run linting, type checking, formatting |
| `/perf-analyze` | Analyze and benchmark system performance |
| `/run-tests` | Run test suites (unit, integration, e2e) |

### Maintenance & Architecture
| Command | Description |
|---------|-------------|
| `/fix-issue` | Fix known issues by number (1-8) or priority |
| `/api-test` | Test API endpoints and WebSocket |
| `/analyze-architecture` | Analyze system architecture |

### Usage Examples
```
/dt-simulate run          # Run simulation demo
/rl-debug env             # Focus on environment issues
/vision-test api          # Test vision API endpoints
/run-tests unit           # Run unit tests only
/fix-issue critical       # Fix all critical issues
/code-quality fix         # Auto-fix code issues
```

## Claude Skills

### Domain Experts
| Skill | Expertise |
|-------|-----------|
| `digital-twin-expert` | SimPy simulation, production lines, machine states |
| `rl-scheduling-expert` | Ray RLlib, Gymnasium, reward functions, policies |
| `vision-ai-expert` | Swin Transformer, ONNX, Grad-CAM, defect detection |
| `integration-specialist` | Feedback loops, unified pipeline, cross-component integration |

## Project Structure Quick Reference

### Backend Core Paths
- `/src/api/main_integrated.py` - API Gateway (1,155 lines)
- `/src/digital_twin/` - Factory simulation
- `/src/rl_scheduling/` - RL scheduling
- `/src/vision/` - Defect detection
- `/src/integration/` - Unified pipeline

### Frontend Core Paths
- `/frontend/src/App.tsx` - React app root
- `/frontend/src/components/DigitalTwin3D/` - 3D visualization

### Known Issues (Priority Order)
1. **CRITICAL**: State sync missing (feedback_loop + digital_twin_env)
2. **CRITICAL**: External lines validation missing
3. **HIGH**: Service initialization order
4. **HIGH**: Job status error handling
5. **HIGH**: WebSocket race condition
