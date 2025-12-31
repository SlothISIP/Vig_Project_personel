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

## Project Completion Plan

**Reference Document:** `COMPLETION_PLAN.md`

### Current Status
- **Phase:** Phase 1 & 2 Complete, Phase 3 Next
- **Completion Level:** ~45% toward production
- **Estimated Remaining:** 1-2 weeks

### Phase Overview
| Phase | Description | Duration | Status |
|-------|-------------|----------|--------|
| **Phase 1** | Execution Readiness | 2 days | [x] Complete |
| **Phase 2** | Critical Issues Fix | 3 days | [x] Complete |
| **Phase 3** | Deployment Preparation | 3 days | [ ] Not Started |
| **Phase 4** | Test Coverage | 5 days | [ ] Not Started |

### Quick Start (Phase 1)
```bash
# 1. Create directories
mkdir -p data/{raw,processed} models/{onnx,checkpoints} logs config

# 2. Configure environment
cp .env.example .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create ONNX model
python -c "
from src.vision.models.swin_transformer import create_swin_tiny
import torch
from pathlib import Path
Path('models/onnx').mkdir(parents=True, exist_ok=True)
m = create_swin_tiny(num_classes=2, pretrained=True)
m.eval()
torch.onnx.export(m, torch.randn(1,3,224,224), 'models/onnx/swin_defect.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
"

# 5. Verify and start
python scripts/test_setup.py
uvicorn src.api.main_integrated:app --reload --port 8000
```

### Next Steps
1. Complete Phase 1 to make project runnable
2. Fix Critical issues (Phase 2) before any new features
3. See `COMPLETION_PLAN.md` for detailed task breakdown

### When Continuing Work
1. Check current phase status in this section
2. Read `COMPLETION_PLAN.md` for detailed tasks
3. Use `/fix-issue` command for known issues
4. Update status after completing each phase
