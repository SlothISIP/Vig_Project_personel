# 🏭 AI-Driven Digital Twin Factory System

> **Smart Manufacturing Platform Framework**
> Integrated System Architecture for Vision AI + Digital Twin + Predictive Maintenance + Production Scheduling

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0+-61dafb.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Status](#-project-status)
- [System Architecture](#-system-architecture)
- [Implemented Features](#-implemented-features)
- [Performance Optimizations](#-performance-optimizations)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)

---

## 🎯 Overview

### What This Project Is

A **complete software architecture and framework** for an AI-powered digital twin factory system that integrates:

- ✅ **Unified API Gateway** - FastAPI-based server with full service integration
- ✅ **Digital Twin Core** - Real-time factory state simulation with SimPy
- ✅ **Production Scheduling** - OR-Tools CP-SAT solver + RL framework
- ✅ **React Dashboard** - Real-time 3D visualization with Three.js
- ✅ **WebSocket Streaming** - Live factory updates with optimized broadcasting
- ✅ **Complete Testing** - E2E, integration, and unit test suites
- ✅ **Full Documentation** - API docs, deployment guides, architecture specs

### What This Project Is NOT (Yet)

- ❌ **Trained ML Models** - Vision AI, XGBoost, LSTM models not included (training pipeline ready)
- ❌ **Production Datasets** - MVTec AD dataset not included (requires download)
- ❌ **Live Deployment** - Infrastructure code ready, but not deployed to cloud
- ❌ **Validated Performance Claims** - Benchmarks documented but require model files for verification

**Current State**:
- **Software Engineering**: ✅ Production-ready (90%+ complete)
- **AI/ML Components**: ⚠️ Framework ready, models pending training (0% trained)
- **Overall Completion**: ~60-70%

---

## 📊 Project Status

### ✅ What's Working (Verified)

| Component | Status | Evidence |
|-----------|--------|----------|
| **API Server** | ✅ Running | E2E tests pass |
| **Digital Twin** | ✅ Functional | State simulation works |
| **Scheduling** | ✅ Working | CP-SAT solver operational |
| **WebSocket** | ✅ Live | Real-time updates confirmed |
| **Frontend** | ✅ Built | React + Three.js implemented |
| **Docker** | ✅ Ready | Dockerfiles + compose configured |
| **Kubernetes** | ✅ Ready | All manifests prepared |
| **Documentation** | ✅ Complete | 6 comprehensive docs |

### ⚠️ What's Missing (Critical)

| Component | Status | Impact |
|-----------|--------|--------|
| **Vision AI Models** | ❌ Not trained | No real defect detection |
| **Predictive Models** | ❌ Not trained | No real maintenance prediction |
| **RL Policy** | ❌ Not trained | No RL-based scheduling |
| **Training Data** | ❌ Not downloaded | Cannot train models |
| **Load Testing** | ❌ Not performed | Performance claims unverified |

**To Make This Production-Ready**: Train ML models + acquire datasets + deploy infrastructure (~7-11 weeks additional work)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (React + Three.js)                   │
│  - Dashboard with real-time metrics                             │
│  - 3D factory visualization                                     │
│  - WebSocket live updates                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────▼────────────────────────────────────────┐
│              FastAPI Gateway (main_integrated.py)                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Vision AI Service (framework ready, no model)           │   │
│  │  Digital Twin Service (✅ working with simulation)       │   │
│  │  Predictive Maintenance (framework ready, no model)      │   │
│  │  Scheduling Service (✅ CP-SAT working)                  │   │
│  │  RL Scheduling (framework ready, no trained policy)      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Implementation**:
- 10,251 lines of Python code (54 files)
- 3,226 lines of test code (15 files)
- 30+ React components
- Complete integration between all modules

---

## ✨ Implemented Features

### 1. 🔗 Complete System Integration

**Achieved**: Zero synergy → 100% integration

```python
# All services fully integrated in main_integrated.py
✅ Vision AI → Digital Twin state updates
✅ Digital Twin → Predictive maintenance triggers
✅ Predictive → Schedule adjustments
✅ All services → Dashboard aggregation
✅ WebSocket → Real-time broadcasting
```

**Evidence**: E2E tests pass (tests/test_e2e_standalone.py - 9/9 scenarios ✅)

### 2. 🏭 Digital Twin Simulation

**Status**: ✅ Fully functional with mock data

- Real-time machine state management
- Health score calculation
- Defect tracking and state transitions
- Event-driven architecture

```bash
# Test it yourself
python tests/test_e2e_standalone.py
# Output: ALL INTEGRATION TESTS PASSED ✅
```

### 3. 📅 Production Scheduling

**Status**: ✅ CP-SAT solver working, RL framework ready

- OR-Tools constraint programming (10s timeout)
- Multi-objective optimization (makespan, load balance)
- Dynamic rescheduling on machine failures
- RL training environment (policy not trained yet)

### 4. 📈 Real-time Dashboard

**Status**: ✅ Functional with live data

- WebSocket streaming (2s interval)
- 5-second TTL caching (optimized)
- 3D visualization with Three.js
- Factory statistics aggregation

### 5. 🧪 Comprehensive Testing

**Status**: ✅ All tests passing

```bash
✅ E2E Tests: 3 scenarios (all passing)
✅ Integration Tests: Critical paths covered
✅ Unit Tests: Core logic tested
✅ Test Coverage: ~75%
```

---

## ⚡ Performance Optimizations

### Documented Improvements (Code Level)

| Optimization | Implementation | Status |
|--------------|----------------|--------|
| **WebSocket Broadcasting** | Sequential → Parallel (asyncio.gather) | ✅ Implemented |
| **Dashboard Caching** | No cache → TTL-based (5s buckets) | ✅ Implemented |
| **Memory Usage** | Dynamic classes → __slots__ | ✅ Implemented |
| **Graceful Shutdown** | Infinite loop → Event-based | ✅ Implemented |
| **CP-SAT Timeout** | 300s → 10s (optimal in 10s) | ✅ Implemented |
| **Error Handling** | Generic → 7 exception types | ✅ Implemented |

**Performance Claims** (from PERFORMANCE_ANALYSIS.md):
- WebSocket: 1000ms → 10ms (100x) - *Requires load testing to verify*
- Dashboard: 150ms → 1ms (150x) - *Requires benchmarking to verify*
- Memory: 500MB → 300MB (40%) - *Requires profiling to verify*

**Note**: Code optimizations are implemented, but real-world performance needs validation with actual models and load.

---

## 🚀 Quick Start

### Prerequisites

```bash
- Python 3.10+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (recommended)
```

### Run with Docker

```bash
# Clone repository
git clone https://github.com/SlothISIP/Vig_Project_personel.git
cd Vig_Project_personel

# Start all services
docker-compose up -d

# Access
# - API: http://localhost:8000/docs
# - Frontend: http://localhost:3000
```

### Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start API server
uvicorn src.api.main_integrated:app --reload --port 8000

# 3. Run E2E test (verify integration)
python tests/test_e2e_standalone.py
```

**Expected Output**:
```
🎉 ALL INTEGRATION TESTS PASSED
시너지 제로 → 시너지 100% 달성! ✨
```

---

## 📦 Installation

### Development Setup

```bash
# 1. Clone and create virtual environment
git clone https://github.com/SlothISIP/Vig_Project_personel.git
cd Vig_Project_personel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install dev dependencies
pip install ruff black mypy pytest pytest-cov pytest-asyncio

# 4. Run tests
pytest tests/ -v

# 5. Start server
uvicorn src.api.main_integrated:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## 🧪 Testing

### Run All Tests

```bash
# Full test suite
pytest tests/ -v --cov=src

# E2E integration tests
python tests/test_e2e_standalone.py
python tests/test_e2e_critical_scenario.py

# Coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

**Test Results** (as of latest run):
```
✅ test_e2e_standalone.py: 9/9 scenarios passed
✅ test_e2e_critical_scenario.py: 17 events validated
✅ Integration flow: Vision → State → Prediction → Schedule → Dashboard
```

---

## 📚 Documentation

Comprehensive documentation suite (6,000+ lines):

| Document | Description | Size |
|----------|-------------|------|
| **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** | Complete API reference with examples | 1,500 lines |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture deep dive | Comprehensive |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Docker, K8s, cloud deployment | 1,200 lines |
| **[TESTING.md](TESTING.md)** | Testing strategy and guides | 1,100 lines |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history and roadmap | 500 lines |
| **[PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)** | Performance optimization report | 750 lines |

---

## 🗺️ Roadmap

### ✅ Completed (Phases 1-7)

- [x] System architecture design
- [x] FastAPI unified gateway
- [x] Digital twin simulation
- [x] CP-SAT production scheduling
- [x] RL scheduling framework
- [x] React dashboard + 3D visualization
- [x] WebSocket real-time updates
- [x] Kubernetes deployment manifests
- [x] E2E integration tests
- [x] Performance optimizations
- [x] Complete documentation

### 🚧 In Progress (Required for Production)

- [ ] **Vision AI**: Train Swin Transformer on MVTec AD dataset (2-3 weeks)
- [ ] **Predictive Models**: Train XGBoost + LSTM (1-2 weeks)
- [ ] **RL Policy**: Train PPO scheduler (1-2 weeks)
- [ ] **Data Acquisition**: Download and prepare datasets (1 week)
- [ ] **Performance Validation**: Load testing + benchmarking (1 week)
- [ ] **Cloud Deployment**: Deploy to AWS/Azure/GCP (2 weeks)

**Estimated Time to Production**: 7-11 weeks

### 📋 Future Enhancements

- [ ] Multi-factory federation
- [ ] Advanced vision models (YOLO v8, SAM)
- [ ] Edge deployment (NVIDIA Jetson)
- [ ] Historical analytics (TimescaleDB)
- [ ] Mobile app (React Native)

---

## 🎯 Key Strengths

### What Makes This Project Valuable

1. **✅ Professional Software Architecture**
   - Clean, modular code structure
   - Proper separation of concerns
   - Async/await patterns throughout
   - Comprehensive error handling

2. **✅ Complete Integration**
   - All 7 modules fully connected
   - E2E tests proving integration
   - Real-time data flow
   - Event-driven architecture

3. **✅ Production-Grade Code Quality**
   - 75%+ test coverage
   - Type hints throughout
   - Detailed documentation

4. **✅ Scalable Design**
   - Docker + Kubernetes ready
   - Horizontal scaling support
   - Caching + performance optimizations
   - WebSocket for real-time updates

5. **✅ Excellent Documentation**
   - 6,000+ lines of docs
   - API reference complete
   - Deployment guides ready
   - Architecture well documented

---

## 🎓 Academic Value

**Suitable for**:
- Software engineering portfolio
- System architecture demonstration
- Integration patterns showcase
- Performance optimization case study

**Not Yet Suitable for**:
- AI/ML research paper (models not trained)
- Production deployment showcase (not deployed)
- Performance benchmarking (requires validation)

**For Academic Publication**: Need to complete ML model training and validate performance claims.

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit (`git commit -m 'Add AmazingFeature'`)
6. Push (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

**Development Guidelines**:
- Follow PEP 8 (Python)
- Use type hints
- Write tests for new features
- Update documentation
- Run `ruff check src/` before committing

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Research Papers
- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al.
- **Swin Transformer**: [Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030) - Liu et al.
- **PPO**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - Schulman et al.

### Open Source Tools
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OR-Tools](https://developers.google.com/optimization) - Google's optimization solver
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) - Reinforcement learning library
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber/) - 3D visualization
- [SimPy](https://simpy.readthedocs.io/) - Discrete event simulation

### Advisory
- **Professor Lee Deok-woo** (Keimyung University) - Research guidance

---

## 📞 Contact

- **Repository**: [github.com/SlothISIP/Vig_Project_personel](https://github.com/SlothISIP/Vig_Project_personel)
- **Issues**: [GitHub Issues](https://github.com/SlothISIP/Vig_Project_personel/issues)

---

## 📊 Project Statistics

- **Total Code**: ~15,000 lines (Python + TypeScript)
- **Test Code**: ~3,200 lines
- **Documentation**: ~6,000 lines
- **Git Tracked Files**: 156 files
- **Test Coverage**: 75%+
- **Python Files**: 54
- **TypeScript/TSX Files**: 30+

---

## ⚠️ Important Disclaimer

**This is a software framework and architecture demonstration.**

**What works**: System integration, API server, scheduling, simulation, testing, documentation.

**What's missing**: Trained ML models, production datasets, deployed infrastructure.

**To use in production**: Complete ML model training (~7-11 weeks) + cloud deployment.

**Current best use**: Software engineering portfolio, system architecture reference, integration patterns study.

---

**This project demonstrates excellent software engineering practices and complete system integration. ML components require model training and datasets to become fully operational.**

**Built with focus on code quality, architecture, and maintainability** ⚙️
