---
name: Architecture Analyzer
description: Analyze and document system architecture
tags: architecture, analysis, documentation
allowed-tools: Read, Grep, Glob, Task
---

# Architecture Analysis Agent

## Task
Analyze the system architecture and provide detailed documentation.

## System Overview
Digital Twin Factory - AI-Driven Manufacturing Intelligence System

## Core Components

### Backend (Python)
- **API Gateway**: FastAPI (main_integrated.py)
- **Vision AI**: Swin Transformer + ONNX + Grad-CAM
- **Digital Twin**: SimPy discrete event simulation
- **Scheduling**: OR-Tools CP-SAT + Ray RLlib
- **Predictive**: XGBoost + LSTM

### Frontend (React)
- **3D Visualization**: Three.js + React Three Fiber
- **Dashboard**: Recharts + Tailwind CSS
- **Real-time**: WebSocket

### Infrastructure
- **Database**: PostgreSQL + TimescaleDB
- **Cache**: Redis
- **Queue**: RabbitMQ
- **Storage**: MinIO
- **Orchestration**: Kubernetes

## Instructions

1. **If no arguments**: Full architecture overview
   - Generate component diagram
   - List all integrations
   - Identify architectural patterns
   - Note design decisions

2. **If "backend" argument**: Backend architecture focus
   - API structure
   - Module dependencies
   - Data flow

3. **If "frontend" argument**: Frontend architecture focus
   - Component hierarchy
   - State management
   - API integration

4. **If "data" argument**: Data architecture focus
   - Database schema
   - Event flow
   - Message queues

5. **If "deploy" argument**: Deployment architecture
   - Container setup
   - Kubernetes config
   - CI/CD pipeline

6. **If "deps" argument**: Dependency analysis
   - Module dependencies
   - Circular dependency check
   - Version compatibility

## Key Files
- ARCHITECTURE.md
- PROJECT_STRUCTURE.md
- API_DOCUMENTATION.md
- DEPLOYMENT.md

Arguments: $ARGUMENTS
