# 🚀 Deployment Guide

> **Complete deployment instructions for all environments**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pre-deployment Checklist](#pre-deployment-checklist)
- [Environment Configuration](#environment-configuration)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platforms](#cloud-platforms)
- [CI/CD Setup](#cicd-setup)
- [Monitoring Setup](#monitoring-setup)
- [Troubleshooting](#troubleshooting)
- [Rollback Procedures](#rollback-procedures)

---

## Overview

The AI-Driven Digital Twin Factory System can be deployed in multiple ways:

1. **Local Development**: Direct Python/Node.js execution
2. **Docker Compose**: Single-server containerized deployment
3. **Kubernetes**: Production-grade orchestration
4. **Cloud Platforms**: AWS, Azure, GCP managed services

**Recommended Production Stack**:
- **Small-Medium**: Docker Compose on single server
- **Large-Scale**: Kubernetes cluster
- **Serverless**: AWS Lambda + ECS Fargate (future)

---

## Pre-deployment Checklist

### Required Resources

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| **CPU** | 2 cores | 4 cores | 8+ cores |
| **RAM** | 4 GB | 8 GB | 16+ GB |
| **Storage** | 20 GB | 50 GB | 100+ GB |
| **Network** | 10 Mbps | 100 Mbps | 1 Gbps |
| **GPU** | None | NVIDIA T4 | A100/V100 |

### Software Requirements

```bash
# Required
- Python 3.10+
- Node.js 18+
- Docker 24+
- Docker Compose 2.20+

# Optional (for production)
- Kubernetes 1.28+
- kubectl 1.28+
- Helm 3.12+
```

### Checklist

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Code linted and formatted (`ruff check src/`, `black src/`)
- [ ] Type checking passed (`mypy src/`)
- [ ] Environment variables configured
- [ ] SSL certificates obtained (production)
- [ ] Database backups configured
- [ ] Monitoring set up
- [ ] Documentation reviewed
- [ ] Security scan completed
- [ ] Load testing performed

---

## Environment Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# ======================
# Environment
# ======================
ENVIRONMENT=development  # development | staging | production
DEBUG=true              # false in production
LOG_LEVEL=INFO          # DEBUG | INFO | WARNING | ERROR | CRITICAL

# ======================
# API Configuration
# ======================
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4           # Number of Uvicorn workers
API_RELOAD=true         # Hot reload (development only)

# ======================
# CORS Configuration
# ======================
CORS_ORIGINS=["http://localhost:3000"]  # Frontend URLs
CORS_ALLOW_CREDENTIALS=true

# ======================
# Model Paths
# ======================
VISION_MODEL_PATH=./models/swin_tiny_fp16.onnx
PREDICTIVE_MODEL_PATH=./models/xgboost_maintenance.pkl
RL_MODEL_PATH=./models/ppo_scheduler.pkl

# ======================
# Performance Tuning
# ======================
CACHE_TTL_SECONDS=5
WEBSOCKET_BROADCAST_INTERVAL=2
SCHEDULER_TIMEOUT_SECONDS=10
MAX_UPLOAD_SIZE_MB=10

# ======================
# Database (Optional)
# ======================
DATABASE_URL=postgresql://user:pass@localhost:5432/digital_twin
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# ======================
# Redis (Optional)
# ======================
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# ======================
# Security
# ======================
SECRET_KEY=your-secret-key-here-change-in-production
API_KEY=your-api-key-here-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=15

# ======================
# Monitoring (Optional)
# ======================
PROMETHEUS_ENABLED=false
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=false
GRAFANA_PORT=3001
```

### Production Environment Variables

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
API_RELOAD=false

# Strong secrets (use secrets manager in production)
SECRET_KEY=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 32)

# Performance optimization
API_WORKERS=8
CACHE_TTL_SECONDS=10

# Database with connection pooling
DATABASE_URL=postgresql://user:strong_password@db-prod:5432/digital_twin
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis cluster
REDIS_URL=redis://redis-cluster:6379/0

# Monitoring enabled
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

---

## Local Development

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/Vig_Project_personel.git
cd Vig_Project_personel

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development dependencies
pip install ruff black mypy pytest pytest-cov pytest-asyncio

# 5. Create .env file
cp .env.example .env
# Edit .env with your local settings

# 6. Run tests
pytest tests/ -v

# 7. Start API server
uvicorn src.api.main_integrated:app --reload --host 0.0.0.0 --port 8000

# 8. Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Development Workflow

```bash
# Run tests on file changes
pytest-watch tests/

# Auto-format code
black src/

# Lint code
ruff check src/ --fix

# Type check
mypy src/

# Generate coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Docker Deployment

### Docker Compose (Recommended for Small Deployments)

#### Basic Setup

```bash
# 1. Build images
docker-compose build

# 2. Start services
docker-compose up -d

# 3. View logs
docker-compose logs -f api

# 4. Check status
docker-compose ps

# 5. Stop services
docker-compose down
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  # Backend API
  api:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/digital_twin
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - api
    restart: unless-stopped

  # PostgreSQL Database
  db:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=digital_twin
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Prometheus (Optional)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

  # Grafana (Optional)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

#### Dockerfile.backend

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main_integrated:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Production Deployment

```bash
# 1. Build production images
docker-compose -f docker-compose.prod.yml build

# 2. Push to registry
docker tag digital-twin-api:latest ghcr.io/yourusername/digital-twin-api:latest
docker push ghcr.io/yourusername/digital-twin-api:latest

# 3. Deploy to production server
ssh user@production-server
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# 4. Verify deployment
docker-compose -f docker-compose.prod.yml ps
curl http://production-server:8000/health
```

---

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm (optional)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installation
kubectl version --client
helm version
```

### Deployment Steps

```bash
# 1. Create namespace
kubectl create namespace digital-twin-factory

# 2. Create ConfigMap
kubectl apply -f deployment/k8s/configmap.yaml

# 3. Create Secrets
kubectl create secret generic api-secrets \
  --from-literal=secret-key=$(openssl rand -hex 32) \
  --from-literal=api-key=$(openssl rand -hex 32) \
  --namespace=digital-twin-factory

# 4. Create Persistent Volume Claims
kubectl apply -f deployment/k8s/pvc.yaml

# 5. Deploy backend
kubectl apply -f deployment/k8s/backend-deployment.yaml
kubectl apply -f deployment/k8s/backend-service.yaml

# 6. Deploy frontend
kubectl apply -f deployment/k8s/frontend-deployment.yaml
kubectl apply -f deployment/k8s/frontend-service.yaml

# 7. Set up Ingress
kubectl apply -f deployment/k8s/ingress.yaml

# 8. Enable autoscaling
kubectl apply -f deployment/k8s/hpa.yaml

# 9. Verify deployment
kubectl get pods -n digital-twin-factory
kubectl get services -n digital-twin-factory
kubectl get ingress -n digital-twin-factory
```

### Kubernetes Manifests

#### backend-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: digital-twin-factory
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: api
        image: ghcr.io/yourusername/digital-twin-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: secret-key
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

#### hpa.yaml (Horizontal Pod Autoscaler)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: digital-twin-factory
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### ingress.yaml

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: digital-twin-ingress
  namespace: digital-twin-factory
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    - app.yourdomain.com
    secretName: digital-twin-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8000
  - host: app.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
```

### Rolling Updates

```bash
# Update backend image
kubectl set image deployment/backend \
  api=ghcr.io/yourusername/digital-twin-api:v1.2.0 \
  -n digital-twin-factory

# Monitor rollout
kubectl rollout status deployment/backend -n digital-twin-factory

# Rollback if needed
kubectl rollout undo deployment/backend -n digital-twin-factory

# View rollout history
kubectl rollout history deployment/backend -n digital-twin-factory
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment backend --replicas=5 -n digital-twin-factory

# View HPA status
kubectl get hpa -n digital-twin-factory
kubectl describe hpa backend-hpa -n digital-twin-factory
```

---

## Cloud Platforms

### AWS Deployment

#### ECS Fargate

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name digital-twin-api

# 2. Build and push image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag digital-twin-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/digital-twin-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/digital-twin-api:latest

# 3. Create ECS cluster
aws ecs create-cluster --cluster-name digital-twin-cluster

# 4. Create task definition (see task-definition.json)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 5. Create service
aws ecs create-service \
  --cluster digital-twin-cluster \
  --service-name backend-service \
  --task-definition backend-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

#### AWS RDS for PostgreSQL

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier digital-twin-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --master-username admin \
  --master-user-password <strong-password> \
  --allocated-storage 100 \
  --vpc-security-group-ids sg-xxx \
  --db-subnet-group-name default
```

### Azure Deployment

#### Azure Container Instances

```bash
# 1. Create resource group
az group create --name digital-twin-rg --location eastus

# 2. Create ACR
az acr create --resource-group digital-twin-rg --name digitaltwinacr --sku Basic

# 3. Build and push image
az acr build --registry digitaltwinacr --image digital-twin-api:latest .

# 4. Deploy container
az container create \
  --resource-group digital-twin-rg \
  --name backend \
  --image digitaltwinacr.azurecr.io/digital-twin-api:latest \
  --cpu 2 \
  --memory 4 \
  --port 8000 \
  --environment-variables ENVIRONMENT=production
```

### GCP Deployment

#### Cloud Run

```bash
# 1. Build image
gcloud builds submit --tag gcr.io/PROJECT_ID/digital-twin-api

# 2. Deploy to Cloud Run
gcloud run deploy backend \
  --image gcr.io/PROJECT_ID/digital-twin-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10
```

---

## CI/CD Setup

### GitHub Actions

Already configured in `.github/workflows/ci-cd.yaml`

**Workflow**:
1. **Test**: Run unit & integration tests
2. **Build**: Build Docker images
3. **Push**: Push to container registry
4. **Deploy**: Deploy to Kubernetes/cloud

**Triggering Deployment**:
```bash
# Deploy to production (create tag)
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Deploy to staging (push to develop)
git push origin develop
```

### Manual Deployment

```bash
# 1. Build production images locally
docker-compose -f docker-compose.prod.yml build

# 2. Tag images
docker tag digital-twin-api:latest ghcr.io/yourusername/digital-twin-api:v1.0.0

# 3. Push to registry
docker push ghcr.io/yourusername/digital-twin-api:v1.0.0

# 4. Deploy to Kubernetes
kubectl set image deployment/backend \
  api=ghcr.io/yourusername/digital-twin-api:v1.0.0 \
  -n digital-twin-factory

# 5. Verify deployment
kubectl rollout status deployment/backend -n digital-twin-factory
```

---

## Monitoring Setup

### Prometheus + Grafana

```bash
# 1. Install Prometheus Operator (using Helm)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# 2. Access Grafana
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
# Open http://localhost:3000 (admin/prom-operator)

# 3. Import dashboards
# Dashboard ID: 1860 (Node Exporter Full)
# Dashboard ID: 6417 (Kubernetes Cluster Monitoring)
# Custom dashboard: deployment/grafana-dashboard.json
```

### Application Metrics

```python
# Add Prometheus metrics to FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Metrics available at /metrics
```

---

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker-compose logs api

# Common causes:
# - Missing environment variables
# - Port conflicts
# - File permissions

# Fix: Check .env file, stop conflicting services, fix permissions
```

#### 2. Database Connection Failed

```bash
# Test connection
docker exec -it digital-twin-db psql -U postgres -d digital_twin

# Common causes:
# - Database not ready
# - Wrong credentials
# - Network issues

# Fix: Add healthcheck, verify DATABASE_URL, check network
```

#### 3. High Memory Usage

```bash
# Check container stats
docker stats

# Common causes:
# - Too many workers
# - Memory leaks
# - Large model files loaded multiple times

# Fix: Reduce API_WORKERS, implement model caching, monitor with prometheus
```

#### 4. WebSocket Disconnects

```bash
# Check timeout settings
# - Load balancer timeout
# - Nginx proxy_read_timeout
# - Application keep-alive

# Fix: Increase timeouts, implement reconnection logic
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with debugger
python -m pdb src/api/main_integrated.py

# Profile performance
python -m cProfile -o profile.stats src/api/main_integrated.py
```

---

## Rollback Procedures

### Docker Compose Rollback

```bash
# 1. Stop current version
docker-compose down

# 2. Pull previous version
docker-compose pull
# OR: Manually tag previous version as latest

# 3. Start previous version
docker-compose up -d

# 4. Verify
curl http://localhost:8000/health
```

### Kubernetes Rollback

```bash
# Automatic rollback to previous revision
kubectl rollout undo deployment/backend -n digital-twin-factory

# Rollback to specific revision
kubectl rollout history deployment/backend -n digital-twin-factory
kubectl rollout undo deployment/backend --to-revision=3 -n digital-twin-factory

# Verify rollback
kubectl rollout status deployment/backend -n digital-twin-factory
kubectl get pods -n digital-twin-factory
```

### Database Rollback

```bash
# 1. Stop application
kubectl scale deployment backend --replicas=0 -n digital-twin-factory

# 2. Restore from backup
pg_restore -U postgres -d digital_twin < backup_20251119.sql

# 3. Restart application
kubectl scale deployment backend --replicas=2 -n digital-twin-factory
```

---

## Security Checklist

- [ ] HTTPS/TLS enabled with valid certificates
- [ ] Strong secrets generated (not default values)
- [ ] Secrets stored in secrets manager (not in code)
- [ ] CORS properly configured
- [ ] Rate limiting enabled
- [ ] Authentication/authorization implemented
- [ ] Docker images scanned for vulnerabilities
- [ ] Network policies configured (Kubernetes)
- [ ] Database backups automated
- [ ] Monitoring and alerting set up
- [ ] Firewall rules configured
- [ ] Log aggregation enabled

---

## Performance Optimization

```bash
# 1. Enable caching
CACHE_TTL_SECONDS=10

# 2. Increase workers
API_WORKERS=8  # 2× CPU cores

# 3. Use connection pooling
DATABASE_POOL_SIZE=20

# 4. Enable compression
# (Already enabled in FastAPI)

# 5. Use CDN for static assets
# Upload frontend build to S3 + CloudFront

# 6. Optimize database
# Add indexes, analyze queries, enable query caching
```

---

## Maintenance

### Regular Tasks

```bash
# Daily
- Check logs for errors
- Monitor resource usage
- Verify backups

# Weekly
- Review security scans
- Update dependencies (security patches)
- Clean up old images/containers

# Monthly
- Performance review
- Capacity planning
- Disaster recovery drill
```

### Backup Strategy

```bash
# Database backup (daily)
pg_dump -U postgres digital_twin > backup_$(date +%Y%m%d).sql

# Model files backup (weekly)
tar -czf models_$(date +%Y%m%d).tar.gz models/

# Configuration backup (on change)
git commit -am "Update configuration"
git push origin main
```

---

## Support

**Documentation**: [https://docs.yourdomain.com](https://docs.yourdomain.com)
**Issues**: [https://github.com/yourusername/Vig_Project_personel/issues](https://github.com/yourusername/Vig_Project_personel/issues)
**Email**: support@yourdomain.com

---

**Last Updated**: 2025-11-19
