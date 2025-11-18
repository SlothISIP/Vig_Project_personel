# Digital Twin Factory - 배포 가이드

Digital Twin Factory 시스템을 Kubernetes 클러스터에 배포하는 방법을 설명합니다.

## 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [로컬 개발 환경](#로컬-개발-환경)
3. [Docker 이미지 빌드](#docker-이미지-빌드)
4. [Kubernetes 배포](#kubernetes-배포)
5. [Helm Chart 사용](#helm-chart-사용)
6. [모니터링 설정](#모니터링-설정)
7. [CI/CD 파이프라인](#cicd-파이프라인)
8. [트러블슈팅](#트러블슈팅)

## 시스템 요구사항

### 클러스터 사양

**최소 사양:**
- Kubernetes 1.25+
- 3 노드 (Worker Nodes)
- 각 노드: 4 vCPU, 16GB RAM
- 총 스토리지: 100GB+

**권장 사양:**
- Kubernetes 1.27+
- 5+ 노드 (Worker Nodes)
- 각 노드: 8 vCPU, 32GB RAM
- 총 스토리지: 500GB+
- GPU 노드 (RL 학습용, 선택사항)

### 필수 도구

```bash
# Kubernetes CLI
kubectl version --client

# Docker
docker --version

# Helm (선택사항)
helm version

# Git
git --version
```

## 로컬 개발 환경

Docker Compose를 사용한 로컬 개발 환경 구성:

### 1. 시작

```bash
# 전체 스택 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 특정 서비스만 시작
docker-compose up -d backend frontend
```

### 2. 접속

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Ray Dashboard**: http://localhost:8265 (RL 학습용)

### 3. 종료

```bash
# 전체 중지
docker-compose down

# 볼륨 포함 삭제
docker-compose down -v
```

## Docker 이미지 빌드

### 백엔드 이미지

```bash
# 빌드
docker build -f Dockerfile.backend -t digital-twin-factory/backend:latest .

# 태그 추가
docker tag digital-twin-factory/backend:latest \
  ghcr.io/your-org/digital-twin-factory/backend:v1.0.0

# 푸시
docker push ghcr.io/your-org/digital-twin-factory/backend:v1.0.0
```

### 프론트엔드 이미지

```bash
# 빌드
docker build -f Dockerfile.frontend -t digital-twin-factory/frontend:latest .

# 태그 및 푸시
docker tag digital-twin-factory/frontend:latest \
  ghcr.io/your-org/digital-twin-factory/frontend:v1.0.0

docker push ghcr.io/your-org/digital-twin-factory/frontend:v1.0.0
```

### Multi-platform 빌드

```bash
# buildx 설정
docker buildx create --use

# 멀티 플랫폼 빌드 및 푸시
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.backend \
  -t ghcr.io/your-org/digital-twin-factory/backend:latest \
  --push .
```

## Kubernetes 배포

### 1. Namespace 생성

```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. ConfigMap 및 Secret 설정

```bash
# ConfigMap 생성
kubectl apply -f k8s/configmap.yaml

# Secret 생성 (민감 정보)
kubectl create secret generic backend-secrets \
  --from-literal=db-password=YOUR_DB_PASSWORD \
  --from-literal=api-key=YOUR_API_KEY \
  -n digital-twin-factory
```

### 3. Storage 설정

```bash
# PersistentVolumeClaims 생성
kubectl apply -f k8s/pvc.yaml

# 확인
kubectl get pvc -n digital-twin-factory
```

### 4. 애플리케이션 배포

```bash
# 백엔드 배포
kubectl apply -f k8s/backend-deployment.yaml

# 프론트엔드 배포
kubectl apply -f k8s/frontend-deployment.yaml

# 확인
kubectl get pods -n digital-twin-factory
kubectl get services -n digital-twin-factory
```

### 5. Ingress 설정

```bash
# Ingress 생성
kubectl apply -f k8s/ingress.yaml

# 확인
kubectl get ingress -n digital-twin-factory
```

### 6. Auto-scaling 설정

```bash
# HorizontalPodAutoscaler 적용
kubectl apply -f k8s/hpa.yaml

# 확인
kubectl get hpa -n digital-twin-factory
```

### 7. 배포 확인

```bash
# Pod 상태
kubectl get pods -n digital-twin-factory

# 로그 확인
kubectl logs -f deployment/backend -n digital-twin-factory
kubectl logs -f deployment/frontend -n digital-twin-factory

# 서비스 접속 테스트
kubectl port-forward svc/backend-service 8000:8000 -n digital-twin-factory
curl http://localhost:8000/health
```

## 설정 업데이트

### ConfigMap 업데이트

```bash
# ConfigMap 수정
kubectl edit configmap backend-config -n digital-twin-factory

# 또는 파일에서 재적용
kubectl apply -f k8s/configmap.yaml

# Deployment 재시작 (변경사항 반영)
kubectl rollout restart deployment/backend -n digital-twin-factory
```

### 이미지 업데이트

```bash
# 새 이미지로 업데이트
kubectl set image deployment/backend \
  backend=ghcr.io/your-org/digital-twin-factory/backend:v1.1.0 \
  -n digital-twin-factory

# 롤아웃 상태 확인
kubectl rollout status deployment/backend -n digital-twin-factory

# 롤백 (필요시)
kubectl rollout undo deployment/backend -n digital-twin-factory
```

### 스케일 조정

```bash
# 수동 스케일링
kubectl scale deployment/backend --replicas=5 -n digital-twin-factory

# HPA 상태 확인
kubectl get hpa backend-hpa -n digital-twin-factory
```

## 모니터링 설정

### Prometheus & Grafana

```bash
# Prometheus Operator 설치
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# ServiceMonitor 생성 (백엔드 메트릭)
kubectl apply -f k8s/monitoring/servicemonitor.yaml

# Grafana 접속
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
# 기본 계정: admin / prom-operator
```

### 로깅 (EFK Stack)

```bash
# Elasticsearch + Fluentd + Kibana 설치
helm repo add elastic https://helm.elastic.co
helm repo update

helm install elasticsearch elastic/elasticsearch -n logging --create-namespace
helm install kibana elastic/kibana -n logging
helm install fluentd fluent/fluentd -n logging

# Kibana 접속
kubectl port-forward svc/kibana-kibana 5601:5601 -n logging
```

## CI/CD 파이프라인

### GitHub Actions 설정

1. **Repository Secrets 설정:**
   - Settings → Secrets → Actions
   - 추가할 Secrets:
     - `KUBECONFIG`: Kubernetes 설정 (base64 인코딩)
     - `DOCKER_USERNAME`: Docker/GitHub Registry 사용자명
     - `DOCKER_PASSWORD`: Docker/GitHub Registry 토큰

2. **KUBECONFIG 생성:**

```bash
# kubeconfig를 base64로 인코딩
cat ~/.kube/config | base64 | pbcopy

# GitHub Secrets에 추가
```

3. **워크플로우 트리거:**

```bash
# main 브랜치에 푸시
git push origin main

# 또는 태그 생성
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

### 수동 배포

```bash
# 1. 이미지 빌드 및 푸시
docker build -f Dockerfile.backend -t ghcr.io/your-org/dtf/backend:v1.0.0 .
docker push ghcr.io/your-org/dtf/backend:v1.0.0

# 2. Kubernetes 업데이트
kubectl set image deployment/backend \
  backend=ghcr.io/your-org/dtf/backend:v1.0.0 \
  -n digital-twin-factory

# 3. 배포 확인
kubectl rollout status deployment/backend -n digital-twin-factory
```

## 운영 가이드

### Health Check

```bash
# Backend health
curl https://digital-twin-factory.example.com/api/health

# Frontend health
curl https://digital-twin-factory.example.com/health
```

### 백업

```bash
# PVC 백업 (velero 사용)
velero backup create dtf-backup-$(date +%Y%m%d) \
  --include-namespaces digital-twin-factory

# 데이터베이스 백업
kubectl exec -n digital-twin-factory postgres-0 -- \
  pg_dump -U dtf_user digital_twin > backup.sql
```

### 복원

```bash
# Velero로 복원
velero restore create --from-backup dtf-backup-20240101

# DB 복원
kubectl exec -i -n digital-twin-factory postgres-0 -- \
  psql -U dtf_user digital_twin < backup.sql
```

## 트러블슈팅

### Pod가 시작되지 않음

```bash
# Pod 상태 확인
kubectl get pods -n digital-twin-factory

# 자세한 정보
kubectl describe pod <pod-name> -n digital-twin-factory

# 로그 확인
kubectl logs <pod-name> -n digital-twin-factory

# 이전 컨테이너 로그
kubectl logs <pod-name> -n digital-twin-factory --previous
```

### 이미지 Pull 실패

```bash
# ImagePullSecret 생성
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<token> \
  -n digital-twin-factory

# Deployment에 추가
kubectl patch serviceaccount backend-sa \
  -p '{"imagePullSecrets": [{"name": "ghcr-secret"}]}' \
  -n digital-twin-factory
```

### 메모리 부족 (OOMKilled)

```bash
# 리소스 사용량 확인
kubectl top pods -n digital-twin-factory

# Deployment에서 메모리 증가
kubectl set resources deployment/backend \
  --limits=memory=8Gi \
  --requests=memory=4Gi \
  -n digital-twin-factory
```

### 네트워크 연결 문제

```bash
# 서비스 확인
kubectl get svc -n digital-twin-factory

# DNS 테스트
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  nslookup backend-service.digital-twin-factory.svc.cluster.local

# 포트 연결 테스트
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  wget -O- http://backend-service.digital-twin-factory:8000/health
```

### 로그 레벨 변경

```bash
# ConfigMap 업데이트
kubectl patch configmap backend-config \
  -p '{"data":{"LOG_LEVEL":"DEBUG"}}' \
  -n digital-twin-factory

# Pod 재시작
kubectl rollout restart deployment/backend -n digital-twin-factory
```

## 보안 권장사항

1. **RBAC 설정**: 최소 권한 원칙 적용
2. **Network Policies**: Pod 간 통신 제한
3. **Secret 암호화**: Sealed Secrets 또는 External Secrets Operator 사용
4. **Image Scanning**: Trivy, Clair 등으로 취약점 스캔
5. **TLS 인증서**: cert-manager로 자동 갱신
6. **Pod Security Standards**: Restricted 정책 적용

## 성능 최적화

1. **리소스 요청/제한 튜닝**: 실제 사용량 기반 조정
2. **HPA 메트릭**: CPU, 메모리 외 커스텀 메트릭 추가
3. **PVC 타입**: SSD 스토리지 클래스 사용
4. **캐싱**: Redis 또는 Memcached 도입
5. **CDN**: 정적 파일 배포에 CDN 사용

## 참고 자료

- [Kubernetes 공식 문서](https://kubernetes.io/docs/)
- [Docker 빌드 가이드](https://docs.docker.com/build/)
- [Helm Chart 작성](https://helm.sh/docs/chart_template_guide/)
- [Prometheus 모니터링](https://prometheus.io/docs/)
- [GitHub Actions](https://docs.github.com/en/actions)

## 지원

문제가 발생하면 GitHub Issues에 보고하세요:
https://github.com/your-org/digital-twin-factory/issues
