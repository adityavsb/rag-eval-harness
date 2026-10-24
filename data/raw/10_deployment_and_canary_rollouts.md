# Deployment Pipelines, GitOps, and Canary Rollout Strategies

## GitOps Deployment Model
Production deployments are orchestrated via ArgoCD following a declarative GitOps workflow:
- The `main` branch state defines the desired cluster state across Kubernetes namespaces.
- Infrastructure changes and application deployments are committed as declarative Helm values stored in version-controlled Git repositories.

## Canary Rollout Architecture
Microservice releases utilize Argo Rollouts to mitigate risk through automated canary analysis:
- **Phase 1 (Traffic Split: 5%)**: Route 5% of production traffic to the canary replica set for 10 minutes.
- **Phase 2 (Traffic Split: 25%)**: If error budget remains intact, increase traffic to 25% for 20 minutes.
- **Phase 3 (Traffic Split: 100%)**: Promote canary to full production workload and retire previous version replicas.

## Automated Rollback Triggers
Canary deployments continuously evaluate analysis metrics against strict health criteria:
- **5xx Error Rate**: If HTTP 5xx responses exceed 0.5% of total requests over a 2-minute rolling window, the rollout automatically aborts and traffic rolls back to the stable replica set within 15 seconds.
- **Latency Degradation**: If P95 latency increases by more than 20% compared to the baseline stable version, the canary rollout is paused for engineering review.
- **Container Health Probes**: Pods must pass `/healthz/live` (liveness probe every 10s) and `/healthz/ready` (readiness probe every 5s) before receiving gateway traffic.
