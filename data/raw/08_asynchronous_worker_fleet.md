# Asynchronous Worker Fleet and Queue Management

## Task Queue Topology
Asynchronous workloads are scheduled across RabbitMQ message brokers with priority-partitioned queue topologies:
- **Priority Queues**: Jobs are routed to one of three priority tiers: `critical` (payment processing, SLA < 2s), `default` (email notifications, user events, SLA < 30s), and `bulk` (monthly reporting, PDF exports, SLA < 15m).
- **Worker Prefetch Limits**: Celery workers run with `worker_prefetch_multiplier = 1` for long-running compute jobs to prevent idle starvation of other available worker nodes.

## Idempotency and Deduplication
To maintain exactly-once execution semantics across distributed workers:
- **Idempotency Keys**: Tasks generating financial state mutations or email dispatches require a unique `idempotency_key` (UUIDv4) supplied by the caller.
- **Deduplication Lock**: Workers acquire a distributed Redis lock keyed as `task_lock:{idempotency_key}` with a 10-minute TTL before initiating execution.
- If a duplicate task arrives with an active lock or completed state receipt in Redis, the duplicate is acknowledged and discarded without re-execution.

## Poison Pill & Dead Letter Queue (DLQ)
- Failed tasks retry with exponential backoff and randomized jitter up to 5 attempts.
- Tasks encountering unrecoverable errors (e.g. malformed payloads or schema violations) are immediately routed to a dead letter queue (`dlq.tasks`) without consuming retry quotas.
- DLQ depth alerts trigger on Slack and PagerDuty whenever message count exceeds 50 messages.
