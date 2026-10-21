# Enterprise API Platform: Architecture Overview

## System Overview
The Enterprise API Platform provides a high-throughput, low-latency microservices gateway and runtime engine. The system is designed around a decoupled event-driven architecture, separating request routing from background task processing.

## Core Architectural Components

### 1. API Gateway Layer
The Gateway acts as the single ingress point for all client requests. It handles:
- TLS termination and SSL certificate offloading.
- Global traffic routing using path-based and header-based route rules.
- Distributed rate limiting enforced via token bucket algorithms backed by Redis.
- Request correlation ID generation for end-to-end distributed tracing.

### 2. Execution Engine & Worker Pools
Request execution is divided into two distinct worker tiers:
- **Synchronous Web Workers**: Asynchronous event-loop workers (running Uvicorn and ASGI handlers) that process non-blocking I/O requests such as database lookups and external API queries.
- **Background Worker Fleet**: Dedicated Celery and Huey task workers that consume heavy asynchronous jobs from RabbitMQ and Redis queues, including document rendering, data export jobs, and batch notification dispatches.

### 3. Event Bus & Messaging Backbone
Internal service communication utilizes Apache Kafka as the durable event backbone:
- **Event Retention**: Critical operational topics maintain a 7-day retention period with compaction enabled for state streams.
- **Partitioning Strategy**: Partition keys are mapped to `tenant_id` to guarantee in-order delivery of state transitions per enterprise customer.

### 4. Data Persistence Tier
- **Relational Storage**: PostgreSQL 16 managed clusters utilizing patroni for automated failover, serving as the source of truth for user accounts and transactional entities.
- **Distributed Cache**: Redis Cluster running with multi-node replication for session states and ephemeral cache keys.
