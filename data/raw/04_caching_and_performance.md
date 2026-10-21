# Multi-Tier Caching and Performance Optimization

## Caching Strategy Overview
The platform employs a two-tier caching hierarchy designed to achieve sub-10 millisecond median read latencies while maintaining strict data consistency across multi-region deployments.

## Layer 1: In-Memory Local Cache (L1)
- **Technology**: Local process-level LRU (Least Recently Used) cache with bounded capacity (default: 50,000 entries).
- **Lookup Latency**: Sub-microsecond (< 100 microseconds).
- **Target Data**: High-frequency, slow-changing metadata such as organization tenant flags, localized configuration strings, and JWKS public certificates.
- **Eviction Policy**: TTL-based automatic invalidation (default TTL: 300 seconds) combined with pub/sub invalidation broadcasts.

## Layer 2: Distributed Redis Cluster (L2)
- **Technology**: Redis 7.2 Cluster with horizontal sharding across 6 master nodes.
- **Lookup Latency**: 1–3 milliseconds over internal VPC peering.
- **Target Data**: User session state, computed analytical summaries, and API query response bodies.
- **Key Eviction**: All keys must specify an explicit Time-To-Live (TTL). Keys without TTLs are rejected by linting and runtime policies.

## Cache Stampede Mitigation (Probabilistic Early Expiration)
To prevent cache stampedes (dog-piling) when hot keys expire under heavy traffic loads, the caching layer implements the XFetch algorithm:
- When reading a key near its expiration window, a probabilistic equation evaluates whether the current worker should proactively recompute the value in the background before hard expiration occurs.
- The probability of early recalculation increases dynamically as the remaining TTL approaches zero.

## Cache Invalidation Protocols
- When state mutations occur in the persistence layer (PostgreSQL), an asynchronous database trigger emits an event to the Kafka `cache-invalidation` topic.
- A dedicated subscriber service broadcasts Redis `UNLINK` commands to remove stale entries from L2 and sends lightweight UDP/WebSocket pulses to invalidate L1 nodes.
