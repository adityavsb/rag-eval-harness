# Observability, Distributed Tracing, and Telemetry

## Distributed Tracing Framework
The platform instruments all inbound requests and outbound RPC calls using the OpenTelemetry (OTel) Python SDK:
- **Propagation Format**: W3C Trace Context headers (`traceparent` and `tracestate`) propagate context across HTTP boundaries and asynchronous Kafka message headers.
- **Trace Sampling Strategy**: Adaptive head-based sampling is set at 5% for successful 2xx/3xx requests, while 100% of errors (HTTP 4xx/5xx) and slow requests (P99 latency > 800ms) are forced into persistent Jaeger/Grafana Tempo storage.
- **Span Conventions**: Database queries record sanitized SQL statements with statement durations, omitting literal parameter values to prevent PII exposure.

## Metrics Collection Architecture
Prometheus collectors scrape metrics endpoints exposed by service pods on port 9090:
- **Scrape Interval**: Default scrape interval is 15 seconds across production services.
- **RED Metrics**: Every microservice publishes Rate (requests/sec), Errors (error count tagged by HTTP status), and Duration (histogram buckets from 5ms to 10s).
- **Custom Metric Cardinality**: Label combinations are limited to a maximum cardinality of 1,000 unique metric permutations per service instance to prevent Prometheus memory exhaustion.

## Structured Logging & Correlation
All application logs are output in single-line JSON format directly to stdout:
- Mandatory fields include `timestamp` (ISO 8601 UTC), `level`, `service_name`, `trace_id`, `span_id`, and `tenant_id`.
- Correlation between log records and APM spans enables zero-click jump from a Grafana error alert directly to the causal Jaeger trace.
