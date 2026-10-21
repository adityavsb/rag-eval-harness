# Error Handling, Resilience, and Fault Tolerance

## Circuit Breaker Implementation
Outbound HTTP integration calls to third-party microservices and external payment gateways are protected by circuit breakers implemented via the Resilience4j / Polly pattern:
- **Closed State**: Normal operation. Requests pass through directly.
- **Open State**: If the rolling error rate exceeds 50% over a 20-second evaluation window, the circuit trips to Open. Inbound requests immediately fail fast with an HTTP 503 Service Unavailable error without calling the downstream dependency.
- **Half-Open State**: After a 30-second cooldown period, the circuit transitions to Half-Open, permitting 5 trial probe requests. If all 5 succeed, the circuit resets to Closed; if any fail, it trips back to Open for another 60-second cooldown.

## Retry Policies with Exponential Backoff and Jitter
Transient network timeouts and HTTP 502/503/504 responses trigger automated retry policies:
- **Maximum Attempts**: 3 retry attempts.
- **Base Backoff**: 200 milliseconds base delay, doubling on each subsequent failure ($200\text{ms} \rightarrow 400\text{ms} \rightarrow 800\text{ms}$).
- **Full Jitter**: A pseudo-random uniform jitter between 0 and the current exponential backoff interval is added to avoid synchronized request spikes across worker clusters.

## Dead Letter Queues (DLQ)
Asynchronous message processing pipelines running on Kafka and RabbitMQ isolate poison-pill messages:
- When a task fails after 3 execution attempts, it is acknowledged and routed to the corresponding Dead Letter Queue (`dlq.<topic_name>`).
- DLQ messages preserve original payload data, exception tracebacks, invocation timestamps, and retry attempt counters in custom header envelopes.
- An alert is dispatched to PagerDuty if the DLQ depth exceeds 50 messages within any 15-minute sliding window.
