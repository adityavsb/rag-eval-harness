# API Versioning, Contract Evolution, and Deprecation Policies

## Versioning Strategy
The API platform adheres to Semantic Versioning (SemVer 2.0.0) implemented via URL path prefixes:
- Current active release channels are mounted under `/api/v1` and `/api/v2`.
- Breaking changes (field removals, type transformations, or required parameter additions) strictly require incrementing the major version path.
- Non-breaking changes (new optional query parameters or additional response payload keys) are introduced directly into active versions without path modifications.

## Deprecation Lifecycle & Sunset Headers
When an API endpoint or version is slated for decommissioning:
- **Deprecation Notice**: The platform injects the standard RFC 8594 `Deprecation: true` response header along with a `Sunset` header specifying the timestamp of final removal (minimum 180-day deprecation window).
- **Migration Link**: A `Link: <url>; rel="deprecation"` header points consumers directly to the migration guide.
- **Consumer Telemetry**: Inbound calls to deprecated endpoints emit Prometheus metric `api_deprecated_endpoint_invocations_total` tagged by `client_id` and `endpoint`. Automated emails alert account owners when traffic persists 30 days before sunset.

## Contract Testing & Schema Validation
All service boundaries enforce strict OpenAPI 3.1.0 specifications:
- Pull requests validate backward compatibility using `oasdiff` against the production OpenAPI schema artifact.
- Any change introducing breaking contract deviations automatically blocks merge in CI.
