# Authentication, Authorization, and Security Policies

## Authentication Protocols
The platform supports OAuth2 with Proof Key for Code Exchange (PKCE) for frontend single-page applications and Mutual TLS (mTLS) for zero-trust service-to-service communication.

### JSON Web Tokens (JWT)
User identities are verified using asymmetric RS256 signed JWT tokens:
- **Token Lifespan**: Access tokens have a strict 15-minute lifespan to limit vulnerability exposure. Refresh tokens have a 7-day expiration and are stored in HTTP-only, secure, SameSite=Strict cookies.
- **Key Rotation**: Public keys for signature verification are dynamically retrieved from the identity provider's JSON Web Key Set (JWKS) endpoint and cached locally for 1 hour.
- **Clock Skew Toleration**: The JWT verification middleware tolerates a maximum clock drift of 30 seconds (`leeway=30`).

## Role-Based Access Control (RBAC)
Authorization is enforced using fine-grained scopes embedded within token claims:
- Scopes follow the format `<resource>:<action>` (for example, `analytics:read` or `billing:admin`).
- Security middleware intercepts inbound requests prior to route handler dispatch and validates that all required scopes are present.
- If a token lacks required scopes, the gateway returns HTTP 403 Forbidden with standard RFC 7807 problem details JSON payload.

## Distributed Rate Limiting
To defend against brute-force attacks and noisy neighbors:
- Public endpoints (such as `/auth/login`) are constrained to 10 requests per minute per IP address.
- Authenticated endpoints enforce a tier-based quota: Standard enterprise tiers receive 1,000 requests per minute; Premium tiers receive 5,000 requests per minute.
- When quotas are exhausted, the gateway responds with HTTP 429 Too Many Requests and specifies the cooldown seconds in the `Retry-After` header.
