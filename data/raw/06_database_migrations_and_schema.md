# Database Migrations, Schema Evolution, and Connection Management

## Schema Evolution Strategy
The platform enforces zero-downtime database migrations using the Expand-and-Contract (Parallel Run) pattern managed through Alembic:
- **Expand Phase**: New columns, tables, and non-blocking indexes are created concurrently without modifying existing columns or breaking active application workers.
- **Contract Phase**: Deprecated columns and tables are dropped only after all application instances have migrated to the new schema version and backward-compatibility grace periods (minimum 14 days) have elapsed.

## Lock Prevention & DDL Safety
To avoid catastrophic table-level lock contention on production PostgreSQL 16 clusters:
- All DDL statements must set `lock_timeout = '2s'` and `statement_timeout = '30s'`.
- Index creation must strictly use `CREATE INDEX CONCURRENTLY` to prevent write locks on active transaction tables.
- Adding `NOT NULL` constraints requires adding a check constraint first as `NOT VALID`, validating it asynchronously, and enforcing the constraint once verified.

## Connection Pooling Architecture
Client applications connect to PostgreSQL via PgBouncer proxy pools:
- **Pooling Mode**: Transaction pooling is standard, allowing connections to be reused across different transactions while prohibiting session-level constructs like temporary tables.
- **Pool Sizing**: Maximum server connections per database node are capped at 300, while PgBouncer maintains up to 5,000 active client connections.
- **Read Replicas**: Read-only queries route through separate replica endpoints with streaming replication lag monitored via `pg_stat_replication`. If replica lag exceeds 500ms, traffic automatically fails over to the primary instance.
