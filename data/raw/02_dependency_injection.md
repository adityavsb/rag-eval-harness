# Dependency Injection & Lifecycle Management

## Architecture Principles
The platform features a hierarchical Dependency Injection (DI) system inspired by modern ASGI frameworks. The DI container decouples service resolution from HTTP request handlers, facilitating automated mocking during integration tests and deterministic resource cleanup.

## Sub-Dependency Resolution
Dependencies can declare nested sub-dependencies in a directed acyclic graph (DAG):
- The container traverses the dependency tree using topological sorting before invoking route handlers.
- Both synchronous callable objects (`def`) and asynchronous coroutines (`async def`) are supported transparently.
- Asynchronous dependencies are awaited concurrently where graph branches do not have inter-dependencies.

## Caching and Scopes
By default, the dependency injection engine caches resolved values for the duration of a single HTTP request:
- **Request Scope (Default `use_cache=True`)**: If multiple route parameters or nested dependencies require `get_db_session()`, the session is resolved once on the first call and reused across all subsequent nodes in the same request DAG.
- **Transient Scope (`use_cache=False`)**: When unique instances are required per invocation (such as cryptographic nonces or dedicated worker connection handles), setting `use_cache=False` forces re-execution of the dependency factory.

## Resource Teardown and Context Managers
Dependencies that yield resources (such as database connections or file handles) utilize Python generator functions:
- Code preceding the `yield` statement executes during request initialization.
- The yielded object is passed as the parameter value to the endpoint handler.
- Code following the `yield` statement executes during request teardown in the `finally` block, ensuring connections are closed even if an unhandled exception occurs during handler execution.
