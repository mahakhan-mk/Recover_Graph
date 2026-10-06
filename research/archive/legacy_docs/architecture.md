# Architecture

Graph Swarm is a modular research monolith. The experimental PydanticAI harness depends on a small advisory/memory boundary. Domain models remain framework-neutral. Neo4j is hidden behind a repository interface.

```text
                    domain
                      ^
          +-----------+------------+
          |           |            |
        graph       memory      retrieval
          ^           ^            ^
          +-----------+------------+
                      |
                   advisory
                      ^
                      |
                    agent
                      |
                      v
                 experiments
```

Production-only components such as HTTP services, telemetry collectors, dashboards, authentication, and multi-framework adapters are deliberately excluded.
