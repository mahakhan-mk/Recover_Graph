# Methodology

RecoverGraph retrieves outcome-verified recovery patterns at a pre-tool advisory boundary. Earlier executions produce failure episodes and concrete resolutions; an objective evaluator verifies outcomes before recovery abstraction and indexing. The selected five-pattern memory is frozen before transfer and remains read-only during reported runs.

The agent harness uses PydanticAI with `pydantic-ai-harness` step persistence. Its controlled tools are `read_file`, `write_file`, `edit_file`, `run_tests`, and `run_command`. Before a planned tool call, the advisory service retrieves semantically similar patterns and applies deterministic chronology, verification, and applicability checks. Advice can prompt PydanticAI `ModelRetry` reconsideration before the tool action proceeds. Otherwise the service abstains and the action proceeds.

B0 disables RecoverGraph advice. T enables the same advisory mechanism against the frozen five-pattern memory. Both conditions use the same Kilo transport and `qwen/qwen3-coder` transfer model, controlled tools, request/action limits, retry/timeouts, and frozen SWE-smith fail-to-pass objective evaluator. Workspaces run in isolated Docker containers with task network access disabled.

The memory uses Neo4j with normalized 384-dimensional MiniLM embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and cosine similarity. Cohere (`cohere/north-mini-code:free`) is recorded as the acquisition abstraction model; no abstraction-model call is needed to replay transfer against the frozen memory. No model weights are updated, and no online memory writes occur during treatment.
