"""Named Cypher statements used by the Neo4j repository."""

SAVE_RUN = """
MERGE (n:Run {id: $id})
SET n.task_id = $task_id,
    n.started_at = $started_at
"""

SAVE_TASK = """
MERGE (n:Task {id: $id})
SET n.family_id = $family_id,
    n.repository = $repository,
    n.chronological_index = $chronological_index
"""

SAVE_ACTION = """
MERGE (n:Action {id: $id})
SET n.run_id = $run_id,
    n.task_id = $task_id,
    n.tool = $tool,
    n.operation = $operation,
    n.arguments_json = $arguments_json,
    n.planned_at = $planned_at,
    n.result_tool_name = $result_tool_name,
    n.success = $success,
    n.exit_code = $exit_code,
    n.output = $output,
    n.error = $error,
    n.started_at = $started_at,
    n.completed_at = $completed_at
"""

SAVE_TOOL = """
MERGE (n:Tool {name: $name})
SET n.name = $name
"""

SAVE_ENVIRONMENT = """
MERGE (n:Environment {id: $id})
SET n.repository = $repository,
    n.runtime = $runtime,
    n.versions_json = $versions_json,
    n.markers_json = $markers_json
"""

SAVE_FAILURE = """
MERGE (n:FailureEpisode {id: $id})
SET n.action_id = $action_id,
    n.failure_type = $failure_type,
    n.signature = $signature,
    n.symptom = $symptom,
    n.observed_at = $observed_at
"""

SAVE_RESOLUTION = """
MERGE (n:Resolution {id: $id})
SET n.failure_id = $failure_id,
    n.description = $description,
    n.status = $status,
    n.successful_observations = $successful_observations,
    n.failed_observations = $failed_observations
"""

SAVE_OUTCOME = """
MERGE (n:Outcome {id: $id})
SET n.action_id = $action_id,
    n.success = $success,
    n.tests_passed = $tests_passed,
    n.tests_failed = $tests_failed,
    n.exit_code = $exit_code,
    n.observed_at = $observed_at
"""

LINK_TASK_ACTION = """
MATCH (task:Task {id: $task_id})
MATCH (action:Action {id: $action_id})
MERGE (task)-[:HAS_ACTION]->(action)
RETURN count(*) AS linked
"""

LINK_ACTION_TOOL = """
MATCH (action:Action {id: $action_id})
MATCH (tool:Tool {name: $tool_name})
MERGE (action)-[:USED]->(tool)
RETURN count(*) AS linked
"""

LINK_ACTION_RUN = """
MATCH (action:Action {id: $action_id})
MATCH (run:Run {id: $run_id})
MERGE (action)-[:PART_OF]->(run)
RETURN count(*) AS linked
"""

LINK_ACTION_FAILURE = """
MATCH (action:Action {id: $action_id})
MATCH (failure:FailureEpisode {id: $failure_id})
MERGE (action)-[:PART_OF_FAILURE]->(failure)
RETURN count(*) AS linked
"""

LINK_FAILURE_ENVIRONMENT = """
MATCH (failure:FailureEpisode {id: $failure_id})
MATCH (environment:Environment {id: $environment_id})
MERGE (failure)-[:OCCURRED_IN]->(environment)
RETURN count(*) AS linked
"""

LINK_FAILURE_RESOLUTION = """
MATCH (failure:FailureEpisode {id: $failure_id})
MATCH (resolution:Resolution {id: $resolution_id})
MERGE (failure)-[:RESOLVED_BY]->(resolution)
RETURN count(*) AS linked
"""

LINK_RESOLUTION_OUTCOME = """
MATCH (resolution:Resolution {id: $resolution_id})
MATCH (outcome:Outcome {id: $outcome_id})
MERGE (resolution)-[:VERIFIED_BY]->(outcome)
RETURN count(*) AS linked
"""

GET_FAILURE = """
MATCH (failure:FailureEpisode {id: $failure_id})
RETURN failure
"""

GET_ENVIRONMENT = """
MATCH (failure:FailureEpisode {id: $failure_id})
MATCH (failure)-[:OCCURRED_IN]->(environment:Environment)
RETURN environment
ORDER BY environment.id
"""

GET_ACTION_CONTEXT = """
MATCH (failure:FailureEpisode {id: $failure_id})
MATCH (task:Task)-[:HAS_ACTION]->(action:Action)-[:PART_OF_FAILURE]->(failure)
MATCH (action)-[:PART_OF]->(run:Run)
RETURN DISTINCT action, task, run
ORDER BY action.id
"""

GET_TOOLS = """
MATCH (failure:FailureEpisode {id: $failure_id})
MATCH (action:Action)-[:PART_OF_FAILURE]->(failure)
MATCH (action)-[:USED]->(tool:Tool)
RETURN DISTINCT tool
ORDER BY tool.name
"""

GET_RESOLUTIONS = """
MATCH (failure:FailureEpisode {id: $failure_id})
MATCH (failure)-[:RESOLVED_BY]->(resolution:Resolution)
OPTIONAL MATCH (resolution)-[:VERIFIED_BY]->(outcome:Outcome)
RETURN resolution, collect(DISTINCT outcome) AS outcomes
ORDER BY resolution.id
"""
