"""Named Cypher statements used by the Neo4j repository."""

SAVE_RUN = """
MERGE (n:Run {id: $id})
SET n.task_id = $task_id,
    n.started_at = $started_at
"""

SAVE_TASK = """
MERGE (n:Task {id: $id})
SET n.problem_statement = $problem_statement,
    n.family_id = $family_id,
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
    n.failed_observations = $failed_observations,
    n.observed_at = $observed_at
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

SAVE_RECOVERY_PATTERN = """
MATCH (failure:FailureEpisode {id: $source_failure_id})
MATCH (resolution:Resolution {id: $source_resolution_id})
MATCH (outcome:Outcome {id: $source_outcome_id})
MATCH (task:Task {id: $source_task_id})
MATCH (failure)-[:RESOLVED_BY]->(resolution)
MATCH (resolution)-[:VERIFIED_BY]->(outcome)
MATCH (resolution)-[:OBSERVED_CHANGE]->(recovery_action:Action)
MATCH (task)-[:HAS_ACTION]->(recovery_action)
MATCH (recovery_action)-[:PART_OF]->(source_run:Run)
MATCH (task)-[:HAS_ACTION]->(failed_action:Action)-[:PART_OF_FAILURE]->(failure)
MATCH (failed_action)-[:PART_OF]->(source_run)
WHERE task.chronological_index = $source_chronological_index
  AND failure.failure_type = $source_failure_type
  AND failed_action.tool = $source_tool
  AND failed_action.operation = $source_operation
OPTIONAL MATCH (existing:RecoveryPattern {id: $id})
WITH failure, resolution, outcome, task, existing,
     existing IS NULL OR (
         existing.source_failure_id = $source_failure_id
         AND existing.source_resolution_id = $source_resolution_id
         AND existing.source_outcome_id = $source_outcome_id
         AND existing.source_task_id = $source_task_id
         AND existing.source_chronological_index = $source_chronological_index
     ) AS provenance_matches
FOREACH (_ IN CASE WHEN provenance_matches THEN [1] ELSE [] END |
    MERGE (pattern:RecoveryPattern {id: $id})
    SET pattern.title = $title,
        pattern.guidance = $guidance,
        pattern.source_failure_id = $source_failure_id,
        pattern.source_resolution_id = $source_resolution_id,
        pattern.source_outcome_id = $source_outcome_id,
        pattern.source_task_id = $source_task_id,
        pattern.source_chronological_index = $source_chronological_index,
        pattern.source_tool = $source_tool,
        pattern.source_operation = $source_operation,
        pattern.applicability_tool = $applicability_tool,
        pattern.applicability_operation = $applicability_operation,
        pattern.source_failure_type = $source_failure_type,
        pattern.environment_runtime = $environment_runtime,
        pattern.environment_versions_json = $environment_versions_json,
        pattern.environment_dependencies_json = $environment_dependencies_json,
        pattern.environment_markers_json = $environment_markers_json,
        pattern.verification_status = $verification_status,
        pattern.evidence_count = $evidence_count,
        pattern.evidence_summary = $evidence_summary,
        pattern.embedding = $embedding,
        pattern.created_at = $created_at,
        pattern.invalidated_at = $invalidated_at
    MERGE (pattern)-[:SOURCE_FAILURE]->(failure)
    MERGE (pattern)-[:SOURCE_RESOLUTION]->(resolution)
    MERGE (pattern)-[:SOURCE_OUTCOME]->(outcome)
    MERGE (pattern)-[:SOURCE_TASK]->(task)
)
RETURN provenance_matches
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

LINK_RESOLUTION_OBSERVED_CHANGE = """
MATCH (resolution:Resolution {id: $resolution_id})
MATCH (action:Action {id: $action_id})
MERGE (resolution)-[:OBSERVED_CHANGE]->(action)
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
MATCH (task:Task)-[:HAS_ACTION]->(failed_action:Action)-[:PART_OF_FAILURE]->(failure)
MATCH (failed_action)-[:PART_OF]->(run:Run)
MATCH (task)-[:HAS_ACTION]->(action:Action)-[:PART_OF]->(run)
RETURN DISTINCT action, task, run
ORDER BY action.id
"""

GET_TOOLS = """
MATCH (failure:FailureEpisode {id: $failure_id})
MATCH (task:Task)-[:HAS_ACTION]->(failed_action:Action)-[:PART_OF_FAILURE]->(failure)
MATCH (failed_action)-[:PART_OF]->(run:Run)
MATCH (task)-[:HAS_ACTION]->(action:Action)-[:PART_OF]->(run)
MATCH (action)-[:USED]->(tool:Tool)
RETURN DISTINCT tool
ORDER BY tool.name
"""

GET_RESOLUTIONS = """
MATCH (failure:FailureEpisode {id: $failure_id})
MATCH (failure)-[:RESOLVED_BY]->(resolution:Resolution)
OPTIONAL MATCH (resolution)-[:VERIFIED_BY]->(outcome:Outcome)
OPTIONAL MATCH (resolution)-[:OBSERVED_CHANGE]->(observed_change:Action)
RETURN resolution,
       collect(DISTINCT outcome) AS outcomes,
       collect(DISTINCT observed_change) AS observed_changes
ORDER BY resolution.id
"""

GET_RECOVERY_PATTERN = """
MATCH (pattern:RecoveryPattern {id: $pattern_id})
MATCH (pattern)-[:SOURCE_FAILURE]->(failure:FailureEpisode)
MATCH (pattern)-[:SOURCE_RESOLUTION]->(resolution:Resolution)
MATCH (pattern)-[:SOURCE_OUTCOME]->(outcome:Outcome)
MATCH (pattern)-[:SOURCE_TASK]->(task:Task)
MATCH (failure)-[:OCCURRED_IN]->(environment:Environment)
MATCH (failure)-[:RESOLVED_BY]->(resolution)
MATCH (resolution)-[:VERIFIED_BY]->(outcome)
MATCH (resolution)-[:OBSERVED_CHANGE]->(recovery_action:Action)
MATCH (task)-[:HAS_ACTION]->(recovery_action)
MATCH (recovery_action)-[:PART_OF]->(source_run:Run)
MATCH (task)-[:HAS_ACTION]->(failed_action:Action)-[:PART_OF_FAILURE]->(failure)
MATCH (failed_action)-[:PART_OF]->(source_run)
WHERE pattern.source_failure_id = failure.id
  AND pattern.source_resolution_id = resolution.id
  AND pattern.source_outcome_id = outcome.id
  AND pattern.source_task_id = task.id
  AND pattern.source_chronological_index = task.chronological_index
  AND pattern.source_failure_type = failure.failure_type
  AND pattern.source_tool = failed_action.tool
  AND pattern.source_operation = failed_action.operation
RETURN pattern, failure, resolution, outcome, task, environment,
       recovery_action, failed_action
"""

GET_RECOVERY_EVIDENCE = """
MATCH (task:Task)-[:HAS_ACTION]->(failed_action:Action)-[:PART_OF_FAILURE]->(failure:FailureEpisode)
MATCH (failed_action)-[:PART_OF]->(source_run:Run)
MATCH (failure)-[:RESOLVED_BY]->(resolution:Resolution)
MATCH (resolution)-[:OBSERVED_CHANGE]->(recovery_action:Action)
MATCH (task)-[:HAS_ACTION]->(recovery_action)
MATCH (recovery_action)-[:PART_OF]->(source_run)
MATCH (resolution)-[:VERIFIED_BY]->(outcome:Outcome)
MATCH (failure)-[:OCCURRED_IN]->(environment:Environment)
WHERE failure.id = $failure_id
  AND failure.action_id = failed_action.id
  AND failed_action.task_id = task.id
  AND recovery_action.task_id = task.id
RETURN DISTINCT failure, resolution, outcome, task, environment,
       failed_action, recovery_action
ORDER BY resolution.id, outcome.id, failed_action.id, recovery_action.id
"""

UPDATE_RECOVERY_PATTERN_EMBEDDING = """
MATCH (pattern:RecoveryPattern {id: $pattern_id})
SET pattern.embedding = $embedding
RETURN pattern
"""

CREATE_RECOVERY_PATTERN_VECTOR_INDEX = """
CREATE VECTOR INDEX recovery_pattern_embedding_idx IF NOT EXISTS
FOR (pattern:RecoveryPattern) ON (pattern.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
    }
}
"""

QUERY_RECOVERY_PATTERN_VECTORS = """
CALL db.index.vector.queryNodes(
    'recovery_pattern_embedding_idx',
    $limit,
    $query_embedding
) YIELD node AS pattern, score AS vector_score
OPTIONAL MATCH (pattern)-[:SOURCE_FAILURE]->(failure:FailureEpisode)
OPTIONAL MATCH (pattern)-[:SOURCE_TASK]->(source_task:Task)
OPTIONAL MATCH (source_task)-[:HAS_ACTION]->(failed_action:Action)-[:PART_OF_FAILURE]->(failure)
RETURN pattern, vector_score, failure, source_task, failed_action
ORDER BY vector_score DESC
"""

FIND_HISTORICAL_RECOVERY_CANDIDATES = """
MATCH (failed_action:Action)-[:PART_OF_FAILURE]->(failure:FailureEpisode)
MATCH (failure)-[:OCCURRED_IN]->(historical_environment:Environment)
MATCH (failure)-[:RESOLVED_BY]->(resolution:Resolution)
MATCH (resolution)-[:VERIFIED_BY]->(qualifying_outcome:Outcome)
OPTIONAL MATCH (resolution)-[:VERIFIED_BY]->(outcome:Outcome)
WHERE failed_action.tool = $tool
  AND failed_action.operation = $operation
  AND failed_action.run_id <> $current_run_id
  AND failure.observed_at <= $planned_at
  AND resolution.status = 'observed_successful'
  AND qualifying_outcome.success = true
WITH failed_action, failure, historical_environment, resolution,
     collect(DISTINCT outcome) AS outcomes
RETURN failed_action.id AS failed_action_id,
       failed_action.run_id AS source_run_id,
       failed_action.tool AS tool,
       failed_action.operation AS operation,
       failed_action.planned_at AS planned_at,
       failure.id AS failure_id,
       failure.action_id AS failure_action_id,
       failure.failure_type AS failure_type,
       failure.signature AS failure_signature,
       failure.symptom AS symptom,
       failure.observed_at AS failure_observed_at,
       historical_environment.id AS environment_id,
       historical_environment.repository AS repository,
       historical_environment.runtime AS runtime,
       historical_environment.versions_json AS versions_json,
       historical_environment.markers_json AS markers_json,
       resolution.id AS resolution_id,
       resolution.failure_id AS resolution_failure_id,
       resolution.description AS resolution_description,
       resolution.status AS resolution_status,
       resolution.successful_observations AS successful_observations,
       resolution.failed_observations AS failed_observations,
       resolution.observed_at AS resolution_observed_at,
       outcomes
ORDER BY resolution.successful_observations DESC,
         size([item IN outcomes WHERE item.success = true]) DESC,
         failure.observed_at DESC,
         resolution.id ASC,
         failure.id ASC
"""

COUNT_RECOVERY_PATTERN_VECTORS = """
MATCH (pattern:RecoveryPattern)
WHERE pattern.embedding IS NOT NULL
RETURN count(pattern) AS count
"""
