CREATE INDEX failure_signature IF NOT EXISTS
FOR (n:FailureEpisode) ON (n.signature);

CREATE INDEX failure_type IF NOT EXISTS
FOR (n:FailureEpisode) ON (n.failure_type);

CREATE INDEX action_operation IF NOT EXISTS
FOR (n:Action) ON (n.operation);
