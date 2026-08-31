# GS-E002 Preserved Evidence

This directory preserves the GS-E002 research evidence produced by the frozen-agent experiment.

The frozen treatment model was `openai/gpt-oss-120b`. The acquisition phase established a known failure and observed recovery. The control phase repeated the task without advisory memory; the known failure recurred and the objective task still succeeded. The treatment phase delivered historical advice before execution; the known failure still recurred, the objective task succeeded, and the recorded behavior observation was unchanged.

These files are preserved research evidence, not runtime configuration. The original ignored run copies remain under `artifacts/runs/`. The preserved files are byte-identical copies; see `manifest.json` for source and SHA256 metadata.

The copied pilot records retain their original historical fixture-path metadata. That metadata is intentionally unchanged as part of preserving the experimental result and is not an active repository path.
