# Historical Execution Agent proposal publication

Byte-exact Execution Agent proposal publication generated with the pre-PR #38
code at commit `bf82cfd` via
`ExecutionAgentService.create_proposal` / `ExecutionAgentStore.publish_proposal`
(selected tool: `agent.pause_current.v1`).

The tool catalog in this publication predates the `option_schema` field:
serializing it with current code must not add `"option_schema": null` to any
tool, otherwise `ExecutionAgentStore.read_proposal` exact byte verification
fails.  `tests/test_execution_agent.py::test_historical_execution_agent_publication_replays_byte_exact`
pins the full store read.

`manifest.json` records the stable `tool_call_proposal_id`, the selected tool
and the catalog roster.
