# User Context

This file describes Benton only for work inside this main/direct workspace.
Do not share this context in group chats or unrelated shared contexts.

## Human

- Name inferred from the environment path: Benton.
- Usually works in Chinese for project status, planning, and summaries.
- Values direct, practical engineering help: read the code, do the work, verify
  it, and report the result clearly.
- Prefers concrete evidence over vague reassurance. Mention commands run and
  whether they passed.
- Often asks for project-state summaries, implementation help, UI/workflow
  cleanup, and research/artifact generation around AI4S and OLED molecular
  discovery.

## Collaboration Preferences

- Be concise but not shallow. Lead with the answer, then give the evidence.
- When editing code, inspect the repo first and follow existing patterns.
- When the task is straightforward, make the change rather than stopping at a
  proposal.
- Do not overwrite existing local changes without explicit approval.
- For reviews, prioritize bugs, regressions, missing tests, and concrete
  file/line findings.
- Use Chinese for user-facing summaries unless there is a clear reason to use
  English.

## Risk And Privacy Preferences

- Ask before actions that leave the machine: remote jobs, network acquisition,
  uploads, emails, posts, or external LLM calls using private data.
- Keep project memory factual and useful. Do not store secrets, raw datasets, or
  private data samples in memory files.
- Remote/GPU work and expensive generation should be plan-only by default until
  explicitly confirmed.

## Project Interest

Benton is building an AI4S Agent / OpenClaw-style workflow for molecular
discovery. The work emphasizes:

- data cleaning and trainability review,
- Uni-Mol and baseline model training,
- candidate prediction and screening,
- REINVENT4-style generation,
- evidence-grounded literature-to-dataset workflows,
- auditable artifacts and human confirmation gates,
- an agentic layer that plans, verifies, explains, and replans without hiding
  deterministic execution behind free-form chat.

