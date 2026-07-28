# Literature intake and parsing

Stage 5 bridges conversation attachments into the existing scientific task
control plane. It does not treat chat history or an uploaded file as scientific
truth, and it does not add a second worker queue.

## Authority boundary

The flow is:

1. A user uploads PDFs as content-addressed conversation attachments.
2. The selected conversation message is frozen as an immutable
   `conversation_execution_request.v1` with exact attachment SHA-256 values.
3. The literature-intake endpoint copies verified bytes into an owned,
   immutable corpus roster under
   `projects/<project_id>/literature-intakes/<intake_id>/inputs/`.
4. `literature_intake.v1` binds the frozen request digest, parser profile,
   complete input roster, corpus digest, RunPlan, and authorization policy.
5. Parsing runs through `RunPlanExecutor`, `StageState`, the artifact registry,
   and existing Gate snapshots. The intake manifest is not a task state machine.

Conversation messages remain UI records. ParsedDocument output and parser audit
become scientific artifacts only after the existing parser task succeeds and
the executor registers its outputs.

## Authorization

- One local PDF no larger than 25 MiB with `pdfplumber_local`: clicking
  **文献解析** is the explicit authorization. The existing
  `parse_document_pdfplumber` task runs directly.
- A single PDF above the click-authorization limit is treated like a batch and
  requires the data-mining Gate.
- Two to twenty local PDFs: the existing executor first publishes
  `WAITING_USER` for `gate_2_data_mining`. No parser is dispatched until the
  exact execution snapshot is approved. The gated
  `parse_pdf_corpus_pdfplumber` task then parses the immutable roster.
- Remote MinerU, OCR/model fallbacks, and other high-cost profiles are not
  enabled by this Stage 5 endpoint. They remain gated work for the constrained
  transport profiles introduced in Stage 6.

## Safety and replay

- Only frozen PDF attachments are accepted; media type/name and `%PDF-` bytes
  are checked before corpus registration.
- Corpus filenames are generated from ordinal positions and content digests,
  never from user-supplied paths.
- Symlinks and unexpected corpus roster entries fail closed.
- Every owned PDF is rehashed when an intake is read or approved.
- Repeating registration returns the existing intake and existing StageState;
  it does not execute a completed task again.
- Parser outputs must remain under the run directory before they can enter the
  artifact registry.
