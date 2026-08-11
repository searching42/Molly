# BR2 real MinerU runtime acceptance v1

This evidence records the fresh runtime acceptance for PR #49 on reviewed code
HEAD `47ce2aa8623d07d6002adcf2b5576f0b9abee949`, based on
`2886aac2e2f8bd2896856bd40f06d7124a2ea5dd`.

One real OLED/TADF PDF was submitted through the current Molly path:

```text
Harness Controller
  -> RemoteExecutionService / fixed worker protocol
  -> MinerU worker adapter
  -> real MinerU 3.4.0 (pipeline backend)
  -> parsed-corpus publication
  -> existing remote verifier and Controller adoption
```

The worker was changed only to select MinerU's supported `pipeline` backend
after the configured default hybrid/vLLM backend failed at runtime with the
provider's FlashInfer `sm75` startup error. The fixed worker protocol,
Controller authority, publication contract, and verifier remain in use.

The accepted output is a non-empty existing `ParsedDocument` structure with 12
pages, 126 elements, and 2 tables. The final Controller inspection was
`SUCCEEDED` with `COMPLETE_EXECUTION`; the remote publication passed the
existing parsed-corpus verifier and was adopted once after one dispatch.

The existing operational `DATA_MINING` precondition Gate was approved to permit
the runtime action. This is not the later human confirmation Gate: contextual
mapping, candidate-dataset creation, and confirmation did not start. No raw PDF,
worker log, hostname, username, network address, command, credential, or
private absolute path is included in this evidence directory.

See [acceptance_manifest.json](acceptance_manifest.json) for bound identities
and [runtime_summary.json](runtime_summary.json) for the boundary result.
