# Custom Corpus Real Literature Read-Only Acceptance

## Purpose

The real literature read-only acceptance harness provides a local-only,
non-materializing check for a tiny operator-selected OLED or organic
optoelectronic literature set. It reads a safe manifest and redacted parsed
summary files, then emits aggregate evidence for parseability, candidate-table
presence, property-field coverage, and failure taxonomy.

## Why Real Literature Acceptance Starts Now

The governance chain, execution state machine, and execution provenance binding
layer now protect controlled writer execution from skipped states and orphan
artifacts. The next risk is whether real literature metadata and parsed outputs
can be inspected safely without leaking raw content. This harness answers that
question before any writer execution, dataset materialization, or training
artifact work.

## Position in the Governance Chain

```text
real literature local manifest
-> local parsed-output presence check
-> redacted paper-level aggregate scan
-> candidate table aggregate detection
-> property field coverage aggregate
-> failure taxonomy aggregate
-> real literature read-only acceptance evidence
-> future real candidate quarantine dry-run
```

This branch is read-only and separate from controlled writer execution.

## Recommended Five-Paper OLED Local Set

The following examples are recommended for local operator-confirmed access only.
They are examples for local operator-confirmed access only. The repository must
not include PDFs, raw parsed content, raw tables, raw text, extracted values,
paper titles in outputs, or DOI strings in outputs.

1. Uoyama et al., "Highly efficient organic light-emitting diodes from delayed fluorescence", Nature 2012.
2. Nakanotani et al., "High-efficiency organic light-emitting diodes with fluorescent emitters", Nature Communications 2014.
3. Kaji et al., "Purely organic electroluminescent material realizing 100% conversion from electricity to light", Nature Communications 2015.
4. Evans et al., "Singlet and triplet to doublet energy transfer: improving organic light-emitting diodes with radicals", arXiv or published work.
5. Bunzmann et al., "Optically and electrically excited intermediate electronic states in donor:acceptor based OLEDs", arXiv.

These examples are references for local selection only. Harness outputs must use
safe paper ids and aggregate counts instead of titles.

## Input Manifest

Manifest schema:

```text
custom_corpus_real_literature_read_only_acceptance_manifest.v1
```

Allowed manifest content is limited to schema labels, acceptance id, corpus id,
domain, input mode, operator-confirmed access flag, paper count, safe paper ids,
safe parsed-output basenames, and optional safe note labels.

The manifest must not contain paper titles, DOI strings, URLs, raw paths,
absolute paths, PDF paths, raw text, raw table content, molecular strings,
exact numeric values, credentials, or auth headers.

## Parsed-Output Summary Contract

The harness reads only known summary filenames under each safe parsed-output
basename:

```text
parsed_output_summary.json
acceptance_summary.json
```

The parsed-output summary schema is:

```text
custom_corpus_real_literature_parsed_output_summary.v1
```

The summary may contain aggregate counts, safe property category labels,
candidate status counts, and safe failure labels. It must not contain raw
article text, raw tables, exact values, paper titles, DOI strings, PDF names,
or local paths.

## Acceptance Checks

The harness validates manifest schema, safe ids, local basename-only parsed
output references, operator-confirmed access, clean output directory,
parseable-paper count, candidate-table count, property-candidate count,
property category coverage, candidate status counts, failure taxonomy, and
redaction status.

## Acceptance Status Semantics

`acceptance_passed` means the manifest is safe, operator access is confirmed,
minimum aggregate thresholds are met, redaction passes, and no execution or
materialization boundary flag is true.

`acceptance_needs_review` means no hard blocker exists, but missing parsed
outputs, failure categories, unconfirmed access allowed by policy, count
mismatch, or unmet minimum thresholds require review.

`acceptance_blocked` means schema, safety, redaction, path, output-directory,
access, or boundary validation failed.

## Failure Taxonomy

Safe failure labels include parsed output missing, invalid parsed JSON, invalid
parsed schema, paper id mismatch, missing table count, missing candidate-table
count, missing property-candidate count, missing property categories, missing
candidate status counts, table not found, compound alias unresolved, property
header unmapped, unit missing, value ambiguous, molecule structure missing,
redaction blocked, dry-run not attempted, and writer not attempted.

## OLED Property Categories

Safe normalized categories include homo, lumo, plqy, delta_est, s1, t1,
emission_peak, absorption_peak, eqe, current_efficiency, power_efficiency,
luminance, lifetime, device_voltage, host, dopant, and unknown_property.

The harness counts categories only. It does not output values.

## Output Files

Outputs are written under a clean run directory named by the safe acceptance id:

```text
real_literature_read_only_acceptance_report.json
real_literature_read_only_acceptance_summary.json
redacted_real_literature_read_only_acceptance_evidence.md
```

The summary references the report by basename only and binds the report bytes
with SHA-256.

## Redaction Policy

Manifest content, parsed summary content, report JSON, summary JSON, and
Markdown evidence are scanned before output. Unsafe material causes fail-closed
behavior with a minimal blocked summary and no unsafe Markdown.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_real_literature_read_only_acceptance \
  --manifest real_literature_manifest.json \
  --parsed-output-root local_parsed_outputs \
  --output-dir acceptance_output \
  --max-papers 5 \
  --operator-id safe-operator-id
```

## Out of Scope

This harness is for local read-only acceptance only.
This harness does not commit PDFs.
This harness does not commit raw MinerU outputs.
This harness does not emit raw text.
This harness does not emit raw tables.
This harness does not emit raw values.
This harness does not emit paper titles in outputs.
This harness does not emit DOI strings in outputs.
This harness does not execute the controlled writer.
This harness does not create execution requests.
This harness does not run execution request preflight.
This harness does not explicitly confirm execution.
This harness does not materialize training datasets.
This harness does not create CSV/JSONL/Parquet/LMDB artifacts.
This harness does not generate conformers.
This harness does not generate DPA3 structures.
This harness does not run model training or evaluation.

It also does not read PDFs, run MinerU, call LLMs or agents, run corpus
workflows, create candidate rows, serialize training rows, modify
DatasetConfirmation, or perform chemistry calculations.

## Next Step

The next step is future real candidate quarantine dry-run, not writer
execution.
