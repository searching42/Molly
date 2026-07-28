# OLED exact-chain observation materialization candidate (PR-Q)

## Purpose

PR-Q closes the source-value gap left intentionally by PR-P. It rejoins every
PR-P resolved material/cell reference with the exact PR-K request, accepted
PR-I semantic adjudication, PR-J source-transcription review packet, and PR-J
source-transcription adjudication. It then constructs deterministic
`OledPropertyObservation` candidates with a stable Registry material ID.

This is observation-candidate materialization, not reviewed-evidence staging,
Gold conversion, dataset admission, or training-data creation.

## Exact inputs and hash bridges

The file entry requires five distinct regular JSON files:

1. PR-P `oled_observation_staging_preflight.v1`;
2. PR-K `oled_supplementary_material_identity_candidate_request.v1`;
3. PR-I `oled_supplementary_semantic_adjudication.v1`;
4. PR-J `oled_supplementary_source_transcription_review_packet.v1`; and
5. PR-J `oled_supplementary_source_transcription_adjudication.v1`.

All inputs are read without following symbolic path components. PR-Q checks
their exact file SHA-256 values against the downstream chain wherever that
chain carries an immutable byte binding:

```text
PR-P embedded PR-M -> exact PR-K file SHA-256
PR-K -> exact PR-I adjudication file SHA-256
PR-K -> exact PR-J review-packet file SHA-256
PR-K -> exact PR-J adjudication file SHA-256
PR-J adjudication -> exact PR-J review-packet file SHA-256
```

The semantic digests are also replayed across the same chain. A semantically
identical JSON file with different bytes is rejected when the downstream
artifact bound the original file SHA-256.

PR-Q also closes the cross-artifact causal chain that no individual artifact
can establish alone:

```text
PR-I adjudication generated_at
<= PR-J review packet generated_at
<= PR-J human reviewed_at
<= PR-J adjudication generated_at
<= PR-K generated_at
<= PR-M human reviewed_at
```

A fully rehashed chain whose artifacts remain standalone-valid is still
rejected if the PR-J packet predates its exact PR-I semantic input or if human
review predates the exact PR-J packet.

PR-Q embeds all five validated models so standalone validation can rederive
every candidate and its semantic bindings. Exact external input bytes cannot
be recovered from embedded models, so
`standalone_input_bytes_revalidation_supported=false` remains explicit.

## Candidate construction

Only a PR-P staging cell can become a candidate. For each such cell, PR-Q
requires all of the following:

- the PR-P identity group and dependent-cell roster exactly equal the bound
  PR-K group;
- the source-cell and disposition digests resolve to a PR-I cell whose known
  property mapping was accepted and remains materialization-eligible;
- the semantic review item ID/digest and column mapping match exactly;
- the source row, column, subject literal, property literal, table digest,
  PDF hash, parsed-document hash, and page all match the exact PR-J review
  packet;
- the corresponding PR-J adjudicated table was accepted as a bounded source
  transcription; and
- the source-cell digest remains in PR-J's later-identity-eligible roster.

The resulting candidate contains:

- the selected Registry material ID and complete Registry entry;
- the source-reported material name;
- exact PDF/table/page/row/column and review-item provenance;
- the complete PR-I source cell and accepted mapping summary;
- the numeric value plus exact `reported_value_text` and
  `reported_decimal_places`;
- the source-reported unit and canonical normalized unit;
- the accepted causal layer and comparison context;
- one supplementary source-cell evidence object; and
- the canonical observation produced by the existing layered OLED schema.

`OledPropertyObservation.property_label` uses the stable ontology property ID
so the existing schema canonicalizer can validate it. The source header is not
lost: it remains exact in `mapping_summary.property_label`, `column_name`, the
semantic source cell, and observation metadata.

Trailing zeros are evidence, not formatting noise. A reported value such as
`-1.70` is represented numerically as `-1.7` while retaining
`reported_value_text="-1.70"` and `reported_decimal_places=2`.

## Comparison-context boundary

PR-Q copies only the exact comparison context accepted through PR-I. It does
not infer temperature, host, concentration, sample form, excitation wavelength,
or lifetime fitting method.

The existing layered schema assesses every candidate as:

- `not_required` when the ontology does not require comparison context;
- `complete` when all required fields are present; or
- `incomplete` with an exact missing-field list.

An incomplete candidate is preserved but has `comparison_ready=false`. This
keeps evidence available without treating incomparable photophysical values as
cross-paper comparable. PR-I acceptance confirms the reviewed response
disposition; it does not turn absent context into a reported fact or establish
universal physical correctness.

## paper016-shaped automated boundary

The accepted fixture maps the `TDBA-Si` row to one existing Registry entry and
materializes five known-property candidates:

```text
5 PR-P staged cells
5 exact source-bound observation candidates
5 comparison-context-not-required candidates
14 ontology-review-pending cells still excluded
0 device-only cells admitted
```

Representative literals `-1.70`, `-5.50`, `3.30`, and `2.78` retain two
reported decimal places. The HOMO/LUMO labels and unusual reported ordering are
preserved as reviewed; PR-Q does not reinterpret them. The HOMO-LUMO-gap and
oscillator-strength columns remain outside this branch because their PR-I
outcome still requires ontology work.

The fixture proves deterministic chain behavior, not the real-world identity
of a paper016 material.

## Controlled workflow

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_observation_materialization_candidate \
  --staging-preflight /operator/local/observation_staging_preflight.json \
  --material-identity-request /operator/local/material_identity_candidate_request.json \
  --semantic-adjudication /operator/local/semantic_adjudication.json \
  --transcription-review-packet /operator/local/source_transcription_review_packet.json \
  --transcription-adjudication /operator/local/source_transcription_adjudication.json \
  --output /operator/local/observation_materialization_candidates.json
```

The output must be fresh and distinct from every input. Output-parent
descriptors are pinned through publication. Symlinked paths, duplicate input
paths, input overwrite, stale hashes, changed literals, causal timestamp
reversal across PR-I/PR-J/PR-K/PR-M, parent replacement, or invalid layered
observations fail without a partial output. CLI failures expose only a stable
redacted error object.

## Explicitly false after PR-Q

- source values corrected or inferred;
- ontology extensions applied;
- reviewed-evidence staging performed;
- direct admission eligibility granted;
- Registry or alias mutation;
- Gold, dataset, feature, or training writes;
- source PDF or raw parsed-document reads;
- device-only admission; and
- network, external-service, LLM, or MinerU calls.

## Next boundary

The next safe step is a reviewed-evidence staging preflight over the exact PR-Q
artifact. It should define how source-bound candidates enter a quarantine or
reviewed-evidence ledger, keep incomplete comparison context queryable and
non-comparison-ready, and continue excluding ontology-pending and device-only
records. It must not jump directly from PR-Q to Gold or a curated dataset.
