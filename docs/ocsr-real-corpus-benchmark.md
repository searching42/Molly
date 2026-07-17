# OCSR real-corpus benchmark

This boundary evaluates immutable OCSR candidate artifacts against independently
reviewed molecular-structure truth from real papers. It measures whether a
candidate copied the source graph; it does not resolve material identity or
publish Registry, Gold, or dataset state.

## Contracts

The ground-truth manifest uses
`ocsr_real_corpus_ground_truth_manifest.v1`. Every sample binds:

- one candidate by `run_id`, `candidate_id`, reported alias, and exact crop
  SHA-256;
- the exact source document and locator containing the molecular diagram;
- an exact reference document and locator supporting the reviewed structure;
- the source-reported systematic name or structure literal used as the
  independent reference;
- the resolver and version used to translate that reference;
- RDKit-canonical isomeric SMILES and InChIKey; and
- named reviewer, review time, review note, and an explicit confirmation that
  the independently derived graph matches the source depiction.

The source diagram and structure reference may be the same document or two
different documents. For example, a main-paper figure can be bound as
`source_diagram` while systematic names and characterization in supporting
information are separately bound as `structure_reference`. The evaluator
derives these roles from the manifest and rejects missing, extra, incorrectly
hashed, cross-paper, or incorrectly assigned document bindings.

Both the per-sample payload and whole truth manifest are content-digested. The
evaluator additionally re-canonicalizes every truth graph and re-derives its
InChIKey before comparison. A digest-valid but chemically noncanonical truth
manifest therefore fails closed.

## Outcomes and metrics

Every requested candidate must have exactly one truth sample. Missing or extra
candidates fail the entire benchmark instead of silently changing the
denominator.

The evaluator assigns one of three outcomes:

- `exact_match`: a ready candidate has the exact reviewed InChIKey;
- `wrong_graph`: a ready, parseable candidate has a different InChIKey; or
- `false_rejection`: the candidate executor rejected an image whose reviewed
  graph is known.

The report records exact-InChIKey accuracy, ready and rejection rates,
false-ready rate, molecular-formula agreement, per-paper summaries, and
confidence summaries split by exact and wrong ready graphs. Model confidence is
descriptive only; it cannot turn a wrong graph into an accepted result.

The report contract re-derives all counts and rates, evidence-cardinality
counts, per-paper summaries, confidence summaries, internal artifact/result
binding consistency, document-paper role coverage, and its outer report digest.
It does not claim that parsing the report alone can reconstruct hashes of
external files or models that are not embedded in the report.

## Exact-input replay verification

An independently verified result requires the report plus the exact truth
manifest, candidate artifacts, and source/reference documents. Verification
re-reads all files without following symlinks, rebuilds the complete benchmark
with the persisted report timestamp, and requires exact model equality with the
persisted report. This independently reconstructs candidate-result and truth
sample digests, manifest file SHA/digest, checkpoint SHA, source-document
SHA/size, all outcomes, metrics, and the report digest.

Only after exact replay succeeds does the verifier publish an
`ocsr_real_corpus_benchmark_verification.v1` artifact. The receipt binds the
exact report bytes and all replayed upstream bindings. Re-verification still
requires the exact upstream files; parsing either the report or receipt alone
is not represented as external provenance verification.

## Corpus-size claim

The report is a `bounded_real_paper_canary` until it contains at least three
distinct source-document SHA values and twenty distinct items in each of these
dimensions:

- exact crop SHA-256;
- source-document SHA plus source locator; and
- source-document SHA plus source locator plus crop SHA-256.

One source-document SHA may not bind multiple paper IDs; that is a provenance
conflict and fails the evaluation. Duplicate crops or locators may remain in a
bounded diagnostic report, but they are deduplicated for the scale claim. A
repeated molecule counts again only when it is an independently located source
diagram with different crop bytes. Only evidence satisfying the distinct-paper
and distinct-diagram thresholds is labeled `real_corpus_benchmark`.

## Run

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.ocsr_real_corpus_benchmark \
  --ground-truth /operator/benchmark/ground_truth.json \
  --candidate-artifact /operator/benchmark/paper001_candidates.json \
  --candidate-artifact /operator/benchmark/paper002_candidates.json \
  --source-document paper001-main=/corpus/paper001.pdf \
  --source-document paper001-si=/corpus/paper001_si.pdf \
  --source-document paper002-main=/corpus/paper002.pdf \
  --output /operator/benchmark/report.json
```

Independently replay the persisted report and publish a verification artifact:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.ocsr_real_corpus_benchmark \
  --verify-report /operator/benchmark/report.json \
  --ground-truth /operator/benchmark/ground_truth.json \
  --candidate-artifact /operator/benchmark/paper001_candidates.json \
  --candidate-artifact /operator/benchmark/paper002_candidates.json \
  --source-document paper001-main=/corpus/paper001.pdf \
  --source-document paper001-si=/corpus/paper001_si.pdf \
  --source-document paper002-main=/corpus/paper002.pdf \
  --verification-output /operator/benchmark/verified.json
```

Before any input is read, the output parent is opened without following
symlinks and pinned by directory descriptor. Inputs are read as exact regular
files without following symlinks. The report is published relative to the
pinned descriptor with `O_NOFOLLOW | O_CREAT | O_EXCL`, a complete-write loop,
file and parent `fsync`, exact-byte readback, inode checks, and full model
revalidation. Verification artifacts use the same publication guarantees.
Cleanup only removes the invocation-owned inode.

The command never modifies candidate artifacts and always records the following
boundaries as false:

- material identity resolution;
- Registry mutation;
- Gold publication; and
- dataset publication.

The initial paper018 run is documented in
[`evidence/ocsr-paper018-real-corpus-canary-20260717.md`](evidence/ocsr-paper018-real-corpus-canary-20260717.md).
