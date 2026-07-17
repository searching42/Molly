# Contextual alias resolution

PR-AN adds a bounded candidate-generation step for aliases that cannot identify a
chemical structure by themselves. It consumes an exact request and an exact
UTF-8 text extraction from one paper, locates a source-reported systematic-name
heading for each alias, resolves that name with the official OPSIN web service,
and independently canonicalizes the returned graph with RDKit/InChI.

This is deliberately not an identity-admission step. A successful result means
only that the paper text contains the heading and that OPSIN and RDKit agree on
the structure represented by that heading. It does not prove that the heading
matches a molecular diagram, CIF, Registry material, Gold entry, or dataset row.

## Input contract

The request uses `contextual_alias_resolution_request.v1` and binds:

- `run_id`, `paper_id`, and `source_document_id`;
- one safe relative parsed-text filename and its exact SHA-256;
- a sorted, unique list of `(candidate_id, reported_alias)` items.

The file runner rejects duplicate JSON keys, symbolic path components, changed
files, digest mismatches, and any request/text replacement between the initial
read and publication.

## Resolution profile

The `supplementary_heading_opsin_rdkit.v1` profile searches for the exact
`(alias):` heading marker. It joins at most four preceding wrapped lines and
removes only a line-ending hyphen at a join boundary. Zero matches produce
`alias_not_found`; multiple matches produce `alias_ambiguous`. Neither outcome
calls the resolver.

One match is submitted to the fixed official endpoint
`https://www.ebi.ac.uk/opsin/ws/{encoded-name}.json`. Redirects are disabled.
The exact response bytes, response SHA-256, endpoint, status, and extracted
fields are embedded in the result. Model validation replays those bindings.
For a successful response, RDKit independently derives canonical isomeric
SMILES, standard InChI, InChIKey, and molecular formula; any OPSIN/RDKit standard
identifier disagreement fails the whole invocation closed.

## Publication boundary

The artifact uses `contextual_alias_resolution_artifact.v1`. It embeds the
normalized request and self-validating per-result digests. Every artifact fixes:

- `candidate_only=true`;
- `source_match_validated=false`;
- `identity_resolved=false`;
- `registry_mutated=false`;
- `gold_written=false`;
- `dataset_written=false`.

Publication pins a symlink-free output parent, creates the final file with
`O_NOFOLLOW | O_CREAT | O_EXCL`, completes short writes, fsyncs the file and
parent, and verifies inode ownership, exact bytes, size, and model validity.
Existing outputs are never replaced, and cleanup removes only the inode created
by the current invocation.

## CLI

```bash
PYTHONPATH=src .venv/bin/python -m ai4s_agent.contextual_alias_resolution \
  --request /absolute/path/request.json \
  --output /absolute/path/contextual-alias-candidates.json
```

The default runner performs live HTTPS requests to OPSIN. Tests inject a fixed
resolver and do not depend on network availability.
