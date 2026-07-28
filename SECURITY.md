# Security Policy

## Reporting a vulnerability or privacy issue

Do not open a public issue for suspected credentials, personal data, private
infrastructure details, or an exploitable security defect. Use GitHub's private
vulnerability reporting for this repository.

Include only the minimum information needed to reproduce the problem. Do not
paste live credentials, private datasets, user content, or machine-specific
configuration into a report.

## Repository privacy boundary

This public repository must contain only source code, public documentation,
synthetic fixtures, and sanitized evidence. Runtime state, real user/project
data, private papers, secrets, hostnames, usernames, absolute infrastructure
paths, and machine-specific profiles must remain in user-scoped private storage.

The supported development line is `main`.

## Optional exact-value audit

Public tests use generic rules for credential formats, private configuration
files, non-example user directories and network identities, and
infrastructure-shaped hostnames. They intentionally do not contain a literal,
encoded, publicly salted, or unkeyed-digest list of retired private values.

Authorized maintainers may perform an additional exact-value scan with a
newline-delimited denylist stored outside the checkout or under an explicitly
Git-ignored path. Blank lines and lines beginning with `#` are ignored. Never
commit the denylist or paste its contents into an issue, PR, test fixture, or
CI variable visible to untrusted jobs.

```bash
MOLLY_PRIVATE_DENYLIST_PATH=/path/outside/checkout/private-denylist.txt \
  python scripts/audit_private_denylist.py
```

The scanner compares those private literals with current `git ls-files`
content and paths. Reports identify only the denylist entry number and matched
tracked file; they do not echo or hash the private value. If the environment
variable is unset, the optional command exits successfully without running an
exact-value scan. Public CI does not depend on a private denylist.

## Public Git history limitation

Working-tree cleanup affects only the current tracked tree. Deleting a file or
string in a new commit does not remove it from existing public Git objects,
clones, caches, or mirrors. Any public-history cleanup requires a separately
coordinated repository rewrite or rebuild and explicit downstream remediation.
Until that work is reviewed and completed, repository privacy risk must not be
described as closed, and `R11-001` retains its existing status in `todo.md`.
