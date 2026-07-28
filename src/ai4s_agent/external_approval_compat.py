from __future__ import annotations


def install_external_approval_error_compat() -> None:
    import ai4s_agent.external_approvals as external_approvals

    def normalize_external_approval_error(message: str) -> str:
        return str(message).replace(
            "user_approved_external_search=True",
            "user_approved_external_evidence=True (legacy: user_approved_external_search=True)",
        )

    external_approvals._normalize_external_approval_error = normalize_external_approval_error  # type: ignore[attr-defined]
