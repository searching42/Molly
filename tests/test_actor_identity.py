from __future__ import annotations

from flask import Flask, request

from ai4s_agent.actor_identity import resolve_actor


def _resolve_with_request(*, headers=None, json=None, data=None, query_string=None, method: str = "POST", required: bool = False):
    app = Flask(__name__)
    with app.test_request_context(
        "/actor",
        method=method,
        headers=headers or {},
        json=json,
        data=data,
        query_string=query_string,
    ):
        return resolve_actor(request, required=required)


def test_resolve_actor_prefers_x_actor_over_json_body() -> None:
    actor = _resolve_with_request(headers={"X-Actor": "header-user"}, json={"actor": "body-user"})

    assert actor.actor == "header-user"
    assert actor.source == "header:X-Actor"
    assert actor.required is False


def test_resolve_actor_uses_json_form_and_query_sources() -> None:
    assert _resolve_with_request(json={"approved_by": "json-approver"}).source == "json:approved_by"
    assert _resolve_with_request(json={"revoked_by": "json-revoker"}).source == "json:revoked_by"
    assert _resolve_with_request(data={"actor": "form-user"}, json=None).source == "form:actor"
    assert _resolve_with_request(query_string={"actor": "query-user"}, json=None, method="GET").source == "query:actor"


def test_resolve_actor_missing_required_records_required_context() -> None:
    actor = _resolve_with_request(json={}, method="POST")

    assert actor.actor == ""
    assert actor.source == "missing"
    assert actor.required is False

    required_actor = _resolve_with_request(json={}, method="POST", required=True)
    assert required_actor.actor == ""
    assert required_actor.source == "missing"
    assert required_actor.required is True
