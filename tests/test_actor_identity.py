from __future__ import annotations

from flask import Flask, g, request

from ai4s_agent.actor_identity import resolve_actor, resolve_authenticated_actor


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


def test_resolve_actor_accepts_confirmed_by_alias_for_memory_payloads() -> None:
    json_actor = _resolve_with_request(json={"confirmed_by": "memory-reviewer"})
    form_actor = _resolve_with_request(data={"confirmed_by": "form-reviewer"}, json=None)

    assert json_actor.actor == "memory-reviewer"
    assert json_actor.source == "json:confirmed_by"
    assert form_actor.actor == "form-reviewer"
    assert form_actor.source == "form:confirmed_by"


def test_resolve_actor_missing_required_records_required_context() -> None:
    actor = _resolve_with_request(json={}, method="POST")

    assert actor.actor == ""
    assert actor.source == "missing"
    assert actor.required is False

    required_actor = _resolve_with_request(json={}, method="POST", required=True)
    assert required_actor.actor == ""
    assert required_actor.source == "missing"
    assert required_actor.required is True


def test_authenticated_actor_rejects_all_client_assertions_by_default() -> None:
    app = Flask(__name__)
    with app.test_request_context(
        "/actor?actor=query-user",
        method="POST",
        headers={"X-Actor": "header-user"},
        json={"actor": "body-user"},
    ):
        actor = resolve_authenticated_actor(request, required=True)
    assert actor.actor == ""
    assert actor.source == "missing"
    assert actor.required is True


def test_authenticated_actor_uses_middleware_environ_or_fixed_server_owner() -> None:
    app = Flask(__name__)
    app.config["AI4S_AGENT_AUTHORIZATION_OWNER"] = "configured-owner"
    with app.test_request_context("/actor", headers={"X-Actor": "spoofed"}):
        configured = resolve_authenticated_actor(request, required=True)
        request.environ["ai4s.authenticated_principal"] = "proxy-principal"
        environ = resolve_authenticated_actor(request, required=True)
        g.ai4s_authenticated_principal = "middleware-principal"
        middleware = resolve_authenticated_actor(request, required=True)

    assert configured.actor == "configured-owner"
    assert configured.source == "config:AI4S_AGENT_AUTHORIZATION_OWNER"
    assert environ.actor == "proxy-principal"
    assert environ.source == "wsgi.environ:ai4s.authenticated_principal"
    assert middleware.actor == "middleware-principal"
    assert middleware.source == "flask.g:ai4s_authenticated_principal"
