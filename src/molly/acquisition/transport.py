"""Bounded HTTPS transport with DNS validation and manual redirects.

The default transport connects to the exact IP address returned by the
validated resolver and uses the configured hostname for TLS SNI/certificate
verification.  This avoids the unsafe resolve-then-let-a-client-resolve-again
pattern.  Tests may inject a deterministic transport or connection factory;
the production path never inherits proxy or shell authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import random
import re
import socket
import ssl
from collections.abc import Callable, Iterable, Mapping
import threading
import time
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from .errors import AcquisitionPolicyError, AcquisitionTimeoutError, AcquisitionTransportError
from .models import MAX_REDIRECTS, ProviderConfig, ProviderRoute
from .policy import (
    CanonicalURL,
    canonicalize_url,
    normalize_content_type,
    parse_retry_after,
    resolve_redirect,
    sanitize_url,
    validate_resolved_addresses,
)


REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
RETRYABLE_STATUSES = frozenset({408, 429, *range(500, 600)})
_MAX_HEADER_BYTES = 64 * 1024
_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True, repr=False)
class NetworkResponse:
    """Bounded response data after one manually validated request chain."""

    status_code: int
    headers: Mapping[str, str | Iterable[str]]
    body: bytes
    requested_url: str
    resolved_url: str
    redirect_chain: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.status_code, bool) or not isinstance(self.status_code, int):
            raise AcquisitionTransportError("response status is malformed")
        if not isinstance(self.body, bytes):
            raise AcquisitionTransportError("response body must be bytes")
        collected: dict[str, list[str]] = {}
        for key, values in self.headers.items():
            if not isinstance(key, str):
                raise AcquisitionTransportError("response header name is malformed")
            normalized_key = key.casefold()
            if not re.fullmatch(r"[a-z0-9-]{1,128}", normalized_key):
                raise AcquisitionTransportError("response header name is malformed")
            if isinstance(values, str):
                items = (values,)
            else:
                items = tuple(values)
            if not items or any(not isinstance(item, str) for item in items):
                raise AcquisitionTransportError("response header value is malformed")
            if any(any(char in item for char in "\x00\r\n") for item in items):
                raise AcquisitionTransportError("response header value contains invalid data")
            collected.setdefault(normalized_key, []).extend(items)
        object.__setattr__(
            self,
            "headers",
            MappingProxyType({key: tuple(values) for key, values in collected.items()}),
        )
        object.__setattr__(self, "redirect_chain", tuple(self.redirect_chain))

    def header_values(self, name: str) -> tuple[str, ...]:
        return self.headers.get(name.casefold(), ())

    def header(self, name: str) -> str | None:
        values = self.header_values(name)
        return values[0] if values else None

    def __repr__(self) -> str:
        return (
            f"NetworkResponse(status_code={self.status_code}, "
            f"body_bytes={len(self.body)}, resolved_url=<sanitized>)"
        )


class DNSResolver(Protocol):
    def resolve(self, hostname: str, port: int) -> Iterable[str]:
        ...


class SystemDNSResolver:
    """Resolve only for the transport, never for model-provided input."""

    def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        results = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        return tuple(dict.fromkeys(item[4][0] for item in results))


class _Connection(Protocol):
    def sendall(self, data: bytes) -> None:
        ...

    def makefile(self, mode: str):
        ...

    def settimeout(self, value: float | None) -> None:
        ...

    def close(self) -> None:
        ...


def _default_connection_factory(
    address: str, route: ProviderRoute, timeout: float
) -> _Connection:
    ip = ipaddress.ip_address(address)
    family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
    raw = socket.socket(family, socket.SOCK_STREAM)
    raw.settimeout(timeout)
    try:
        raw.connect((address, route.port, 0, 0) if ip.version == 6 else (address, route.port))
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        # The IP is used for the TCP connection; the configured DNS hostname
        # is retained for certificate verification and SNI.
        return context.wrap_socket(raw, server_hostname=route.host)
    except Exception:
        raw.close()
        raise


@dataclass(slots=True)
class _RateGate:
    semaphore: threading.BoundedSemaphore
    lock: threading.Lock
    next_allowed: float = 0.0


class SafeNetworkTransport:
    """Synchronous, GET-only, size-bounded pinned HTTPS transport."""

    def __init__(
        self,
        *,
        resolver: DNSResolver | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        random_source: Callable[[], float] = random.random,
        connection_factory: Callable[[str, ProviderRoute, float], _Connection]
        | None = None,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.resolver = resolver or SystemDNSResolver()
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._random = random_source
        self._connection_factory = connection_factory or _default_connection_factory
        self._wall_clock = wall_clock
        self._gates: dict[tuple[str, str], _RateGate] = {}
        self._gates_lock = threading.Lock()

    def _gate_for(self, config: ProviderConfig, route: ProviderRoute) -> _RateGate:
        key = (config.provider_id, route.host)
        with self._gates_lock:
            gate = self._gates.get(key)
            if gate is None:
                gate = _RateGate(
                    semaphore=threading.BoundedSemaphore(config.max_concurrent_requests),
                    lock=threading.Lock(),
                )
                self._gates[key] = gate
            return gate

    def _acquire_rate_slot(
        self,
        config: ProviderConfig,
        route: ProviderRoute,
        *,
        deadline: float,
    ) -> _RateGate:
        gate = self._gate_for(config, route)
        if not gate.semaphore.acquire(timeout=self._remaining(deadline, self._monotonic)):
            raise AcquisitionTimeoutError(
                "acquisition total timeout exceeded while waiting for a slot"
            )
        try:
            while True:
                remaining = self._remaining(deadline, self._monotonic)
                now = self._monotonic()
                with gate.lock:
                    wait_for = max(0.0, gate.next_allowed - now)
                    if wait_for == 0:
                        gate.next_allowed = now + (1.0 / config.requests_per_second)
                        return gate
                if wait_for >= remaining:
                    raise AcquisitionTimeoutError(
                        "acquisition total timeout exceeded while waiting for rate limit"
                    )
                self._sleeper(wait_for)
        except Exception:
            gate.semaphore.release()
            raise

    @staticmethod
    def _release_rate_slot(gate: _RateGate) -> None:
        gate.semaphore.release()

    @staticmethod
    def _remaining(deadline: float, monotonic: Callable[[], float]) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise AcquisitionTimeoutError("acquisition total timeout exceeded")
        return remaining

    @staticmethod
    def _request_headers(route: ProviderRoute, headers: Mapping[str, str]) -> dict[str, str]:
        result: dict[str, str] = {"accept-encoding": "identity", "connection": "close"}
        allowed = set(route.allowed_header_names)
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise AcquisitionPolicyError("request headers must be text")
            normalized = key.casefold()
            if normalized in {"host", "content-length", "connection"}:
                raise AcquisitionPolicyError("transport-owned request header cannot be overridden")
            if normalized not in allowed:
                raise AcquisitionPolicyError("request header is not configured for this route")
            if any(char in value for char in "\x00\r\n"):
                raise AcquisitionPolicyError("request header contains a control character")
            try:
                value.encode("ascii")
            except UnicodeEncodeError as exc:
                raise AcquisitionPolicyError("request header must use ASCII text") from exc
            result[normalized] = value
        return result

    def _read_response(
        self,
        connection: _Connection,
        *,
        url: CanonicalURL,
        route: ProviderRoute,
        config: ProviderConfig,
        deadline: float,
    ) -> tuple[int, dict[str, tuple[str, ...]], bytes]:
        remaining = self._remaining(deadline, self._monotonic)
        connection.settimeout(min(config.read_timeout_seconds, remaining))
        reader = connection.makefile("rb")
        try:
            status_line = reader.readline(_MAX_HEADER_BYTES + 1)
            if not status_line or len(status_line) > _MAX_HEADER_BYTES:
                raise AcquisitionTransportError("HTTP status line is missing or too large")
            try:
                version, status_text, _ = status_line.decode("latin-1").rstrip("\r\n").split(" ", 2)
                status = int(status_text)
            except (UnicodeDecodeError, ValueError) as exc:
                raise AcquisitionTransportError("HTTP status line is malformed") from exc
            if version not in {"HTTP/1.0", "HTTP/1.1"} or not 100 <= status <= 599:
                raise AcquisitionTransportError("HTTP protocol/status is unsupported")

            header_values: dict[str, list[str]] = {}
            header_bytes = len(status_line)
            while True:
                line = reader.readline(_MAX_HEADER_BYTES + 1)
                header_bytes += len(line)
                if header_bytes > _MAX_HEADER_BYTES:
                    raise AcquisitionTransportError("HTTP headers exceed the bounded limit")
                if line in {b"\r\n", b"\n"}:
                    break
                if not line or b":" not in line:
                    raise AcquisitionTransportError("HTTP response header is malformed")
                raw_name, raw_value = line.split(b":", 1)
                try:
                    name = raw_name.decode("ascii").casefold()
                    value = raw_value.decode("latin-1").strip()
                except UnicodeDecodeError as exc:
                    raise AcquisitionTransportError("HTTP response header is not valid") from exc
                if not re_header_name(name) or any(char in value for char in "\x00\r\n"):
                    raise AcquisitionTransportError("HTTP response header contains invalid data")
                header_values.setdefault(name, []).append(value)

            headers = {name: tuple(values) for name, values in header_values.items()}
            transfer_encodings = tuple(
                value.casefold().strip() for value in headers.get("transfer-encoding", ())
            )
            if transfer_encodings and any(
                value not in {"identity", ""} for value in transfer_encodings
            ):
                raise AcquisitionPolicyError(
                    "Transfer-Encoding is not enabled; identity framing is required"
                )
            encodings = tuple(value.casefold().strip() for value in headers.get("content-encoding", ()))
            if encodings and any(value not in {"identity", ""} for value in encodings):
                raise AcquisitionPolicyError("compressed Content-Encoding is not enabled")

            lengths = headers.get("content-length", ())
            content_length: int | None = None
            if lengths:
                if len(set(lengths)) != 1 or not lengths[0].isdigit():
                    raise AcquisitionPolicyError("Content-Length is contradictory or malformed")
                content_length = int(lengths[0])
                if content_length > config.max_response_bytes:
                    raise AcquisitionPolicyError("Content-Length exceeds the configured response limit")

            body = bytearray()
            while True:
                remaining = self._remaining(deadline, self._monotonic)
                connection.settimeout(min(config.read_timeout_seconds, remaining))
                chunk = reader.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > config.max_response_bytes:
                    raise AcquisitionPolicyError("streamed response exceeds the configured response limit")
            if content_length is not None and len(body) != content_length:
                raise AcquisitionTransportError("response body length does not match Content-Length")
            return status, headers, bytes(body)
        except socket.timeout as exc:
            raise AcquisitionTransportError("acquisition read timed out") from exc
        finally:
            reader.close()

    def _request_once(
        self,
        url: CanonicalURL,
        *,
        route: ProviderRoute,
        config: ProviderConfig,
        headers: Mapping[str, str],
        secret_values: Iterable[str],
        deadline: float,
    ) -> NetworkResponse:
        addresses = validate_resolved_addresses(
            route.host, self.resolver.resolve(route.host, route.port)
        )
        address = addresses[0]
        connection: _Connection | None = None
        gate = self._acquire_rate_slot(config, route, deadline=deadline)
        try:
            timeout = min(config.connect_timeout_seconds, self._remaining(deadline, self._monotonic))
            connection = self._connection_factory(address, route, timeout)
            getpeername = getattr(connection, "getpeername", None)
            if callable(getpeername):
                peer = getpeername()
                peer_address = str(peer[0]) if isinstance(peer, tuple) and peer else str(peer)
                try:
                    normalized_peer = str(ipaddress.ip_address(peer_address))
                except ValueError as exc:
                    raise AcquisitionPolicyError("connected peer address is malformed") from exc
                if normalized_peer != address:
                    raise AcquisitionPolicyError("connected peer is outside the validated DNS result")
            request_headers = self._request_headers(route, headers)
            parsed = urlsplit(url.value)
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            request_lines = [
                f"GET {target} HTTP/1.1",
                f"Host: {route.host}" if route.port == 443 else f"Host: {route.host}:{route.port}",
            ]
            for key, value in request_headers.items():
                request_lines.append(f"{key}: {value}")
            connection.sendall(("\r\n".join(request_lines) + "\r\n\r\n").encode("ascii"))
            status, response_headers, body = self._read_response(
                connection,
                url=url,
                route=route,
                config=config,
                deadline=deadline,
            )
            return NetworkResponse(
                status_code=status,
                headers=response_headers,
                body=body,
                requested_url=url.value,
                resolved_url=url.value,
            )
        except AcquisitionTransportError:
            raise
        except AcquisitionPolicyError:
            raise
        except (OSError, ssl.SSLError, TimeoutError) as exc:
            raise AcquisitionTransportError(
                f"bounded HTTPS request failed for "
                f"{sanitize_url(url.value, secret_values=secret_values)}"
            ) from exc
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            self._release_rate_slot(gate)

    def _one_redirect_chain(
        self,
        initial: CanonicalURL,
        *,
        route: ProviderRoute,
        config: ProviderConfig,
        headers: Mapping[str, str],
        secret_values: Iterable[str],
        deadline: float,
    ) -> NetworkResponse:
        current = initial
        redirect_chain: list[str] = []
        visited = {current.value}
        current_headers = dict(headers)
        for _ in range(MAX_REDIRECTS + 1):
            response = self._request_once(
                current,
                route=route,
                config=config,
                headers=current_headers,
                secret_values=secret_values,
                deadline=deadline,
            )
            if response.status_code not in REDIRECT_STATUSES:
                if 200 <= response.status_code <= 299:
                    normalize_content_type(
                        response.header_values("content-type"),
                        allowed=route.accepted_content_types,
                    )
                return NetworkResponse(
                    status_code=response.status_code,
                    headers=response.headers,
                    body=response.body,
                    requested_url=initial.value,
                    resolved_url=current.value,
                    redirect_chain=tuple(redirect_chain),
                )
            location = response.header("location")
            if not location:
                raise AcquisitionPolicyError("redirect response has no Location")
            if len(redirect_chain) >= MAX_REDIRECTS:
                raise AcquisitionPolicyError("redirect limit exceeded")
            target = resolve_redirect(current.value, location, route)
            if target.value in visited:
                raise AcquisitionPolicyError("redirect loop detected")
            visited.add(target.value)
            redirect_chain.append(sanitize_url(target.value, secret_values=secret_values))
            # The route currently has an exact host binding.  Keep this
            # defensive branch so a future multi-host route cannot forward
            # authorization/cookie headers across authorities.
            if target.hostname != current.hostname:
                current_headers = {
                    key: value
                    for key, value in current_headers.items()
                    if key.casefold() not in {"authorization", "cookie", "proxy-authorization", "x-api-key"}
                }
            current = target
        raise AcquisitionPolicyError("redirect limit exceeded")

    def fetch(
        self,
        url: str,
        *,
        route: ProviderRoute,
        config: ProviderConfig,
        headers: Mapping[str, str] = {},
        secret_values: Iterable[str] = (),
    ) -> NetworkResponse:
        """Fetch one exact route with bounded retries and manual redirects."""

        if route.provider_id and route.provider_id != config.provider_id:
            raise AcquisitionPolicyError("route/provider binding mismatch")
        initial = canonicalize_url(
            url,
            allowed_hosts=(route.host,),
            allowed_port=route.port,
            path_prefix=route.path_prefix or "/",
            allowed_query_fields=route.allowed_query_fields,
        )
        if route.allowed_methods != ("GET",):
            raise AcquisitionPolicyError("only GET is supported by the acquisition transport")
        retry_count = 0
        deadline = self._monotonic() + config.total_timeout_seconds
        while True:
            try:
                response = self._one_redirect_chain(
                    initial,
                    route=route,
                    config=config,
                    headers=headers,
                    secret_values=secret_values,
                    deadline=deadline,
                )
            except (AcquisitionPolicyError, AcquisitionTransportError) as exc:
                if isinstance(exc, AcquisitionPolicyError):
                    raise
                if isinstance(exc, AcquisitionTimeoutError):
                    raise
                if retry_count >= config.retry_policy.maximum_retries:
                    raise
                delay = self._retry_delay(config, retry_count, None)
                retry_count += 1
                if delay >= self._remaining(deadline, self._monotonic):
                    raise AcquisitionTimeoutError(
                        "acquisition total timeout exceeded before retry"
                    )
                self._sleeper(delay)
                continue
            if response.status_code not in RETRYABLE_STATUSES:
                return response
            if retry_count >= config.retry_policy.maximum_retries:
                return response
            retry_after = parse_retry_after(
                response.header("retry-after"), now=self._wall_clock()
            )
            delay = self._retry_delay(config, retry_count, retry_after)
            retry_count += 1
            if delay >= self._remaining(deadline, self._monotonic):
                raise AcquisitionTimeoutError("acquisition total timeout exceeded before retry")
            self._sleeper(delay)

    def _retry_delay(
        self, config: ProviderConfig, retry_index: int, retry_after: float | None
    ) -> float:
        policy = config.retry_policy
        exponential = min(
            policy.maximum_delay_seconds,
            policy.initial_delay_seconds * (2**retry_index),
        )
        jitter = exponential * policy.jitter_fraction * max(0.0, min(1.0, self._random()))
        bounded = min(policy.maximum_delay_seconds, exponential + jitter)
        if retry_after is not None:
            bounded = max(exponential, retry_after)
        return min(policy.maximum_delay_seconds, policy.maximum_retry_after_seconds, bounded)


def re_header_name(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9-]{1,128}", value))


class NetworkTransport(Protocol):
    def fetch(
        self,
        url: str,
        *,
        route: ProviderRoute,
        config: ProviderConfig,
        headers: Mapping[str, str] = {},
        secret_values: Iterable[str] = (),
    ) -> NetworkResponse:
        ...


__all__ = [
    "DNSResolver",
    "NetworkResponse",
    "NetworkTransport",
    "SafeNetworkTransport",
    "SystemDNSResolver",
]
