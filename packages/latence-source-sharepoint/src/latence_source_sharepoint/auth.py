"""Authentication for the SharePoint Source — the adopter's mechanism, secrets env-only.

The framework never owns a SharePoint credential. It reads one, at run time, from the
process environment the adopter populated with *their* secret manager (Key Vault CSI driver,
Vault agent, a CI secret, ``direnv`` — the framework does not care which), uses it for the
duration of the run, and persists nothing. Two real mechanisms are supported, which is why
:class:`TokenProvider` is a seam:

* :class:`EnvTokenProvider` — the adopter already mints app tokens (MSAL, a sidecar, a
  workload-identity broker) and drops the current bearer token into an env var. The framework
  does no OAuth at all.
* :class:`ClientCredentialsTokenProvider` — the adopter registered an Entra ID app and hands
  the framework the *names* of the env vars holding its client id and client secret; the
  connector performs the standard OAuth 2.0 client-credentials exchange and caches the token
  until shortly before it expires.

**The env-only posture is enforced, not merely documented.** :func:`reject_inline_secrets`
scans the Provider config (recursively) for a key that looks like it holds a credential and
raises :class:`~latence_core.errors.ConfigError`. A stack YAML therefore *cannot* carry a
token, which is what keeps it out of ``manifest.json``, out of git, and out of a bug report.
Config names env VARS (``token_env: SHAREPOINT_TOKEN``); it never names values.

Nothing in this module puts a secret into an exception message, a repr, or a log line — the
offline suite asserts that with the real secret value, not a flag.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_core.errors import ConfigError, ProviderError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from latence_source_sharepoint.transport import HttpTransport

# The default Entra ID token endpoint host.
DEFAULT_LOGIN_BASE_URL = "https://login.microsoftonline.com"

# The Graph application-permission scope for client-credentials.
GRAPH_DEFAULT_SCOPE = "https://graph.microsoft.com/.default"

# Refresh this many seconds BEFORE the token's stated expiry, so a long listing pass never
# trips over an expiry mid-flight.
_EXPIRY_SKEW_SECONDS = 120.0

# Config keys that would mean an operator inlined a credential. Checked against the LOWERCASED
# key, at every nesting depth. Deliberately broad: a false positive costs a rename, a false
# negative costs a leaked tenant.
_SECRET_CONFIG_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "bearer_token",
        "certificate",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)


def reject_inline_secrets(config: Mapping[str, Any], *, _path: str = "") -> None:
    """Raise if the config carries anything that looks like an inlined credential.

    Recurses into nested mappings and lists so ``auth: {client_secret: ...}`` is caught as
    surely as a top-level key. The message names the offending *key path* only — never the
    value, which is the thing we are protecting.
    """
    offenders: list[str] = []
    for key, value in config.items():
        path = f"{_path}{key}"
        if str(key).lower() in _SECRET_CONFIG_KEYS:
            offenders.append(path)
        elif isinstance(value, dict):
            reject_inline_secrets(value, _path=f"{path}.")
        elif isinstance(value, list):
            for index, entry in enumerate(value):
                if isinstance(entry, dict):
                    reject_inline_secrets(entry, _path=f"{path}[{index}].")
    if offenders:
        msg = (
            "SharePointSource config carries inlined credential key(s) "
            f"{sorted(offenders)!r}. Secrets are environment-only: the framework persists no "
            "credential and a stack file must never contain one. Name the environment "
            "variable instead (e.g. token_env: SHAREPOINT_TOKEN, or "
            "client_secret_env: SHAREPOINT_CLIENT_SECRET)."
        )
        raise ConfigError(msg)


@runtime_checkable
class TokenProvider(Protocol):
    """Supply the current bearer token for a Graph request."""

    def token(self, *, force_refresh: bool = False) -> str:
        """The bearer token to send. ``force_refresh`` discards any cached value first.

        :class:`~latence_source_sharepoint.graph.GraphClient` sets ``force_refresh`` after a
        ``401``, so a token that expired mid-run is renewed once rather than failing the run.
        """
        ...


def _read_env(env: Mapping[str, str], name: str, *, what: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        msg = (
            f"SharePointSource: environment variable {name!r} is unset or empty, so no "
            f"{what} is available. Secrets are environment-only — export it from your secret "
            "manager before running the pipeline; do not put it in the stack file."
        )
        raise ConfigError(msg)
    return value


class EnvTokenProvider:
    """Read a ready-made bearer token from an environment variable, every time it is needed.

    Re-reads on every call rather than caching, so an adopter whose sidecar rewrites the
    variable mid-run (a rotating token) is picked up without restarting the pipeline.
    """

    def __init__(self, *, env_var: str, env: Mapping[str, str] | None = None) -> None:
        self._env_var = env_var
        self._env = os.environ if env is None else env

    def token(self, *, force_refresh: bool = False) -> str:
        return _read_env(self._env, self._env_var, what="SharePoint access token")


class ClientCredentialsTokenProvider:
    """OAuth 2.0 client-credentials against Entra ID, with the secret read from the environment.

    The client id and client secret are read from *named* environment variables at the moment
    of the exchange. The resulting token is cached in memory only, until
    :data:`_EXPIRY_SKEW_SECONDS` before its stated expiry, and is never written anywhere.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        client_id_env: str,
        client_secret_env: str,
        transport: HttpTransport,
        scope: str = GRAPH_DEFAULT_SCOPE,
        login_base_url: str = DEFAULT_LOGIN_BASE_URL,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not tenant_id:
            msg = (
                "SharePointSource: auth 'client_credentials' requires a non-empty 'tenant_id' "
                "(the Entra ID directory id). A tenant id is not a secret."
            )
            raise ConfigError(msg)
        self._tenant_id = tenant_id
        self._client_id_env = client_id_env
        self._client_secret_env = client_secret_env
        self._transport = transport
        self._scope = scope
        self._login_base_url = login_base_url.rstrip("/")
        self._env = os.environ if env is None else env
        self._clock = clock
        self._cached: str | None = None
        self._expires_at = 0.0

    def token(self, *, force_refresh: bool = False) -> str:
        if force_refresh:
            self._cached = None
        if self._cached is not None and self._clock() < self._expires_at:
            return self._cached
        self._cached, lifetime = self._exchange()
        self._expires_at = self._clock() + max(lifetime - _EXPIRY_SKEW_SECONDS, 0.0)
        return self._cached

    def _exchange(self) -> tuple[str, float]:
        client_id = _read_env(self._env, self._client_id_env, what="SharePoint client id")
        client_secret = _read_env(
            self._env, self._client_secret_env, what="SharePoint client secret"
        )
        url = f"{self._login_base_url}/{self._tenant_id}/oauth2/v2.0/token"
        form = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": self._scope,
            }
        ).encode("ascii")
        response = self._transport.request(
            "POST",
            url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=form,
        )
        if response.status != 200:
            # Entra ID's error body echoes the request, which can include the client id but
            # never the secret; even so only the machine-readable code is surfaced, so a
            # rejected exchange cannot spill a request body into a log.
            raise ProviderError(self._exchange_failure_message(response.body, response.status))
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            msg = (
                "SharePointSource: the Entra ID token endpoint returned a non-JSON body for "
                "the client-credentials exchange."
            )
            raise ProviderError(msg) from exc
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            msg = (
                "SharePointSource: the Entra ID token endpoint returned no 'access_token' for "
                "the client-credentials exchange."
            )
            raise ProviderError(msg)
        raw_expiry = payload.get("expires_in", 3600)
        lifetime = float(raw_expiry) if isinstance(raw_expiry, int | float) else 3600.0
        return token, lifetime

    def _exchange_failure_message(self, body: bytes, status: int) -> str:
        code = "unknown_error"
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            code = payload["error"]
        return (
            f"SharePointSource: the Entra ID client-credentials exchange failed "
            f"(HTTP {status}, error {code!r}) for tenant {self._tenant_id!r}. Check the app "
            f"registration and the values of {self._client_id_env}/{self._client_secret_env}."
        )


def build_token_provider(
    config: Mapping[str, Any],
    *,
    transport: HttpTransport,
    env: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> TokenProvider:
    """Construct the :class:`TokenProvider` an ``auth`` config block asks for.

    ``{"mode": "env_token", "token_env": "SHAREPOINT_TOKEN"}`` (the default) or
    ``{"mode": "client_credentials", "tenant_id": ..., "client_id_env": ...,
    "client_secret_env": ...}``. An unknown mode is a :class:`ConfigError`, never a silent
    fallback to a weaker mechanism.
    """
    mode = str(config.get("mode", "env_token")).lower()
    if mode == "env_token":
        return EnvTokenProvider(
            env_var=str(config.get("token_env", "LATENCE_SHAREPOINT_TOKEN")), env=env
        )
    if mode == "client_credentials":
        return ClientCredentialsTokenProvider(
            tenant_id=str(config.get("tenant_id", "")),
            client_id_env=str(config.get("client_id_env", "LATENCE_SHAREPOINT_CLIENT_ID")),
            client_secret_env=str(
                config.get("client_secret_env", "LATENCE_SHAREPOINT_CLIENT_SECRET")
            ),
            transport=transport,
            scope=str(config.get("scope", GRAPH_DEFAULT_SCOPE)),
            login_base_url=str(config.get("login_base_url", DEFAULT_LOGIN_BASE_URL)),
            env=env,
            clock=clock,
        )
    msg = (
        f"SharePointSource: unknown auth mode {mode!r}. Supported: 'env_token' (the adopter "
        "mints the token and exports it) or 'client_credentials' (Entra ID app registration, "
        "with the client id/secret read from named environment variables)."
    )
    raise ConfigError(msg)
