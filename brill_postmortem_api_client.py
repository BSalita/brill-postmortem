"""HTTP client for the postmortem-brill API.

The Streamlit app uses this instead of importing the library. Configure with
BRILL_POSTMORTEM_API_BASE_URL (default http://127.0.0.1:8531).
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, Optional

import polars as pl
import requests

BRILL_POSTMORTEM_API_BASE_URL = os.environ.get(
    "BRILL_POSTMORTEM_API_BASE_URL", "http://127.0.0.1:8531"
).rstrip("/")
_TIMEOUT_S = 300
# Generate fetches the field, then runs double and single dummy on every deal.
_GENERATE_TIMEOUT_S = 960


class BrillApiClientError(RuntimeError):
    def __init__(self, detail: str, status_code: Optional[int] = None):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _raw_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout_s: float = _TIMEOUT_S,
) -> requests.Response:
    url = f"{BRILL_POSTMORTEM_API_BASE_URL}{path}"
    try:
        resp = requests.request(
            method,
            url,
            params={k: v for k, v in (params or {}).items() if v is not None},
            json=json,
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        raise BrillApiClientError(
            f"postmortem-brill API unreachable at {BRILL_POSTMORTEM_API_BASE_URL}: {exc}"
        ) from exc
    if not resp.ok:
        try:
            body = resp.json()
            detail = body.get("detail") or resp.text
        except ValueError:
            detail = resp.text
        raise BrillApiClientError(str(detail), status_code=resp.status_code)
    return resp


def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout_s: float = _TIMEOUT_S,
) -> Any:
    return _raw_request(method, path, params=params, json=json, timeout_s=timeout_s).json()


def health() -> Dict[str, Any]:
    return _request("GET", "/health", timeout_s=2)


def tournaments(limit: int = 10, include_open: bool = False) -> Dict[str, Any]:
    return _request(
        "GET", "/brill/tournaments", {"limit": limit, "include_open": include_open}, timeout_s=60
    )


def games(limit: int = 100) -> Dict[str, Any]:
    return _request("GET", "/brill/games", {"limit": limit})


def generate(tournament_id: str, sd_samples: int = 10, force: bool = False) -> Dict[str, Any]:
    return _request(
        "POST",
        "/brill/generate",
        json={"tournament_id": tournament_id, "sd_samples": sd_samples, "force": force},
        timeout_s=_GENERATE_TIMEOUT_S,
    )


def entrants(key: str) -> Dict[str, Any]:
    """Standings of one tournament, and its winner."""
    return _request("GET", "/brill/entrants", {"tournament": key}, timeout_s=60)


def boards(key: str, columns: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
    return _request("GET", "/brill/boards", {"key": key, "columns": columns, "limit": limit})


def sql(sql: str, key: str, limit: int = 500) -> Dict[str, Any]:
    return _request("POST", "/brill/sql", json={"key": key, "sql": sql, "limit": limit})


def schema(key: str, pattern: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    return _request("GET", "/brill/schema", {"key": key, "pattern": pattern, "limit": limit})


def postmortem_dataframe(key: str) -> pl.DataFrame:
    resp = _raw_request("GET", "/brill/parquet", {"key": key})
    return pl.read_parquet(io.BytesIO(resp.content))


def pbn_text(key: str) -> str:
    return _raw_request("GET", "/brill/pbn", {"key": key}).text
