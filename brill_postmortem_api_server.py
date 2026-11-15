"""First-party REST API for postmortem-brill.

Lists Brill Bridge Challenge tournaments, and fetches, augments, and caches
one on request. Streamlit is an HTTP client of this process; only this
process signs in to the Brill site.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

import brill_client
import brill_postmortem_create as create
import brill_postmortem_service as service


app = FastAPI(title="postmortem-brill API", version="1.0.0")


class SqlRequest(BaseModel):
    sql: str
    tournament: Optional[str] = None
    key: Optional[str] = None  # same as tournament
    entrant: Optional[str] = None
    limit: int = 500


class GenerateRequest(BaseModel):
    tournament_id: str
    sd_samples: int = 10
    force: bool = False


def _run(callable_, /, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except brill_client.BrillAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except brill_client.BrillError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (FileNotFoundError, KeyError, ValueError) as exc:
        detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        raise HTTPException(status_code=422, detail=str(detail)) from exc
    except duckdb.Error as exc:
        raise HTTPException(status_code=422, detail=f"SQL error: {exc}") from exc


_GENERATE_TIMEOUT_S = 900


def _generate_isolated(tournament_id: str, sd_samples: int = 10, force: bool = False) -> dict:
    """Run generate in a child process so a native crash cannot kill uvicorn."""
    here = pathlib.Path(__file__).resolve().parent
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        meta_path = pathlib.Path(handle.name)
    try:
        command = [
            sys.executable,
            str(here / "brill_postmortem_create.py"),
            tournament_id,
            "--sd-samples",
            str(sd_samples),
            "--meta-out",
            str(meta_path),
        ]
        if force:
            command.append("--force")
        try:
            proc = subprocess.run(
                command,
                cwd=str(here),
                capture_output=True,
                text=True,
                timeout=_GENERATE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Generate timed out after {_GENERATE_TIMEOUT_S}s for tournament {tournament_id}"
            ) from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise RuntimeError(detail[-4000:] if len(detail) > 4000 else detail)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    finally:
        meta_path.unlink(missing_ok=True)
    # The child reports expected failures (sign-in, not started, ...) as data.
    if "error" in meta:
        raise HTTPException(status_code=int(meta.get("status") or 500), detail=meta["error"])
    return meta


# The sidebar asks for the list on every rerun; the site need not hear each one.
_TOURNAMENTS_TTL_S = 300
_tournaments_cache: Dict[Any, Any] = {}
_tournaments_lock = threading.Lock()


def _list_tournaments(limit: int, include_open: bool) -> List[Dict[str, Any]]:
    cache_key = (limit, include_open)
    with _tournaments_lock:
        hit = _tournaments_cache.get(cache_key)
        if hit and time.time() - hit[0] < _TOURNAMENTS_TTL_S:
            return hit[1]
    rows = brill_client.list_tournaments(limit=limit, include_open=include_open)
    with _tournaments_lock:
        _tournaments_cache[cache_key] = (time.time(), rows)
    return rows


# How far back a tournament can be named. Four events close each morning.
_RESOLVE_LIMIT = 60
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _tournament_frame(tournament: Optional[str] = None):
    """(df, meta) for a cache key, tournament id, or name; the latest closed
    tournament when omitted. A tournament not cached yet is fetched and built."""
    text = (tournament or "").strip()
    cached = service.find_cached(text) if text else None
    if cached is not None and cached.get("closed"):
        return service.load_postmortem(cached["key"])
    listing = _list_tournaments(_RESOLVE_LIMIT, False)
    if not text or text.lower() == "latest":
        if not listing:
            raise ValueError("No closed Brill tournament was found.")
        row = listing[0]
    else:
        row = service.match_tournament(listing, text)
        if row is None and _UUID_RE.match(text):
            row = {"id": text}
        if row is None:
            names = ", ".join(str(t.get("name")) for t in listing[:8])
            raise ValueError(
                f"No Brill tournament matches {tournament!r}. Pass an id or name from "
                f"/brill/tournaments. Latest: {names}"
            )
    cached = service.find_cached(row["id"])
    if cached is not None and cached.get("closed"):
        return service.load_postmortem(cached["key"])
    try:
        meta = _generate_isolated(row["id"])
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return service.load_postmortem(meta["key"])


def _entrant_frame(tournament: Optional[str], entrant: Optional[str]):
    """(df, meta) flagged for one entrant, or for the whole field."""
    frame, meta = _tournament_frame(tournament)
    frame, who = service.personalize(frame, entrant)
    return frame, {**meta, **who}


@app.get("/health")
def health() -> dict:
    info = _run(service.dataset_info)
    return {
        "status": "ok",
        "service": "postmortem-brill-api",
        "site": brill_client.SITE_URL,
        "credentials_configured": brill_client.credentials_configured(),
        **info,
    }


@app.get("/brill/dataset-info")
def dataset_info() -> dict:
    return {
        **_run(service.dataset_info),
        "site": brill_client.SITE_URL,
        "credentials_configured": brill_client.credentials_configured(),
    }


@app.get("/brill/tournaments")
def tournaments(
    limit: int = Query(10, ge=1, le=200),
    include_open: bool = Query(False),
) -> dict:
    """The latest field events, newest first, each marked cached or not."""
    rows = _run(_list_tournaments, limit, include_open)
    cached_ids = {c.get("tournament_id") for c in service.list_cached_postmortems() if c.get("closed")}
    out = [{**row, "cached": row["id"] in cached_ids} for row in rows]
    return {"tournaments": out, "count": len(out)}


@app.get("/brill/games")
def games(limit: int = Query(100, ge=1, le=500)) -> dict:
    rows = _run(service.list_cached_postmortems)[:limit]
    return {"games": rows, "count": len(rows)}


@app.post("/brill/generate")
def generate(request: GenerateRequest) -> dict:
    try:
        return _generate_isolated(
            request.tournament_id,
            sd_samples=request.sd_samples,
            force=request.force,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# The site's final ranking of a closed tournament does not change.
_official_cache: Dict[str, List[Dict[str, Any]]] = {}


def _official_standings(tournament_id: Optional[str]) -> List[Dict[str, Any]]:
    """The site's ranking, best first; empty when the site cannot be read.

    It only names the winner, whose report opens first, so the entrants are
    still listed without it.
    """
    if not tournament_id:
        return []
    rows = _official_cache.get(tournament_id)
    if rows is None:
        try:
            rows = brill_client.official_standings(tournament_id)
        except brill_client.BrillError:
            return []
        if rows:
            _official_cache[tournament_id] = rows
    return rows


@app.get("/brill/entrants")
def entrants(tournament: Optional[str] = Query(None)) -> dict:
    """Entrants of one tournament with their standings, best first, and the
    winner, whose report opens first."""
    frame, meta = _run(_tournament_frame, tournament)
    rows = _run(service.standings, frame)
    winner = service.winner(frame, _official_standings(meta.get("tournament_id")))
    return {"meta": meta, "entrants": rows, "count": len(rows), "winner": winner}


@app.get("/brill/boards")
def boards(
    tournament: Optional[str] = Query(None),
    key: Optional[str] = Query(None),
    entrant: Optional[str] = Query(None),
    only_my_boards: bool = Query(True),
    columns: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
) -> dict:
    """Per-table, per-board rows. With an entrant, that entrant's boards only
    unless only_my_boards is false."""
    frame, meta = _run(_entrant_frame, tournament or key, entrant)
    if meta.get("entrant") and only_my_boards:
        frame = frame.filter("Boards_I_Played")
    selected = [column.strip() for column in columns.split(",")] if columns else None
    return _run(service.board_results, frame, meta, columns=selected, limit=limit)


@app.post("/brill/sql")
def sql(request: SqlRequest) -> dict:
    """DuckDB SQL on table self. With an entrant the Boards_* flags are theirs;
    the {Pair_Direction} family of macros is filled in either way."""
    frame, meta = _run(_entrant_frame, request.tournament or request.key, request.entrant)
    result = _run(service.run_sql, frame, request.sql, limit=request.limit)
    result["meta"] = meta
    return result


@app.get("/brill/schema")
def schema(
    tournament: Optional[str] = Query(None),
    key: Optional[str] = Query(None),
    pattern: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=5000),
) -> dict:
    frame, _meta = _run(_tournament_frame, tournament or key)
    return _run(service.schema_columns, frame, pattern=pattern, limit=limit)


@app.get("/brill/parquet")
def parquet(key: Optional[str] = Query(None)) -> Response:
    frame, meta = _run(service.load_postmortem, key)
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    filename = meta.get("cache_file") or "postmortem-brill.parquet"
    return Response(
        content=buffer.getvalue(),
        media_type="application/vnd.apache.parquet",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/brill/pbn")
def pbn(key: Optional[str] = Query(None)) -> PlainTextResponse:
    """The field PBN a cached tournament was built from."""
    _frame, meta = _run(service.load_postmortem, key)
    path = service.CACHE_DIR / f'{meta["key"]}.pbn'
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No PBN stored for {meta['key']}.")
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        headers={"Content-Disposition": f'attachment; filename="{meta.get("pbn_name") or path.name}"'},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("BRILL_POSTMORTEM_API_PORT", "8531"))
    print(
        f"[postmortem-brill-api] start {datetime.now(timezone.utc).isoformat()} port={port}",
        flush=True,
    )
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=900)
