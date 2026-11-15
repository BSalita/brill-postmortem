"""Read tournaments and board results from Brill Bridge Challenge.

The site (https://brill.aalborgdata.dk) is a browser app over a Supabase
database. This client makes the same REST reads the site's own JavaScript
makes, signed in as the account in .env:

  BRILL_EMAIL, BRILL_PASSWORD   the account to sign in as (required)
  BRILL_SITE_URL                default https://brill.aalborgdata.dk
  BRILL_SUPABASE_URL, BRILL_SUPABASE_KEY
                                override the project URL and publishable key,
                                which are otherwise read from the site's
                                js/supabase-client.js

The database decides what an account may read: a running event returns only
the account's own boards, a closed one returns the whole field.
"""

from __future__ import annotations

import os
import pathlib
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

import brill_pbn

_APP_DIR = pathlib.Path(__file__).resolve().parent
load_dotenv(_APP_DIR / '.env')

SITE_URL = os.environ.get('BRILL_SITE_URL', 'https://brill.aalborgdata.dk').rstrip('/')
_TIMEOUT_S = 30
_PAGE_ROWS = 1000
# Tournament types where every entrant plays the same boards at their own table.
FIELD_TYPES = ('imp_across', 'mp_across', 'mp_field', 'imp_field')
_TOURNAMENT_COLUMNS = (
    'id,name,description,status,tournament_type,scoring_method,boards_per_round,'
    'starts_at,ends_at,created_at,daily_run_id'
)
_RESULT_COLUMNS = (
    '*,participant:challenge_participants!participant_id('
    'id,display_name,participant_type,robot_name,user_id,rank,total_score,join_order)'
)


class BrillError(RuntimeError):
    """The site refused or could not answer a request."""


class BrillAuthError(BrillError):
    """No usable sign-in: credentials missing or rejected."""


_lock = threading.Lock()
_config: Optional[Tuple[str, str]] = None
_session: Dict[str, Any] = {}


def _supabase_config() -> Tuple[str, str]:
    """(project URL, publishable key), from env or the site's own client script."""
    global _config
    if _config is not None:
        return _config
    url = os.environ.get('BRILL_SUPABASE_URL', '').strip()
    key = os.environ.get('BRILL_SUPABASE_KEY', '').strip()
    if not (url and key):
        try:
            resp = requests.get(f'{SITE_URL}/js/supabase-client.js', timeout=_TIMEOUT_S)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise BrillError(f'Brill site unreachable at {SITE_URL}: {exc}') from exc
        found_url = re.search(r"SUPABASE_URL\s*=\s*'(https://[^']+)'", resp.text)
        found_key = re.search(r"SUPABASE_ANON_KEY\s*=\s*'([^']+)'", resp.text)
        if not (found_url and found_key):
            raise BrillError(
                f'Could not read the Supabase settings from {SITE_URL}/js/supabase-client.js. '
                'Set BRILL_SUPABASE_URL and BRILL_SUPABASE_KEY in .env.'
            )
        url, key = url or found_url.group(1), key or found_key.group(1)
    _config = (url.rstrip('/'), key)
    return _config


def credentials_configured() -> bool:
    return bool(os.environ.get('BRILL_EMAIL', '').strip() and os.environ.get('BRILL_PASSWORD', ''))


def _sign_in() -> str:
    # Re-read .env so credentials added or changed while the API runs take effect.
    load_dotenv(_APP_DIR / '.env', override=True)
    email = os.environ.get('BRILL_EMAIL', '').strip()
    password = os.environ.get('BRILL_PASSWORD', '')
    if not (email and password):
        raise BrillAuthError(
            f'Brill sign-in is not configured. Set BRILL_EMAIL and BRILL_PASSWORD in '
            f'{_APP_DIR / ".env"}.'
        )
    url, key = _supabase_config()
    try:
        resp = requests.post(
            f'{url}/auth/v1/token',
            params={'grant_type': 'password'},
            headers={'apikey': key, 'Content-Type': 'application/json'},
            json={'email': email, 'password': password},
            timeout=_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise BrillError(f'Brill sign-in unreachable: {exc}') from exc
    if not resp.ok:
        try:
            body = resp.json()
            detail = body.get('error_description') or body.get('msg') or body.get('message')
        except ValueError:
            detail = None
        raise BrillAuthError(
            f'Brill sign-in failed for BRILL_EMAIL ({resp.status_code}): {detail or resp.reason}'
        )
    body = resp.json()
    _session['access_token'] = body['access_token']
    _session['expires_at'] = time.time() + int(body.get('expires_in') or 3600)
    _session['user_id'] = (body.get('user') or {}).get('id')
    return body['access_token']


def _access_token(force: bool = False) -> str:
    with _lock:
        if not force and _session.get('access_token') and time.time() < _session['expires_at'] - 60:
            return _session['access_token']
        return _sign_in()


def _select(table: str, params: Dict[str, str]) -> List[Dict[str, Any]]:
    """All rows of a PostgREST select, signed in, following pagination."""
    url, key = _supabase_config()
    rows: List[Dict[str, Any]] = []
    limit = int(params.get('limit') or 0)
    token = _access_token()
    retried = False
    while True:
        page = dict(params)
        page['offset'] = str(len(rows))
        page['limit'] = str(min(_PAGE_ROWS, limit - len(rows)) if limit else _PAGE_ROWS)
        try:
            resp = requests.get(
                f'{url}/rest/v1/{table}',
                params=page,
                headers={'apikey': key, 'Authorization': f'Bearer {token}'},
                timeout=_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise BrillError(f'Brill database unreachable ({table}): {exc}') from exc
        if resp.status_code == 401 and not retried:
            token = _access_token(force=True)  # expired early; sign in once more
            retried = True
            continue
        if not resp.ok:
            try:
                detail = resp.json().get('message')
            except ValueError:
                detail = None
            raise BrillError(f'Brill {table} read failed ({resp.status_code}): {detail or resp.text[:200]}')
        batch = resp.json()
        rows.extend(batch)
        if len(batch) < int(page['limit']) or (limit and len(rows) >= limit):
            return rows


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def is_closed(tournament: Dict[str, Any]) -> bool:
    """True once the whole field is readable: completed, or past its closing time."""
    if tournament.get('status') == 'completed':
        return True
    ends_at = _parse_time(tournament.get('ends_at'))
    return ends_at is not None and ends_at <= datetime.now(timezone.utc)


def list_tournaments(limit: int = 10, include_open: bool = False) -> List[Dict[str, Any]]:
    """The latest field events, newest first. Open events are left out by default:
    until an event closes the site shows an account only its own boards."""
    limit = max(1, min(int(limit), 200))
    rows = _select('tournaments', {
        'select': _TOURNAMENT_COLUMNS,
        'tournament_type': f'in.({",".join(FIELD_TYPES)})',
        'order': 'created_at.desc',
        # Four dailies open each morning; over-read so `limit` closed ones remain.
        'limit': str(limit + 20),
    })
    out = []
    for row in rows:
        closed = is_closed(row)
        if closed or include_open:
            out.append({**row, 'closed': closed})
    return out[:limit]


def get_tournament(tournament_id: str) -> Dict[str, Any]:
    rows = _select('tournaments', {'select': _TOURNAMENT_COLUMNS, 'id': f'eq.{tournament_id}'})
    if not rows:
        raise KeyError(f'No Brill tournament {tournament_id!r}.')
    return {**rows[0], 'closed': is_closed(rows[0])}


def field_round(tournament_id: str) -> Optional[Dict[str, Any]]:
    """The tournament's field round: the one that has a challenge, as the site picks it."""
    rounds = _select('tournament_rounds', {
        'select': '*', 'tournament_id': f'eq.{tournament_id}', 'order': 'round_number'})
    return next((r for r in rounds if r.get('challenge_id')), rounds[0] if rounds else None)


def official_standings(tournament_id: str) -> List[Dict[str, Any]]:
    """The site's own final ranking, best first: {name, rank, is_account}.

    is_account marks the entry of the account this client signs in with.
    """
    rows = _select('tournament_players', {
        'select': 'display_name,final_rank,user_id',
        'tournament_id': f'eq.{tournament_id}',
        'order': 'final_rank.asc.nullslast',
    })
    account = _session.get('user_id')  # set by the sign-in the read above needed
    return [
        {
            'name': row.get('display_name'),
            'rank': row.get('final_rank'),
            'is_account': bool(account) and row.get('user_id') == account,
        }
        for row in rows
    ]


def pbn_filename(tournament: Dict[str, Any]) -> str:
    """The site's download name: 'Daily IMP — 2026-11-12' -> 'daily-imp-2026-11-12.pbn'."""
    stem = re.sub(r'[^a-z0-9]+', '-', (tournament.get('name') or 'tournament').lower()).strip('-')
    return f'{stem or "tournament"}.pbn'


def field_pbn(tournament_id: str) -> Dict[str, Any]:
    """A field event's boards as PBN text, every table the account may read.

    Returns {text, filename, tables, boards, tournament}.
    """
    tournament = get_tournament(tournament_id)
    if tournament.get('tournament_type') not in FIELD_TYPES:
        raise ValueError(f'{tournament.get("name")!r} is not a field event.')
    round_row = field_round(tournament_id)
    if not round_row or not round_row.get('challenge_id'):
        raise ValueError(f'{tournament.get("name")!r} has not been started yet.')
    challenge_id = round_row['challenge_id']
    challenges = _select('challenges', {'select': '*', 'id': f'eq.{challenge_id}'})
    if not challenges:
        raise ValueError(f'The boards of {tournament.get("name")!r} are not readable by this account.')
    rows = _select('challenge_board_results', {
        'select': _RESULT_COLUMNS,
        'challenge_id': f'eq.{challenge_id}',
        'order': 'board_number.asc',
    })
    by_participant = brill_pbn.group_board_results(rows)
    if not by_participant:
        raise ValueError(
            f'No board results are readable for {tournament.get("name")!r}. '
            'The field is published when the event closes.'
        )
    text = brill_pbn.generate_pbn_from_results(
        challenges[0],
        by_participant,
        event=tournament.get('name'),
        date=(tournament.get('starts_at') or tournament.get('created_at') or '')[:10] or None,
    )
    return {
        'text': text,
        'filename': pbn_filename(tournament),
        'tables': len(by_participant),
        'boards': len(rows),
        'tournament': tournament,
    }
