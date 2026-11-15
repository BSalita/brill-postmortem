"""Headless access to cached postmortem-brill augmented dataframes.

Each Brill tournament is fetched once, augmented, and stored as
cache/df-{key}.parquet with a df-{key}.json sidecar (tournament id, name,
tables) and the field PBN it was built from, cache/{key}.pbn. key is the
site's PBN file stem plus the first 8 characters of the tournament id, e.g.
daily-imp-2026-11-12-725de0d1.

This module is Streamlit-free and network-free: it enumerates those parquets
and runs DuckDB SQL against the dataframe registered as 'self', as the app's
SQL favorites do.

Env:
  BRILL_POSTMORTEM_CACHE_DIR  cache directory (default ./cache next to this file)
"""

import json
import os
import pathlib
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import polars as pl

_APP_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = pathlib.Path(os.environ.get("BRILL_POSTMORTEM_CACHE_DIR", str(_APP_DIR / "cache")))

CON_REGISTER_NAME = "self"
DUCKDB_THREADS = 8
DEFAULT_SQL_ROW_LIMIT = 500
MAX_SQL_ROW_LIMIT = 2000
MAX_SCHEMA_COLUMNS = 1000

_CACHE_FILE_RE = re.compile(r"^df-(?P<key>.+)\.parquet$")

# Default column set for the per-board summary; intersected with the actual
# dataframe columns.
BOARD_SUMMARY_COLUMNS = [
    "Board", "Table", "Entrant", "Dealer", "Vul", "Contract",
    "Declarer_Direction", "Result", "Tricks", "Score_NS", "Pct_NS", "MP_NS",
    "IMP_NS", "ParScore_NS", "ParContracts", "DD_Score_NS", "EV_Score_NS", "PBN",
]


def tournament_cache_key(tournament_id: str, filename: str) -> str:
    """Filesystem-safe key: PBN stem plus a short tournament id, so two events
    with the same name stay apart."""
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', pathlib.Path(filename).stem).strip('_')[:60] or 'brill'
    return f"{stem}-{re.sub(r'[^A-Za-z0-9]', '', tournament_id)[:8]}"


def _sidecar(key: str) -> Dict[str, Any]:
    sidecar = CACHE_DIR / f"df-{key}.json"
    if not sidecar.is_file():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def list_cached_postmortems() -> List[Dict[str, Any]]:
    """Cached tournaments, newest file first."""
    out: List[Dict[str, Any]] = []
    if not CACHE_DIR.is_dir():
        return out
    for f in CACHE_DIR.glob("df-*.parquet"):
        m = _CACHE_FILE_RE.match(f.name)
        if m is None:
            continue
        key = m.group("key")
        stat = f.stat()
        out.append(
            {
                **_sidecar(key),
                "key": key,
                "file": f.name,
                "size_bytes": stat.st_size,
                "cached_at": stat.st_mtime,
            }
        )
    out.sort(key=lambda d: d["cached_at"], reverse=True)
    return out


def dataset_info() -> Dict[str, Any]:
    cached = list_cached_postmortems()
    return {
        "cache_dir": str(CACHE_DIR),
        "cached_postmortems": len(cached),
        "keys": [c["key"] for c in cached],
        "note": (
            "GET /brill/tournaments lists the latest closed events; POST "
            "/brill/generate with a tournament_id fetches and caches one."
        ),
    }


def save_augmented_df_to_cache(
    df: pl.DataFrame, key: str, meta: Dict[str, Any], pbn_text: Optional[str] = None
) -> pathlib.Path:
    """Persist the augmented dataframe, its sidecar, and the source PBN."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f'df-{key}.parquet'
    df.write_parquet(cache_file)
    if pbn_text is not None:
        (CACHE_DIR / f'{key}.pbn').write_text(pbn_text, encoding='utf-8')
    sidecar = {**meta, 'cached_at_utc': datetime.now(timezone.utc).isoformat()}
    (CACHE_DIR / f'df-{key}.json').write_text(json.dumps(sidecar), encoding='utf-8')
    with _df_cache_lock:
        _df_cache.clear()
    return cache_file


def find_cached(key_or_tournament_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The cache entry for a cache key or a tournament id, or None."""
    if not key_or_tournament_id:
        return None
    return next(
        (
            c for c in list_cached_postmortems()
            if key_or_tournament_id in (c["key"], c.get("tournament_id"))
        ),
        None,
    )


def _resolve_cache_file(key: Optional[str] = None) -> Tuple[pathlib.Path, Dict[str, Any]]:
    if not key:
        raise ValueError("Pass key (from /brill/games) or tournament_id (from /brill/tournaments).")
    entry = find_cached(key)
    if entry is None:
        raise FileNotFoundError(
            f"No cached postmortem-brill tournament for {key!r}. "
            "Generate it first with POST /brill/generate."
        )
    return CACHE_DIR / entry["file"], entry


# Small in-process cache keyed by (path, mtime); a few frames resident at most.
_df_cache: Dict[Tuple[str, float], pl.DataFrame] = {}
_df_cache_lock = threading.Lock()
_DF_CACHE_MAX = 4


def _read_parquet_cached(path: pathlib.Path) -> pl.DataFrame:
    key = (str(path), path.stat().st_mtime)
    with _df_cache_lock:
        if key in _df_cache:
            return _df_cache[key]
    df = pl.read_parquet(path)
    with _df_cache_lock:
        if len(_df_cache) >= _DF_CACHE_MAX:
            _df_cache.pop(next(iter(_df_cache)))
        _df_cache[key] = df
    return df


def load_postmortem(key: Optional[str] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Load a cached tournament by cache key or tournament id. Returns (df, meta)."""
    path, entry = _resolve_cache_file(key)
    df = _read_parquet_cached(path)
    meta = {
        **{k: v for k, v in entry.items() if k not in ("size_bytes", "cached_at", "file")},
        "cache_file": entry["file"],
        "row_count": df.height,
    }
    return df, meta


def run_sql(df: pl.DataFrame, sql: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Run DuckDB SQL against the dataframe registered as 'self'."""
    limit = max(1, min(limit or DEFAULT_SQL_ROW_LIMIT, MAX_SQL_ROW_LIMIT))
    sql = fill_sql_macros(sql.strip().rstrip(";"))
    if f"from {CON_REGISTER_NAME}" not in sql.lower():
        sql = f"FROM {CON_REGISTER_NAME} " + sql
    # external access off: only the registered dataframe is queryable.
    con = duckdb.connect(
        config={"enable_external_access": "false", "threads": DUCKDB_THREADS}
    )
    try:
        con.register(CON_REGISTER_NAME, df)
        result = con.sql(sql).pl()
    finally:
        con.close()
    truncated = result.height > limit
    result = result.head(limit)
    return {
        "sql": sql,
        "columns": result.columns,
        "rows": result.to_dicts(),
        "row_count": result.height,
        "truncated": truncated,
    }


def board_results(
    df: pl.DataFrame,
    meta: Dict[str, Any],
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-table, per-board rows of the cached augmented dataframe."""
    limit = max(1, min(limit or DEFAULT_SQL_ROW_LIMIT, MAX_SQL_ROW_LIMIT))
    wanted = columns or BOARD_SUMMARY_COLUMNS
    missing = [c for c in wanted if c not in df.columns]
    selected = [c for c in wanted if c in df.columns]
    if not selected:
        raise ValueError(f"None of the requested columns exist. Missing: {missing}")
    sort_cols = [c for c in ("Board", "Table") if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)
    df = df.select(selected).head(limit)
    return {
        "meta": meta,
        "columns": selected,
        "missing_columns": missing,
        "rows": df.to_dicts(),
        "row_count": df.height,
    }


def schema_columns(df: pl.DataFrame, pattern: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Column names and dtypes, optionally filtered by a case-insensitive regex."""
    limit = max(1, min(limit or MAX_SCHEMA_COLUMNS, MAX_SCHEMA_COLUMNS))
    names = sorted(df.columns)
    if pattern:
        rx = re.compile(pattern, re.IGNORECASE)
        names = [c for c in names if rx.search(c)]
    truncated = len(names) > limit
    names = names[:limit]
    dtypes = dict(zip(df.columns, (str(t) for t in df.dtypes)))
    return {
        "total_columns": df.width,
        "matched_columns": len(names),
        "truncated": truncated,
        "columns": {c: dtypes[c] for c in names},
    }


# ---- One entrant's view of a tournament ----

# Every entrant sits South at their own table, so the directions the favorites
# SQL asks for are the same for all of them.
SQL_MACROS = {
    '{Pair_Direction}': 'NS',
    '{Opponent_Pair_Direction}': 'EW',
    '{Player_Direction}': 'S',
    '{Partner_Direction}': 'N',
}


def fill_sql_macros(sql: str) -> str:
    for macro, value in SQL_MACROS.items():
        sql = sql.replace(macro, value)
    return sql


def entrant_names(df: pl.DataFrame) -> List[str]:
    """Entrant names in table order."""
    if 'Entrant' not in df.columns:
        return []
    return df.select('Table', 'Entrant').unique().sort('Table', 'Entrant')['Entrant'].to_list()


def resolve_entrant(df: pl.DataFrame, name: str) -> str:
    """The entrant called name: exact, then ignoring case, then a unique part of the name."""
    names = entrant_names(df)
    wanted = (name or '').strip()
    if wanted in names:
        return wanted
    folded = wanted.casefold()
    exact = [n for n in names if n.casefold() == folded]
    partial = [n for n in names if folded and folded in n.casefold()]
    for matches in (exact, partial):
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"No single entrant matches {name!r}. Entrants: {', '.join(names)}")


def standings(df: pl.DataFrame) -> List[Dict[str, Any]]:
    """One row per entrant, best matchpoint percentage first.

    pct is the mean of the entrant's board percentages against the other
    tables, for IMP events as well; score is the total of their NS scores.
    """
    if 'Entrant' not in df.columns:
        return []
    aggregates = [pl.len().alias('boards')]
    if 'Pct_NS' in df.columns:
        aggregates.append((pl.col('Pct_NS').cast(pl.Float64).mean() * 100).round(2).alias('pct'))
    if 'Score_NS' in df.columns:
        aggregates.append(pl.col('Score_NS').sum().alias('score'))
    table = df.group_by('Entrant', 'Table').agg(aggregates)
    if 'pct' in table.columns:
        table = table.with_columns(
            pl.col('pct').rank(method='min', descending=True).cast(pl.UInt32).alias('rank')
        ).sort(['rank', 'Table'])
    else:
        table = table.sort('Table')
    return table.rename({'Entrant': 'entrant', 'Table': 'table'}).to_dicts()


def winner(df: pl.DataFrame, official: List[Dict[str, Any]]) -> Optional[str]:
    """The winning entrant, whose report opens first.

    official is the site's ranking, best first: {name, rank}. The winner is
    the site's rank 1; without a usable site ranking, the best matchpoint
    percentage here.
    """
    names = set(entrant_names(df))
    for row in official:
        if row.get('rank') == 1 and row.get('name') in names:
            return row['name']
    table = standings(df)
    return table[0]['entrant'] if table else None


def personalize(df: pl.DataFrame, entrant: Optional[str]) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Flag df for one entrant (or the whole field). Returns (df, who)."""
    import brill_board_flags

    if not (entrant or '').strip():
        return brill_board_flags.apply_board_flags(df, None), {'entrant': None}
    name = resolve_entrant(df, entrant)
    row = df.filter(pl.col('Entrant').eq(name)).row(0, named=True)
    who = {
        'entrant': name,
        'table': row.get('Table'),
        'player_direction': 'S',
        'partner_direction': 'N',
        'partner_name': row.get('Player_Name_N'),
        'pair_direction': 'NS',
    }
    return brill_board_flags.apply_board_flags(df, name), who


def _normalized(text: Any) -> str:
    return ' '.join(re.findall(r'[a-z0-9]+', str(text or '').lower()))


def match_tournament(tournaments: List[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
    """The listed tournament with this id or name; a part of the name if only one has it."""
    wanted = (text or '').strip()
    for row in tournaments:
        if row.get('id') == wanted:
            return row
    needle = _normalized(wanted)
    if not needle:
        return None
    exact = [row for row in tournaments if _normalized(row.get('name')) == needle]
    if exact:
        return exact[0]  # newest first
    partial = [row for row in tournaments if needle in _normalized(row.get('name'))]
    return partial[0] if len(partial) == 1 else None
