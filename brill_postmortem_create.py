"""Fetch one Brill tournament, build its field PBN, augment, and cache.

Only the FastAPI server imports this. Streamlit reaches generate through
POST /brill/generate.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
from endplay.parsers import pbn
from endplay.types import Board

import brill_board_flags
import brill_client
import brill_postmortem_service as service

_APP_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_SD_SAMPLES = 10
# Tags the site writes once, on the first record, that describe the whole file.
_FILE_TAGS = ('Event', 'Site', 'Date')


def _ensure_mlbridge_on_path() -> None:
    for path in (_APP_DIR, _APP_DIR.parent, _APP_DIR.parent.parent):
        if (path / 'mlBridge').is_dir():
            text = str(path)
            if text not in sys.path:
                sys.path.append(text)
            return
    raise FileNotFoundError(
        f"mlBridge not found next to {_APP_DIR}, {_APP_DIR.parent}, or {_APP_DIR.parent.parent}"
    )


def parse_field_pbn(text: str, source: str = 'Brill PBN') -> List[Board]:
    try:
        boards = pbn.loads(text)
    except Exception as exc:
        raise ValueError(f"Error parsing {source}: {exc}") from exc
    if not boards:
        raise ValueError(f"{source} has no boards.")
    # Every record needs the same tags, or the dataframe's info struct is ragged.
    first = boards[0].info
    for board in boards:
        for tag in _FILE_TAGS:
            if tag not in board.info and tag in first:
                board.info[tag] = first[tag]
        for tag in ('Table', 'DecisionTime', 'EngineTime'):
            if tag not in board.info:
                board.info[tag] = None
    return boards


def boards_to_mlbridge_df(boards: List[Board], path: pathlib.Path) -> pl.DataFrame:
    _ensure_mlbridge_on_path()
    from mlBridge import mlBridgeEndplayLib

    df = mlBridgeEndplayLib.endplay_boards_to_df({path: boards})
    return mlBridgeEndplayLib.convert_endplay_df_to_mlBridge_df(df)


def apply_entrant_identity(df: pl.DataFrame) -> pl.DataFrame:
    """Give every table its own pair numbers and player ids.

    The entrant sits South at their own table beside three robots, so South's
    id is the entrant's name (made unique when two entrants share one) and the
    robots get per-table ids. Pair numbers are the table number.
    """
    table = pl.col('Table').cast(pl.UInt32, strict=False).fill_null(0)
    df = df.with_columns(table.alias('Table'))
    tables_per_name = pl.col('Table').n_unique().over('Player_Name_S')
    entrant = (
        pl.when(tables_per_name > 1)
        .then(pl.format('{} (T{})', pl.col('Player_Name_S'), pl.col('Table')))
        .otherwise(pl.col('Player_Name_S'))
    )
    return df.with_columns(
        pl.col('Table').alias('Pair_Number_NS'),
        pl.col('Table').alias('Pair_Number_EW'),
        entrant.alias('Entrant'),
        entrant.alias('Player_ID_S'),
        pl.format('T{}_N', pl.col('Table')).alias('Player_ID_N'),
        pl.format('T{}_E', pl.col('Table')).alias('Player_ID_E'),
        pl.format('T{}_W', pl.col('Table')).alias('Player_ID_W'),
    )


def apply_default_board_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Whole-field flags, baked into the cached parquet for SQL callers."""
    return brill_board_flags.apply_board_flags(df, None)


def augment_boards_df(
    df: pl.DataFrame,
    sd_samples: int = DEFAULT_SD_SAMPLES,
    progress: Optional[Any] = None,
    lock_func: Optional[Callable[..., pl.DataFrame]] = None,
) -> pl.DataFrame:
    _ensure_mlbridge_on_path()
    from mlBridge.mlBridgeAugmentLib import AllAugmentations

    sd_samples = max(1, min(int(sd_samples), 1000))
    augmenter = AllAugmentations(
        df,
        None,
        sd_productions=sd_samples,
        progress=progress,
        lock_func=lock_func,
    )
    df, _hrs = augmenter.perform_all_augmentations()
    return df


def pbn_text_to_augmented_df(
    text: str,
    path: pathlib.Path,
    sd_samples: int = DEFAULT_SD_SAMPLES,
) -> pl.DataFrame:
    boards = parse_field_pbn(text, str(path))
    df = boards_to_mlbridge_df(boards, path)
    df = apply_entrant_identity(df)
    df = augment_boards_df(df, sd_samples=sd_samples)
    return apply_default_board_flags(df)


def _cached_meta(tournament_id: str, sd_samples: int) -> Optional[Dict[str, Any]]:
    entry = service.find_cached(tournament_id)
    # A cache built while the event was open holds only part of the field.
    if entry is None or not entry.get('closed'):
        return None
    _df, meta = service.load_postmortem(entry['key'])
    return {**meta, 'cached': True, 'sd_samples': sd_samples}


def generate_postmortem(
    tournament_id: str,
    *,
    sd_samples: int = DEFAULT_SD_SAMPLES,
    force: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Fetch, augment, and cache one tournament. Returns (df, meta)."""
    tournament_id = (tournament_id or '').strip()
    if not tournament_id:
        raise ValueError("A Brill tournament id is required.")
    sd_samples = max(1, min(int(sd_samples), 1000))
    if not force:
        meta = _cached_meta(tournament_id, sd_samples)
        if meta is not None:
            return service.load_postmortem(meta['key'])[0], meta

    fetched = brill_client.field_pbn(tournament_id)
    tournament = fetched['tournament']
    key = service.tournament_cache_key(tournament_id, fetched['filename'])
    df = pbn_text_to_augmented_df(
        fetched['text'], pathlib.Path(fetched['filename']), sd_samples=sd_samples)
    service.save_augmented_df_to_cache(
        df,
        key,
        {
            'tournament_id': tournament_id,
            'tournament_name': tournament.get('name'),
            'scoring_method': tournament.get('scoring_method'),
            'starts_at': tournament.get('starts_at'),
            'closed': bool(tournament.get('closed')),
            'pbn_file': f'{key}.pbn',
            'pbn_name': fetched['filename'],
            'tables': fetched['tables'],
            'url': f'{brill_client.SITE_URL}/#tournament?tournamentId={tournament_id}',
        },
        pbn_text=fetched['text'],
    )
    _df, meta = service.load_postmortem(key)
    return df, {**meta, 'cached': False, 'sd_samples': sd_samples}


def _cli(argv: Optional[List[str]] = None) -> int:
    """Write generate() meta JSON to --meta-out. Used by the API subprocess."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate a postmortem-brill cache entry.")
    parser.add_argument("tournament_id")
    parser.add_argument("--sd-samples", type=int, default=DEFAULT_SD_SAMPLES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--meta-out", required=True)
    args = parser.parse_args(argv)
    # Expected failures go back as data, so the API can answer with the reason
    # and a fitting status rather than a traceback.
    try:
        _df, meta = generate_postmortem(
            args.tournament_id, sd_samples=args.sd_samples, force=args.force
        )
    except brill_client.BrillAuthError as exc:
        meta = {'error': str(exc), 'status': 401}
    except brill_client.BrillError as exc:
        meta = {'error': str(exc), 'status': 502}
    except (KeyError, ValueError) as exc:
        meta = {'error': str(exc.args[0] if exc.args else exc), 'status': 422}
    pathlib.Path(args.meta_out).write_text(json.dumps(meta), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
