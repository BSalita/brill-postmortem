"""The Boards_* personalization flags the favorites SQL filters on.

Polars only, so both the library (which bakes whole-field flags into the
cached parquet) and the Streamlit app (which re-flags for a chosen entrant)
can import it.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

ALL_ENTRANTS = 'All entrants'


def apply_board_flags(df: pl.DataFrame, entrant: Optional[str] = None) -> pl.DataFrame:
    """With an entrant, "I" is that entrant (South at their own table) and
    "partner" is the robot North at the same table. Without one, every table
    counts, so the report covers the whole field."""
    if entrant is None or entrant == ALL_ENTRANTS:
        played = declared = partner_declared = pl.lit(True)
    else:
        played = pl.col('Entrant').eq(entrant).fill_null(False)
        declarer = pl.col('Declarer_Direction') if 'Declarer_Direction' in df.columns else pl.lit(None, dtype=pl.String)
        declared = played & declarer.eq('S').fill_null(False)
        partner_declared = played & declarer.eq('N').fill_null(False)
    df = df.with_columns(
        played.alias('Boards_I_Played'),
        declared.alias('Boards_I_Declared'),
        partner_declared.alias('Boards_Partner_Declared'),
    )
    df = df.with_columns(
        pl.col('Boards_I_Played').alias('Boards_We_Played'),
        pl.col('Boards_I_Played').alias('Our_Boards'),
        (pl.col('Boards_I_Declared') | pl.col('Boards_Partner_Declared')).alias('Boards_We_Declared'),
    )
    contract = pl.col('Contract') if 'Contract' in df.columns else pl.lit('PASS')
    return df.with_columns(
        (
            pl.col('Boards_I_Played')
            & ~pl.col('Boards_We_Declared')
            & contract.ne('PASS')
        ).alias('Boards_Opponent_Declared'),
    )
