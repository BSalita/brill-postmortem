"""postmortem-brill library: PBN -> dataframe identity, flags, cache, API boundary.

Augmentation (double and single dummy) is not run here.
"""

import ast
import pathlib
import tempfile
import unittest
from unittest import mock

import polars as pl

import brill_board_flags
import brill_pbn
import brill_postmortem_create as create
import brill_postmortem_service as service
import test_brill_pbn

ROOT = pathlib.Path(__file__).resolve().parent


def sample_df() -> pl.DataFrame:
    text = brill_pbn.generate_pbn_from_results(
        test_brill_pbn.CHALLENGE,
        brill_pbn.group_board_results(test_brill_pbn.sample_rows()),
        event='Daily IMP — 2026-11-12', date='2026-11-12')
    boards = create.parse_field_pbn(text)
    df = create.boards_to_mlbridge_df(boards, pathlib.Path('daily-imp-2026-11-12.pbn'))
    return create.apply_entrant_identity(df)


class DataframeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = sample_df()

    def test_one_row_per_table_per_board_with_file_tags_on_every_row(self):
        self.assertEqual(self.df.height, 5)
        self.assertEqual(self.df['Date'].unique().to_list(), ['2026.11.12'])
        self.assertEqual(self.df['Event'].n_unique(), 1)

    def test_entrant_is_south_and_tables_are_pair_numbers(self):
        rows = self.df.sort('Board', 'Table').select(
            'Board', 'Table', 'Entrant', 'Player_ID_S', 'Player_ID_N', 'Pair_Number_NS').rows()
        self.assertEqual(rows[0], (1, 1, 'Alice', 'Alice', 'T1_N', 1))
        self.assertEqual(rows[1], (1, 2, 'BEN', 'BEN', 'T2_N', 2))

    def test_entrants_sharing_a_name_are_told_apart_by_table(self):
        df = self.df.with_columns(pl.lit('Same').alias('Player_Name_S'), pl.col('Table').cast(pl.String))
        entrants = sorted(create.apply_entrant_identity(df)['Entrant'].unique().to_list())
        self.assertEqual(entrants, ['Same (T1)', 'Same (T2)', 'Same (T3)'])

    def test_whole_field_flags(self):
        df = brill_board_flags.apply_board_flags(self.df)
        self.assertTrue(df['Boards_I_Played'].all())
        self.assertTrue(df['Boards_We_Declared'].all())

    def test_entrant_flags(self):
        df = brill_board_flags.apply_board_flags(self.df, 'Alice').sort('Board', 'Table')
        flags = df.select(
            'Boards_I_Played', 'Boards_I_Declared', 'Boards_Partner_Declared', 'Boards_Opponent_Declared').rows()
        # Board 1: Alice's robot partner North declared 3NT. Board 2: Alice declared 4SX.
        self.assertEqual(flags[0], (True, False, True, False))
        self.assertEqual(flags[3], (True, True, False, False))
        # Other tables are not Alice's boards.
        self.assertEqual(flags[1], (False, False, False, False))


class CacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patch = mock.patch.object(service, 'CACHE_DIR', pathlib.Path(tmp.name))
        patch.start()
        self.addCleanup(patch.stop)
        self.key = service.tournament_cache_key('725de0d1-66fd-43f4', 'daily-imp-2026-11-12.pbn')
        service.save_augmented_df_to_cache(
            pl.DataFrame({'Board': [2, 1], 'Table': [1, 1], 'Contract': ['4SXS', '3NN']}),
            self.key,
            {'tournament_id': '725de0d1-66fd-43f4', 'tournament_name': 'Daily IMP', 'closed': True},
            pbn_text='% PBN 2.1\n',
        )

    def test_key_names_the_event_and_the_tournament(self):
        self.assertEqual(self.key, 'daily-imp-2026-11-12-725de0d1')

    def test_load_by_key_or_tournament_id(self):
        for handle in (self.key, '725de0d1-66fd-43f4'):
            df, meta = service.load_postmortem(handle)
            self.assertEqual(df.height, 2)
            self.assertEqual(meta['tournament_name'], 'Daily IMP')
        self.assertTrue((service.CACHE_DIR / f'{self.key}.pbn').is_file())

    def test_unknown_key(self):
        with self.assertRaises(FileNotFoundError):
            service.load_postmortem('nope')
        with self.assertRaises(ValueError):
            service.load_postmortem(None)

    def test_sql_and_boards(self):
        df, meta = service.load_postmortem(self.key)
        result = service.run_sql(df, "SELECT Contract WHERE Board = 1")
        self.assertEqual(result['rows'], [{'Contract': '3NN'}])
        boards = service.board_results(df, meta)
        self.assertEqual([r['Board'] for r in boards['rows']], [1, 2])

    def test_cache_built_while_the_event_was_open_is_not_reused(self):
        self.assertIsNotNone(create._cached_meta('725de0d1-66fd-43f4', 10))
        service.save_augmented_df_to_cache(
            pl.DataFrame({'Board': [1]}), self.key, {'tournament_id': '725de0d1-66fd-43f4', 'closed': False})
        self.assertIsNone(create._cached_meta('725de0d1-66fd-43f4', 10))


class StreamlitApiBoundaryTests(unittest.TestCase):
    def test_streamlit_does_not_import_the_library(self):
        source = (ROOT / 'postmortem_brill_streamlit.py').read_text(encoding='utf-8-sig')
        modules = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        forbidden = {'brill_client', 'brill_postmortem_create', 'brill_postmortem_service'}
        self.assertFalse(modules & forbidden)
        self.assertIn('brill_postmortem_api_client', modules)


if __name__ == '__main__':
    unittest.main()
