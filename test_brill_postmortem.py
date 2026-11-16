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


def scored_df() -> pl.DataFrame:
    """sample_df with the two augmented columns the standings read."""
    df = sample_df().sort('Board', 'Table')
    return df.with_columns(
        pl.Series('Pct_NS', [1.0, 0.5, 0.0, 0.0, 1.0], dtype=pl.Float32),
        pl.Series('Score_NS', [400, 150, 0, -500, -100], dtype=pl.Int16),
    )


class EntrantViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = scored_df()

    def test_resolve_entrant(self):
        self.assertEqual(service.resolve_entrant(self.df, 'BEN'), 'BEN')
        self.assertEqual(service.resolve_entrant(self.df, ' alice '), 'Alice')
        self.assertEqual(service.resolve_entrant(self.df, 'be'), 'BEN')

    def test_ambiguous_or_unknown_entrant_lists_the_field(self):
        for name in ('i', 'Nobody'):  # 'i' is in both Alice and Lia
            with self.assertRaisesRegex(ValueError, 'Entrants: Alice, BEN, Lia'):
                service.resolve_entrant(self.df, name)

    def test_standings_rank_by_mean_percentage(self):
        rows = service.standings(self.df)
        self.assertEqual(
            [(r['rank'], r['entrant'], r['pct'], r['boards'], r['score']) for r in rows],
            [(1, 'BEN', 75.0, 2, 50), (2, 'Alice', 50.0, 2, -100), (3, 'Lia', 0.0, 1, 0)])

    def test_winner_is_the_sites_rank_one(self):
        official = [{'name': 'Lia', 'rank': 1}, {'name': 'BEN', 'rank': 2}, {'name': 'Alice', 'rank': 3}]
        self.assertEqual(service.winner(self.df, official), 'Lia')

    def test_winner_without_a_usable_site_ranking_is_the_best_percentage(self):
        self.assertEqual(service.winner(self.df, []), 'BEN')
        # Names the site lists that are not at a table here are ignored.
        self.assertEqual(service.winner(self.df, [{'name': 'Zed', 'rank': 1}]), 'BEN')

    def test_ranking_follows_the_site(self):
        official = [{'name': 'Lia', 'rank': 1}, {'name': 'Alice', 'rank': 2}, {'name': 'BEN', 'rank': 3}]
        self.assertEqual(service.ranking(self.df, official), ['Lia', 'Alice', 'BEN'])

    def test_ranking_puts_entrants_the_site_does_not_rank_last_by_percentage(self):
        # By percentage: BEN, Alice, Lia.
        self.assertEqual(service.ranking(self.df, []), ['BEN', 'Alice', 'Lia'])
        official = [{'name': 'Zed', 'rank': 1}, {'name': 'Lia', 'rank': 2}, {'name': 'BEN', 'rank': None}]
        self.assertEqual(service.ranking(self.df, official), ['Lia', 'BEN', 'Alice'])
        self.assertEqual(service.winner(self.df, official), 'Lia')

    def test_ranking_settles_a_shared_site_rank_by_percentage(self):
        official = [{'name': 'Lia', 'rank': 1}, {'name': 'Alice', 'rank': 1}, {'name': 'BEN', 'rank': 3}]
        self.assertEqual(service.ranking(self.df, official), ['Alice', 'Lia', 'BEN'])

    def test_personalize_for_an_entrant(self):
        df, who = service.personalize(self.df, 'ben')
        self.assertEqual(who, {
            'entrant': 'BEN', 'table': 2, 'player_direction': 'S', 'partner_direction': 'N',
            'partner_name': 'Robot (Brill)', 'pair_direction': 'NS'})
        self.assertEqual(df.filter('Boards_I_Played')['Entrant'].unique().to_list(), ['BEN'])

    def test_personalize_for_the_whole_field(self):
        df, who = service.personalize(self.df, None)
        self.assertEqual(who, {'entrant': None})
        self.assertTrue(df['Boards_I_Played'].all())

    def test_sql_macros_are_filled(self):
        result = service.run_sql(
            self.df,
            "SELECT COUNT(*) AS n, '{Pair_Direction}{Opponent_Pair_Direction}' AS d "
            "FROM self WHERE Vul_{Pair_Direction}")
        self.assertEqual(result['rows'], [{'n': 2, 'd': 'NSEW'}])

    def test_match_tournament(self):
        listed = [
            {'id': 't3', 'name': 'Daily MP — 2026-11-12'},
            {'id': 't2', 'name': 'Daily IMP - strongest hand — 2026-11-12'},
            {'id': 't1', 'name': 'Daily IMP — 2026-11-12'},
        ]
        self.assertEqual(service.match_tournament(listed, 't2')['id'], 't2')
        self.assertEqual(service.match_tournament(listed, 'daily imp 2026-11-12')['id'], 't1')
        self.assertEqual(service.match_tournament(listed, 'strongest hand')['id'], 't2')
        self.assertIsNone(service.match_tournament(listed, 'Daily'))  # three match
        self.assertIsNone(service.match_tournament(listed, 'Monthly teams'))


class ApiTests(unittest.TestCase):
    """The REST API over a cached tournament. No Brill site, no augmentation."""

    LISTED = [
        {'id': 'aaaaaaaa-0000-0000-0000-000000000002', 'name': 'Daily MP — 2026-11-12', 'closed': True},
        {'id': 'aaaaaaaa-0000-0000-0000-000000000001', 'name': 'Daily IMP — 2026-11-12', 'closed': True},
    ]

    def setUp(self):
        import brill_postmortem_api_server as server

        self.server = server
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.generated = []
        self.official = []  # the site's ranking; a BrillError instance is raised instead

        def official_standings(tournament_id):
            if isinstance(self.official, Exception):
                raise self.official
            return self.official

        def generate(tournament_id, sd_samples=10, force=False):
            self.generated.append(tournament_id)
            self._cache(tournament_id, 'generated')
            return {'key': service.find_cached(tournament_id)['key']}

        for patch in (
            mock.patch.object(service, 'CACHE_DIR', pathlib.Path(tmp.name)),
            mock.patch.object(server, '_list_tournaments', lambda limit, include_open: self.LISTED[:limit]),
            mock.patch.object(server, '_generate_isolated', generate),
            mock.patch.object(server.brill_client, 'official_standings', official_standings),
            mock.patch.dict(server._official_cache, clear=True),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self._cache(self.LISTED[0]['id'], 'daily-mp-2026-11-12')

    def _cache(self, tournament_id, stem):
        service.save_augmented_df_to_cache(
            brill_board_flags.apply_board_flags(scored_df()),
            service.tournament_cache_key(tournament_id, f'{stem}.pbn'),
            {'tournament_id': tournament_id, 'tournament_name': stem, 'closed': True},
        )

    # The endpoint functions are called directly: Query(...) defaults are not
    # resolved outside a request, so every argument is passed.
    def _boards(self, entrant=None, only_my_boards=True, columns=None, tournament=None):
        return self.server.boards(
            tournament=tournament, key=None, entrant=entrant, only_my_boards=only_my_boards,
            columns=columns, limit=500)

    def _sql(self, sql, entrant=None):
        return self.server.sql(self.server.SqlRequest(sql=sql, entrant=entrant))

    def _refused(self, call):
        with self.assertRaises(self.server.HTTPException) as caught:
            call()
        self.assertEqual(caught.exception.status_code, 422)
        return caught.exception.detail

    def test_entrants_default_to_the_latest_tournament(self):
        body = self.server.entrants(tournament=None)
        self.assertEqual(body['meta']['tournament_id'], self.LISTED[0]['id'])
        self.assertEqual([e['entrant'] for e in body['entrants']], ['BEN', 'Alice', 'Lia'])
        self.assertEqual(self.generated, [])

    def test_entrants_name_the_winner(self):
        self.official = [{'name': 'Lia', 'rank': 1}, {'name': 'Alice', 'rank': 2}]
        body = self.server.entrants(tournament=None)
        self.assertEqual(body['winner'], 'Lia')
        self.assertEqual(body['ranking'], ['Lia', 'Alice', 'BEN'])

    def test_entrants_without_the_site_ranking_name_the_best_percentage(self):
        self.official = self.server.brill_client.BrillError('site down')
        body = self.server.entrants(tournament=None)
        self.assertEqual(body['winner'], 'BEN')
        self.assertEqual(body['ranking'], ['BEN', 'Alice', 'Lia'])

    def test_boards_for_one_entrant(self):
        body = self._boards(entrant='alice', columns='Board,Entrant,Contract')
        self.assertEqual(body['meta']['entrant'], 'Alice')
        self.assertEqual(body['rows'], [
            {'Board': 1, 'Entrant': 'Alice', 'Contract': '3NN'},
            {'Board': 2, 'Entrant': 'Alice', 'Contract': '4SXS'},
        ])
        self.assertEqual(self._boards(entrant='alice', only_my_boards=False)['row_count'], 5)
        self.assertEqual(self._boards()['row_count'], 5)

    def test_sql_flags_follow_the_entrant(self):
        sql = "SELECT COUNT(*) AS n FROM self WHERE Boards_I_Declared"
        self.assertEqual(self._sql(sql, entrant='Alice')['rows'], [{'n': 1}])
        self.assertEqual(self._sql(sql)['rows'], [{'n': 5}])

    def test_named_tournament_not_cached_is_generated_once(self):
        for _ in range(2):
            body = self.server.entrants(tournament='Daily IMP 2026-11-12')
            self.assertEqual(body['meta']['tournament_id'], self.LISTED[1]['id'])
        self.assertEqual(self.generated, [self.LISTED[1]['id']])

    def test_unknown_names_are_refused_with_a_hint(self):
        self.assertIn('Daily MP', self._refused(lambda: self.server.entrants(tournament='Monthly teams')))
        self.assertIn('Entrants: Alice, BEN, Lia', self._refused(lambda: self._boards(entrant='Zed')))

    def test_bad_sql_is_a_422(self):
        self.assertIn('SQL error', self._refused(lambda: self._sql('SELECT nope FROM self')))


if __name__ == '__main__':
    unittest.main()
