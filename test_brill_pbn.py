"""brill_pbn: board-result rows -> field PBN. No network."""

import unittest

import brill_pbn

DEAL_1 = 'N:AKQ2.K32.A32.432 J43.QJT.KQJ.KQJT T98.A987.T98.A98 765.654.7654.765'
DEAL_2 = 'E:KQJ.KQJ.KQJ4.KQJ T98.T98.T98.T987 A765.A76.A76.A65 432.5432.532.432'

CHALLENGE = {
    'id': 'c1', 'name': 'Daily IMP — 2026-11-12 — Field', 'board_count': 2,
    'scoring_method': 'imp', 'tournament_id': 't1', 'creator_plays': False,
    'max_participants': 500, 'robot_bidding_engine': 'brill', 'robot_play_engine': 'brill',
}


def _participant(pid, name, kind, order):
    return {'id': pid, 'display_name': name, 'participant_type': kind, 'join_order': order}


def sample_rows():
    """Three tables, two boards. Table 1 carries the deals; the robots lean on it."""
    alice = _participant('p1', 'Alice', 'human', 1)
    ben = _participant('p2', 'BEN', 'robot', 2)
    lia = _participant('p3', 'Lia', 'robot', 3)
    return [
        {'participant_id': 'p1', 'participant': alice, 'board_number': 1, 'dealer': 'N',
         'vulnerability': 'None', 'deal_pbn': DEAL_1, 'contract': '3NT', 'declarer': 'N',
         'tricks': 9, 'ns_score': 400,
         'auction_pbn': '[{"bid":"1NT","explanation":"15-17 \\"balanced\\"\\nno 5M"},"Pass","3NT","Pass","Pass","Pass"]',
         'play_pbn': 'CK C8 C5 C2 CQ CA C6 C3',
         'timing': {'bid': {'n': 2, 'ms': 852.4}, 'robot': {'bid': {'n': 4, 'ms': 120}}}},
        {'participant_id': 'p2', 'participant': ben, 'board_number': 1, 'dealer': 'N',
         'vulnerability': 'None', 'deal_pbn': '', 'contract': '1NT', 'declarer': 'N',
         'tricks': 9, 'ns_score': 150, 'auction_pbn': '1N-P-P-P', 'play_pbn': '["CK","C8","C5"]'},
        {'participant_id': 'p3', 'participant': lia, 'board_number': 1, 'dealer': 'N',
         'vulnerability': 'None', 'deal_pbn': DEAL_1, 'contract': None, 'declarer': None,
         'tricks': None, 'ns_score': 0, 'auction_pbn': 'P-P-P-P', 'play_pbn': ''},
        {'participant_id': 'p1', 'participant': alice, 'board_number': 2, 'dealer': 'E',
         'vulnerability': 'NS', 'deal_pbn': DEAL_2, 'contract': '4SX', 'declarer': 'S',
         'tricks': 8, 'ns_score': -500, 'auction_pbn': '1D Pass 4S X Pass Pass Pass',
         'play_pbn': 'H2 HK H8 HA'},
        {'participant_id': 'p2', 'participant': ben, 'board_number': 2, 'dealer': 'N',
         'vulnerability': 'None', 'deal_pbn': None, 'contract': {'level': 3, 'suit': '♠', 'doubled': False},
         'declarer': 'S', 'tricks': 8, 'ns_score': -100, 'auction_pbn': '1D-P-3S-P-P-P', 'play_pbn': None},
    ]


class GeneratePbnTests(unittest.TestCase):
    def setUp(self):
        grouped = brill_pbn.group_board_results(sample_rows())
        self.text = brill_pbn.generate_pbn_from_results(
            CHALLENGE, grouped, event='Daily IMP — 2026-11-12', date='2026-11-12T11:00:00+00:00')

    def test_header_carries_event_and_date(self):
        self.assertTrue(self.text.startswith('% PBN 2.1\n% EXPORT\n[Event "Daily IMP — 2026-11-12"]\n'))
        self.assertIn('[Date "2026.11.12"]', self.text)

    def test_one_record_per_table_per_board(self):
        self.assertEqual(self.text.count('[Board "1"]'), 3)
        self.assertEqual(self.text.count('[Board "2"]'), 2)
        self.assertNotIn('[Room ', self.text)
        for table in (1, 2, 3):
            self.assertIn(f'[Table "{table}"]', self.text)

    def test_entrant_sits_south_beside_the_robot(self):
        self.assertIn('[North "Robot (Brill)"]\n[East "Robot (Brill)"]\n[South "Alice"]', self.text)
        self.assertIn('[South "BEN"]', self.text)

    def test_robot_rows_inherit_deal_dealer_and_vulnerability(self):
        records = self.text.split('[Board "2"]')[1:]
        for record in records:
            self.assertIn('[Dealer "E"]', record)
            self.assertIn('[Vulnerable "NS"]', record)
            self.assertIn(f'[Deal "{DEAL_2}"]', record)

    def test_contracts(self):
        self.assertIn('[Contract "4SX"]\n[Result "8"]', self.text)
        self.assertIn('[Contract "3S"]', self.text)
        self.assertIn('[Declarer ""]\n[Contract "Pass"]\n[Result "0"]', self.text)

    def test_auction_notes_stay_on_one_line(self):
        self.assertIn('1NT =1= Pass 3NT Pass\nPass Pass\n[Note "1:15-17 \'balanced\' no 5M"]', self.text)
        self.assertIn('1NT Pass Pass Pass\n[Play "E"]', self.text)

    def test_timing_tags(self):
        self.assertIn('[Result "9"]\n[DecisionTime "bid=2:852"]\n[EngineTime "bid=4:120"]', self.text)


class PlayColumnTests(unittest.TestCase):
    def test_second_trick_is_rotated_back_to_seat_columns(self):
        # North declares 3NT; East leads. Trick 1: E CK, S C8, W C5, N C2 -> East wins.
        # Trick 2 (East leads again): CQ, S CA wins.
        rows = brill_pbn.play_order_to_pbn_columns(
            'CK C8 C5 C2 CQ CA C6 C3 H9'.split(), 'N', '3NT')
        # Columns are E S W N. Trick 3 is led by South; only South has played.
        self.assertEqual(rows, 'CK C8 C5 C2\nCQ CA C6 C3\n- H9 - -')

    def test_trump_wins_the_trick(self):
        # South declares 4S; West leads H2, North ruffs with S2 and leads D3.
        rows = brill_pbn.play_order_to_pbn_columns('H2 S2 HK HA D3'.split(), 'S', '4S')
        self.assertEqual(rows, 'H2 S2 HK HA\n- D3 - -')

    def test_unknown_tokens_are_left_in_play_order(self):
        self.assertEqual(brill_pbn.play_order_to_pbn_columns(['??', 'C2'], 'N', '3NT'), '?? C2')


class SmallHelperTests(unittest.TestCase):
    def test_bid_to_pbn(self):
        self.assertEqual([brill_pbn.bid_to_pbn(b) for b in ('P', 'D', 'R', '2N', '1nt', '3h', None)],
                         ['Pass', 'X', 'XX', '2NT', '1NT', '3H', 'Pass'])

    def test_contract_to_pbn(self):
        self.assertEqual(brill_pbn.contract_to_pbn('4H Redouble'.replace(' ', '')), '4HXX')
        self.assertEqual(brill_pbn.contract_to_pbn('{"level":6,"suit":"♦","redoubled":true}'), '6DXX')
        self.assertEqual(brill_pbn.contract_to_pbn({'suit': 'PASSOUT'}), 'Pass')

    def test_vulnerable_to_pbn(self):
        self.assertEqual([brill_pbn.vulnerable_to_pbn(v) for v in ('Both', 'N-S', 'ew', None, {'NS': True, 'EW': False})],
                         ['All', 'NS', 'EW', 'None', 'NS'])


if __name__ == '__main__':
    unittest.main()
