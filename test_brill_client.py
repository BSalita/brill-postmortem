"""brill_client against a faked Supabase. No network."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import brill_client
import test_brill_pbn


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.reason = 'reason'
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise brill_client.requests.HTTPError(self.status_code)


def _iso(hours_from_now):
    return (datetime.now(timezone.utc) + timedelta(hours=hours_from_now)).isoformat()


TOURNAMENTS = [
    {'id': 't-open', 'name': 'Daily IMP — 2026-11-13', 'status': 'in_progress',
     'tournament_type': 'imp_across', 'scoring_method': 'imp', 'ends_at': _iso(+12),
     'starts_at': '2026-11-13T11:00:00+00:00'},
    {'id': 't-closed', 'name': 'Daily IMP — 2026-11-12', 'status': 'completed',
     'tournament_type': 'imp_across', 'scoring_method': 'imp', 'ends_at': _iso(-6),
     'starts_at': '2026-11-12T11:00:00+00:00'},
    {'id': 't-late', 'name': 'Daily MP — 2026-11-12', 'status': 'in_progress',
     'tournament_type': 'mp_across', 'scoring_method': 'mp', 'ends_at': _iso(-6),
     'starts_at': '2026-11-12T11:00:00+00:00'},
]


class FakeSupabase:
    """Answers the handful of requests the client makes."""

    def __init__(self):
        self.sign_ins = 0
        self.gets = []

    def post(self, url, **kwargs):
        self.sign_ins += 1
        if kwargs['json']['password'] != 'right':
            return _Response({'error_description': 'Invalid login credentials'}, 400)
        return _Response({'access_token': f'token{self.sign_ins}', 'expires_in': 3600})

    def get(self, url, params=None, headers=None, **kwargs):
        self.gets.append((url, dict(params or {}), dict(headers or {})))
        table = url.rsplit('/', 1)[-1]
        params = params or {}
        if table == 'tournaments':
            rows = TOURNAMENTS
            if 'id' in params:
                rows = [t for t in rows if f"eq.{t['id']}" == params['id']]
        elif table == 'tournament_rounds':
            rows = [{'round_number': 1, 'challenge_id': None}, {'round_number': 1, 'challenge_id': 'c1'}]
        elif table == 'challenges':
            rows = [test_brill_pbn.CHALLENGE]
        elif table == 'challenge_board_results':
            rows = test_brill_pbn.sample_rows()
        else:
            rows = []
        offset, limit = int(params.get('offset', 0)), int(params.get('limit', 1000))
        return _Response(rows[offset:offset + limit])


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeSupabase()
        brill_client._session.clear()
        patches = [
            mock.patch.object(brill_client, '_config', ('https://db.example', 'publishable')),
            mock.patch.object(brill_client, 'load_dotenv', lambda *a, **k: None),  # never the real .env
            mock.patch.object(brill_client.requests, 'post', self.fake.post),
            mock.patch.object(brill_client.requests, 'get', self.fake.get),
            mock.patch.dict(brill_client.os.environ, {'BRILL_EMAIL': 'me@example.com', 'BRILL_PASSWORD': 'right'}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(brill_client._session.clear)

    def test_lists_only_closed_field_events_newest_first(self):
        names = [t['id'] for t in brill_client.list_tournaments(limit=10)]
        self.assertEqual(names, ['t-closed', 't-late'])
        self.assertEqual(len(brill_client.list_tournaments(limit=10, include_open=True)), 3)
        self.assertEqual(len(brill_client.list_tournaments(limit=1)), 1)

    def test_reads_are_signed_in_and_sign_in_is_reused(self):
        brill_client.list_tournaments()
        brill_client.list_tournaments()
        self.assertEqual(self.fake.sign_ins, 1)
        _url, _params, headers = self.fake.gets[0]
        self.assertEqual(headers, {'apikey': 'publishable', 'Authorization': 'Bearer token1'})

    def test_missing_credentials_are_reported_before_any_request(self):
        with mock.patch.dict(brill_client.os.environ, {'BRILL_PASSWORD': ''}):
            with self.assertRaisesRegex(brill_client.BrillAuthError, 'BRILL_EMAIL and BRILL_PASSWORD'):
                brill_client.list_tournaments()
        self.assertEqual(self.fake.sign_ins, 0)

    def test_rejected_credentials(self):
        with mock.patch.dict(brill_client.os.environ, {'BRILL_PASSWORD': 'wrong'}):
            with self.assertRaisesRegex(brill_client.BrillAuthError, 'Invalid login credentials'):
                brill_client.list_tournaments()

    def test_field_pbn_uses_the_round_that_has_a_challenge(self):
        fetched = brill_client.field_pbn('t-closed')
        self.assertEqual(fetched['filename'], 'daily-imp-2026-11-12.pbn')
        self.assertEqual(fetched['tables'], 3)
        self.assertIn('[Event "Daily IMP — 2026-11-12"]', fetched['text'])
        self.assertIn('[Date "2026.11.12"]', fetched['text'])
        result_reads = [p for u, p, _h in self.fake.gets if u.endswith('challenge_board_results')]
        self.assertEqual(result_reads[0]['challenge_id'], 'eq.c1')

    def test_pagination_reads_every_row(self):
        with mock.patch.object(brill_client, '_PAGE_ROWS', 2):
            rows = brill_client._select('challenge_board_results', {'select': '*'})
        self.assertEqual(len(rows), len(test_brill_pbn.sample_rows()))


if __name__ == '__main__':
    unittest.main()
