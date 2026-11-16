# postmortem-brill

GitHub: https://github.com/BSalita/brill-postmortem

Postmortem of the robot competitions at [Brill Bridge Challenge](https://brill.aalborgdata.dk). Every day four eight-board events are played by humans and robots. Each entrant sits South at their own table beside three Brill robots, and everyone plays the same boards.

The app builds the same field PBN the site's download button gives (one record per table per board, e.g. `daily-imp-2026-11-12.pbn`) and runs it through the mlBridge augmentation used by postmortem-pbn. The sidebar offers the latest 10 closed tournaments and, within one, a single entrant or the whole field. The Entrant list starts with Winner (the site's final rank 1), then the entrants in finishing order, then the whole field. A tournament opens on the winner. An entrant picked from the list is a favorite: every tournament chosen afterwards opens on them if they played in it, otherwise on the winner. Picking Winner drops the favorite.

## Architecture

```
Streamlit UI  (port 8530)          brill_postmortem_api_client     https://brill.postmortem.chat
        │
        ▼
REST API  (port 8531)              brill_postmortem_api_server.py
        │
        ▼
Library
  brill_client.py                  sign in, list tournaments, read board results
  brill_pbn.py                     board results -> field PBN (port of the site's exporter)
  brill_postmortem_create.py       PBN -> dataframe, augment, cache
  brill_postmortem_service.py      list/load/SQL/schema over cache/
  brill_board_flags.py             Boards_* flags for one entrant or the whole field
```

Each tournament is cached as `cache/df-{key}.parquet`, with a `df-{key}.json` sidecar and the PBN itself as `cache/{key}.pbn`. `key` is the PBN stem plus the first 8 characters of the tournament id.

## Sign-in

The site only shows board results to a signed-in account, and only shows the whole field once an event has closed (06:00 the morning after). Put an account in `.env` (git-ignored):

```
BRILL_EMAIL=you@example.com
BRILL_PASSWORD=...
```

The Supabase project URL and publishable key are read from the site's own `js/supabase-client.js`. `BRILL_SUPABASE_URL` and `BRILL_SUPABASE_KEY` override them.

## Run

```powershell
pip install -U -r requirements.txt
python brill_postmortem_api_server.py
streamlit run postmortem_brill_streamlit.py --server.port 8530
```

API health: `http://127.0.0.1:8531/health` (`service`: `postmortem-brill-api`, `credentials_configured`).

URL parameters: `?tournament=<id>&entrant=<name>&show_sql_query=1&sd_samples=10`. `entrant=All entrants` is the whole field; without `entrant` the winner is used. An `entrant` in the URL counts as a pick from the list.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Process health, cache summary, whether credentials are set |
| GET | `/brill/tournaments` | Latest closed field events (`limit`, `include_open`) |
| POST | `/brill/generate` | Fetch one tournament (`tournament_id`), augment, cache |
| GET | `/brill/games` | Cached tournaments, newest first |
| GET | `/brill/dataset-info` | Cache directory and keys |
| GET | `/brill/entrants` | Entrants and standings of one tournament (`tournament`), their names in finishing order (`ranking`), and its `winner` |
| GET | `/brill/boards` | Per-table, per-board rows (`tournament`, `entrant`, `only_my_boards`, `columns`, `limit`) |
| POST | `/brill/sql` | DuckDB SQL against table `self` (`tournament`, `entrant`) |
| GET | `/brill/schema` | Column names and dtypes (`tournament`, `pattern`) |
| GET | `/brill/parquet` | Full augmented dataframe |
| GET | `/brill/pbn` | The field PBN the cache entry was built from |

`tournament` is a tournament id, a cache key, or a name such as `Daily IMP 2026-11-12`; omitted, it is the latest closed tournament. A tournament not cached yet is fetched and built on first use. `entrant` flags the `Boards_*` columns for that entrant (case and a unique part of the name are accepted); omitted, the whole field counts. `{Pair_Direction}` and its sibling macros are filled in (the entrant is always South).

MCP tools live in MortyBridgeMCP (`BRILL_POSTMORTEM_API_BASE_URL`, default `http://127.0.0.1:8531`): `brill_tournaments`, `brill_postmortem_entrants`, `brill_postmortem_boards`, `brill_postmortem_sql`, `brill_postmortem_schema`, `brill_postmortem_favorites`, and `brill_postmortem_report_start` / `_status` / `_result` / `_pdf`. Dataset coverage is `bridge_status(component="brill_postmortem")`.

## Notes

- Results are matchpointed across the tables of a board, for IMP events as well: mlBridge has no cross-IMP columns yet.
- Entrant columns: `Table`, `Entrant` (= `Player_ID_S`), `Pair_Number_NS`/`Pair_Number_EW` (the table number). The three robots at a table are `T{n}_N`, `T{n}_E`, `T{n}_W`.
- A passed-out board is written `[Declarer ""]`. The site writes `"-"`, which endplay cannot read.

## Tests

```powershell
python -m unittest test_brill_pbn test_brill_client test_brill_postmortem
```
