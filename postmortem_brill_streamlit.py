"""
postmortem-brill Streamlit application
"""

# Postmortem of the robot competitions at Brill Bridge Challenge
# (https://brill.aalborgdata.dk). Every entrant, human or robot, plays the
# same boards South at their own table beside three Brill robots.
# Invoke using: streamlit run postmortem_brill_streamlit.py --server.port 8530
# Needs the API: python brill_postmortem_api_server.py


import streamlit as st
import streamlit_chat

import pathlib
import polars as pl
import duckdb
import json
import sys
import os

_APP_DIR = pathlib.Path(__file__).resolve().parent
_REQUIRED_LIBS = ('mlBridge', 'streamlitlib')

def _is_lib_dir(path: pathlib.Path) -> bool:
    return path.is_dir() and (path / '__init__.py').is_file()

_resolved_libs = []
for _name in _REQUIRED_LIBS:
    _candidates = (
        _APP_DIR / _name,
        _APP_DIR.parent / _name,
        _APP_DIR.parent.parent / _name,
    )
    _found = next((path for path in _candidates if _is_lib_dir(path)), None)
    if _found is None:
        raise FileNotFoundError(
            f"{_name} not found at " + " or ".join(str(path) for path in _candidates)
        )
    _resolved_libs.append(_found)
# Package root for import mlBridge.*; lib dirs first for legacy import streamlitlib.
for _p in {_APP_DIR, *(path.parent for path in _resolved_libs)}:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)
for _p in _resolved_libs:
    _s = str(_p)
    if _p.name == 'mlBridge':
        if _s not in sys.path:
            sys.path.append(_s)  # logging_config and friends
    else:
        if _s in sys.path:
            sys.path.remove(_s)
        sys.path.insert(0, _s)

from typing import Any, Dict, List, Optional

import streamlitlib
from mlBridge.mlBridgePostmortemLib import (
    HAND_EXPECTATIONS_TABLE,
    PostmortemBase,
    sidebar_report_buttons,
    when_connection_free,
)
from mlBridge.mlBridgeLib import cast_numeric_display_columns
import brill_postmortem_api_client as brill_api
from brill_board_flags import ALL_ENTRANTS, apply_board_flags


BRILL_SITE_URL = 'https://brill.aalborgdata.dk'
# How many of the latest closed tournaments the sidebar offers.
TOURNAMENT_LIMIT = 10

_TOURNAMENT_KEY = 'brill_tournament_select'
_ENTRANT_KEY = 'brill_entrant_select'


def get_session_duckdb_connection():
    """Get or create a DuckDB connection for the current session."""
    if 'con' not in st.session_state or st.session_state.con is None:
        st.session_state.con = duckdb.connect()
        st.session_state.con_register_name = 'self'
    return st.session_state.con


def ShowDataFrameTable(df, key, query='SELECT * FROM self', show_sql_query=True):
    if show_sql_query and st.session_state.show_sql_query:
        st.text(f"SQL Query: {query}")

    # DuckDB FROM-first syntax: add the table when the query does not name one.
    if 'from ' not in query.lower():
        query = 'FROM self ' + query

    try:
        con = get_session_duckdb_connection()
        result_df = cast_numeric_display_columns(when_connection_free(lambda: con.sql(query).pl()))
        st.text(f"Result is a dataframe of {len(result_df)} rows.")
        streamlitlib.ShowDataFrameTable(result_df, key)
    except Exception as e:
        st.error(f"duckdb exception: error:{e} query:{query}")
        return None

    return result_df


# ---- URL Query Parameter Sync ----
# Shareable/bookmarkable: ?tournament=<id>&entrant=<name>&show_sql_query=1&sd_samples=10
def _set_query_param(name: str, value: Any) -> None:
    """Set or remove a single URL query parameter to match a session state value."""
    if value is None or value == '':
        if name in st.query_params:
            del st.query_params[name]
    else:
        st.query_params[name] = str(value)


def sync_url_params_from_state() -> None:
    """Write the currently selected sidebar options back to the URL query string."""
    _set_query_param('tournament', st.session_state.get(_TOURNAMENT_KEY))
    # 'All entrants' is written too: without ?entrant= a tournament opens on
    # the winner.
    _set_query_param('entrant', st.session_state.get(_ENTRANT_KEY))
    show_sql = st.session_state.get('show_sql_query')
    _set_query_param(
        'show_sql_query',
        None if show_sql is None else ('1' if show_sql else '0'),
    )
    _set_query_param('sd_samples', st.session_state.get('single_dummy_sample_count'))


def apply_url_params_to_state() -> None:
    """Seed sidebar-related session state from URL query parameters."""
    qp = st.query_params

    if 'tournament' in qp and str(qp['tournament']).strip():
        st.session_state._url_tournament = str(qp['tournament']).strip()

    if 'entrant' in qp and str(qp['entrant']).strip():
        st.session_state._url_entrant = str(qp['entrant']).strip()

    if 'show_sql_query' in qp:
        v = str(qp['show_sql_query']).strip().lower()
        st.session_state.show_sql_query = v in ('1', 'true', 'yes', 'on')

    if 'sd_samples' in qp:
        try:
            n = int(qp['sd_samples'])
            if 1 <= n <= 1000:
                st.session_state.single_dummy_sample_count = n
        except (ValueError, TypeError):
            pass


def app_info() -> None:
    """Display app information"""
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita")
    st.caption(f"App:{st.session_state.app_datetime} Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}")
    from mlBridge.package_versions import package_versions_line
    st.caption(package_versions_line())
    try:
        from mlBridge.dds_ddss import engine_caption
        st.caption(engine_caption())
    except Exception as exc:
        st.caption(f"DDS: endplay (ddss status unavailable: {exc})")
    return


def chat_input_on_submit():
    prompt = st.session_state.main_prompt_chat_input_key
    ShowDataFrameTable(st.session_state.df, query=prompt, key='user_query_main_doit_key')


def sample_count_on_change():
    st.session_state.single_dummy_sample_count = st.session_state.single_dummy_sample_count_number_input
    st.session_state._brill_generate_force = True
    sync_url_params_from_state()
    change_game_state()


def show_sql_query_change():
    # toggle whether to show sql query
    st.session_state.show_sql_query = st.session_state.sql_query_checkbox
    sync_url_params_from_state()


# ---- Tournaments ----

def load_tournaments(refresh: bool = False) -> List[Dict[str, Any]]:
    """The latest closed tournaments, newest first. Fetched once per session."""
    if refresh or st.session_state.get('brill_tournaments') is None:
        try:
            payload = brill_api.tournaments(limit=TOURNAMENT_LIMIT)
            st.session_state.brill_tournaments = payload.get('tournaments', [])
            st.session_state.brill_tournaments_error = None
        except brill_api.BrillApiClientError as e:
            st.session_state.brill_tournaments = None
            st.session_state.brill_tournaments_error = str(e)
            return []
    return st.session_state.brill_tournaments or []


def tournament_by_id(tournament_id: Optional[str]) -> Optional[Dict[str, Any]]:
    for tournament in st.session_state.get('brill_tournaments') or []:
        if tournament.get('id') == tournament_id:
            return tournament
    return None


def tournament_label(tournament_id: str) -> str:
    tournament = tournament_by_id(tournament_id)
    return (tournament or {}).get('name') or tournament_id


def tournament_changed():
    # A new tournament has a new field: change_game_state picks its default entrant.
    st.session_state[_ENTRANT_KEY] = None
    change_game_state()


def change_game_state():

    st.markdown('<div style="height: 50px;"><a name="top-of-report"></a></div>', unsafe_allow_html=True)
    streamlitlib.render_report_status_caption(st)

    with st.spinner('Preparing Bridge Game Postmortem Report. A tournament not seen before takes a few minutes...'):
        reset_game_data() # wipe out all game state data
        st.session_state.button_title = 'Summarize' # a new game opens on the full postmortem
        st.session_state.brill_base_df = None

        tournament_id = st.session_state.get(_TOURNAMENT_KEY)
        if not tournament_id:
            sync_url_params_from_state()
            return
        tournament = tournament_by_id(tournament_id) or {}
        st.text(f"Selected tournament: {tournament.get('name') or tournament_id}")
        sync_url_params_from_state()

        force = bool(st.session_state.pop('_brill_generate_force', False))
        try:
            meta = brill_api.generate(
                tournament_id,
                sd_samples=st.session_state.single_dummy_sample_count,
                force=force,
            )
            df = brill_api.postmortem_dataframe(meta.get('key'))
            st.session_state.brill_pbn_text = brill_api.pbn_text(meta.get('key'))
        except brill_api.BrillApiClientError as e:
            st.error(str(e))
            return

        assert df.select(pl.col(pl.Object)).is_empty(), f"Found Object columns: {[col for col, dtype in df.schema.items() if dtype == pl.Object]}"
        st.session_state.brill_meta = meta
        st.session_state.brill_base_df = df
        st.session_state.session_id = tournament_id
        st.session_state.group_id = tournament_id
        st.session_state.game_description = meta.get('tournament_name') or tournament.get('name')
        st.session_state.tournament_date = (meta.get('starts_at') or '')[:10] or None
        st.session_state.game_url = meta.get('url') or f'{BRILL_SITE_URL}/#tournament?tournamentId={tournament_id}'
        # Keep an entrant already chosen (?entrant=, or a reload of the same
        # tournament); otherwise open on the default one.
        if st.session_state.get(_ENTRANT_KEY) not in [ALL_ENTRANTS] + entrants_of(df):
            st.session_state[_ENTRANT_KEY] = default_entrant(meta.get('key'), df)
        apply_entrant()

    return


def entrants_of(df: Optional[pl.DataFrame]) -> List[str]:
    """Entrant names in table order."""
    if df is None or 'Entrant' not in df.columns:
        return []
    return (
        df.select('Table', 'Entrant').unique().sort('Table', 'Entrant')['Entrant'].to_list()
    )


def default_entrant(key: Optional[str], df: pl.DataFrame) -> str:
    """The entrant a tournament opens on: the winner. The whole field if unknown."""
    try:
        name = brill_api.entrants(key).get('winner')
    except brill_api.BrillApiClientError:
        name = None
    return name if name in entrants_of(df) else ALL_ENTRANTS


def apply_entrant(keep_hand_expectations: bool = False):
    """Flag the loaded field for the chosen entrant and register it as 'self'."""
    df = st.session_state.get('brill_base_df')
    if df is None:
        return
    entrant = st.session_state.get(_ENTRANT_KEY) or ALL_ENTRANTS
    if entrant != ALL_ENTRANTS and entrant not in entrants_of(df):
        entrant = ALL_ENTRANTS
        st.session_state[_ENTRANT_KEY] = entrant

    # The entrant always sits South; the pair under review is always NS.
    st.session_state.pair_direction = 'NS'
    st.session_state.opponent_pair_direction = 'EW'
    st.session_state.player_direction = 'S'
    st.session_state.partner_direction = 'N'
    if entrant == ALL_ENTRANTS:
        st.session_state.player_id = ALL_ENTRANTS
        st.session_state.player_name = None
        st.session_state.partner_id = None
        st.session_state.partner_name = None
        st.session_state.pair_id = None
    else:
        row = df.filter(pl.col('Entrant').eq(entrant)).row(0, named=True)
        st.session_state.player_id = entrant
        st.session_state.player_name = entrant
        st.session_state.partner_id = row.get('Player_ID_N')
        st.session_state.partner_name = row.get('Player_Name_N')
        st.session_state.pair_id = row.get('Table')

    st.session_state.df = apply_board_flags(df, entrant)
    register_session_df(st.session_state.df, keep_hand_expectations=keep_hand_expectations)
    sync_url_params_from_state()


def register_session_df(df: pl.DataFrame, keep_hand_expectations: bool = False) -> None:
    """Register df as 'self' on a new DuckDB connection for this session.

    Streamlit starts the rerun for a sidebar change at once, while the run it
    replaces may still be inside a query, and DuckDB refuses a second call on
    a busy connection (ResourceInUseException). So a changed dataframe gets a
    connection of its own instead of being re-registered on the one the old
    run is using. The old connection is released when that run ends.

    The connection is published to the session only once its tables are in
    place. keep_hand_expectations carries the Hand Expectations table over:
    it depends on the deals, not on the entrant.
    """
    con = duckdb.connect()
    con.register(st.session_state.get('con_register_name') or 'self', df)
    hand_expectations = st.session_state.get('hand_expectations_df')
    if keep_hand_expectations and hand_expectations is not None:
        con.register(HAND_EXPECTATIONS_TABLE, hand_expectations)
    # PostmortemBase reads db_connection; this module reads con.
    st.session_state.con = con
    st.session_state.db_connection = con


def entrant_changed():
    st.session_state.sql_query_mode = False
    apply_entrant(keep_hand_expectations=True)


ASSISTANT_LOGO_URL = 'https://github.com/BSalita/acbl-postmortem/blob/master/assets/logo_assistant.gif?raw=true' # 🥸 todo: put into config. must have raw=true for github url.

# Sidebar report buttons, keys of favorites['Buttons']. No AI_Predictions: this app makes no predictions.
REPORT_BUTTONS = ('Summarize', 'Summarize_Briefly', 'Hand_Expectations')


def render_sidebar_brand(title: str) -> None:
    """Gif and short title at the top of the sidebar, as in MortyBridgeChatbot."""
    logo_url = st.session_state.get("assistant_logo", ASSISTANT_LOGO_URL)
    st.sidebar.markdown(
        (
            '<div style="display:flex;align-items:center;gap:0.6rem;margin:0 0 0.75rem 0;">'
            f'<img src="{logo_url}" alt="Morty" width="40" height="40" '
            'style="width:2.5rem;height:2.5rem;border-radius:50%;flex:0 0 auto;">'
            f'<div style="font-size:1.05rem;font-weight:700;line-height:1.25;">{title}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def create_sidebar():
    render_sidebar_brand("Brill Postmortem")
    st.sidebar.caption('Build:'+st.session_state.app_datetime)

    tournaments = load_tournaments()
    if not tournaments:
        error = st.session_state.get('brill_tournaments_error')
        st.sidebar.error(error or 'No closed Brill tournaments were found.')
        if st.sidebar.button('Retry', key='brill_retry_tournaments_key'):
            load_tournaments(refresh=True)
            st.rerun()
        return

    ids = [t['id'] for t in tournaments]
    if st.session_state.get(_TOURNAMENT_KEY) not in ids:
        # First run: ?tournament= if it is one of the latest, else the latest.
        wanted = st.session_state.pop('_url_tournament', None)
        st.session_state[_TOURNAMENT_KEY] = wanted if wanted in ids else ids[0]
    st.sidebar.selectbox(
        'Tournament',
        options=ids,
        format_func=tournament_label,
        key=_TOURNAMENT_KEY,
        on_change=tournament_changed,
        help=f'The latest {TOURNAMENT_LIMIT} closed tournaments at {BRILL_SITE_URL}, newest first.',
    )

    entrants = entrants_of(st.session_state.get('brill_base_df'))
    if entrants:
        options = [ALL_ENTRANTS] + entrants
        if st.session_state.get(_ENTRANT_KEY) not in options:
            st.session_state[_ENTRANT_KEY] = ALL_ENTRANTS
        st.sidebar.selectbox(
            'Entrant',
            options=options,
            key=_ENTRANT_KEY,
            on_change=entrant_changed,
            help="Whose boards the report is about. A tournament opens on the winner. Every entrant sits South at their own table beside three Brill robots.",
        )

    if st.session_state.player_id is None:
        return

    st.session_state.pdf_link = st.sidebar.empty()

    meta = st.session_state.get('brill_meta') or {}
    if st.session_state.get('brill_pbn_text'):
        st.sidebar.download_button(
            'Download PBN',
            data=st.session_state.brill_pbn_text,
            file_name=meta.get('pbn_name') or 'brill.pbn',
            mime='text/plain',
            key='brill_download_pbn_key',
            help='Every table of this tournament as one PBN file.',
        )

    with st.sidebar.expander('Developer Settings', False):
        # do not use .sidebar in expander. it's already in the sidebar.
        st.session_state.single_dummy_sample_count = st.number_input('Single Dummy Sample Count',value=st.session_state.single_dummy_sample_count,key='single_dummy_sample_count_number_input',on_change=sample_count_on_change,min_value=1,max_value=1000,step=1,help='Number of random deals to generate for calculating single dummy probabilities. Larger number (10 to 30) is more accurate but slower. Use 1 to 5 for fast, less accurate results.')

    read_configs()

    if st.session_state.get('df') is not None and st.session_state.get('favorites'):
        sidebar_report_buttons(get_session_duckdb_connection(), REPORT_BUTTONS)

    # Other Morty websites
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Other Morty websites**")
    st.sidebar.markdown("🔗 [Bridge Chatbot](https://chat.7nt.info)")
    st.sidebar.markdown("🔗 [ACBL Elo Ratings](https://acbl-elo.7nt.info)")
    st.sidebar.markdown("🔗 [ACBL Postmortem](https://acbl.postmortem.chat)")
    st.sidebar.markdown("🔗 [ACBL Statistics](https://acbl-stats.7nt.info)")
    st.sidebar.markdown("🔗 [Brill Postmortem](https://brill.postmortem.chat)")
    st.sidebar.markdown("🔗 [FFBridge Elo Ratings](https://ffbridge-elo.7nt.info)")
    st.sidebar.markdown("🔗 [FFBridge Postmortem](https://ffbridge.postmortem.chat)")
    st.sidebar.markdown("🔗 [FFBridge Statistics](https://ffbridge-stats.7nt.info)")
    st.sidebar.markdown("🔗 [PBN/LIN Postmortem](https://pbn.postmortem.chat)")

    # Show SQL query - shows the SQL above each SQL-produced grid; synced with ?show_sql_query=
    st.sidebar.markdown("---")
    st.sidebar.checkbox('Show SQL query',value=st.session_state.show_sql_query,key='sql_query_checkbox',on_change=show_sql_query_change,help='Show the SQL query used to produce each table.')

    return


def read_configs():
    """Load default.favorites.json. Entrants have no favorites files of their own."""
    st.session_state.default_favorites_file = _APP_DIR / 'default.favorites.json'
    st.session_state.player_id_favorites = None
    st.session_state.debug_favorites = None

    if st.session_state.default_favorites_file.exists():
        with open(st.session_state.default_favorites_file, 'r', encoding='utf-8') as f:
            st.session_state.favorites = json.load(f)
    else:
        st.session_state.favorites = None
        return

    # Every prompt of the full report must be a vetted prompt.
    if 'missing_in_summarize' not in st.session_state:
        summarize_prompts = st.session_state.favorites['Buttons']['Summarize']['prompts']
        vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']
        st.session_state.summarize_keys = {p.lstrip('@') for p in summarize_prompts}
        st.session_state.vetted_keys = set(vetted_prompts.keys())
        st.session_state.missing_in_vetted = st.session_state.summarize_keys - st.session_state.vetted_keys
        assert len(st.session_state.missing_in_vetted) == 0, f"Oops. {st.session_state.missing_in_vetted} not in {st.session_state.vetted_keys}."
        st.session_state.missing_in_summarize = st.session_state.vetted_keys - st.session_state.summarize_keys
    return


def reset_game_data():

    # Default values for session state variables
    reset_defaults = {
        'game_description_default': None,
        'group_id_default': None,
        'session_id_default': None,
        'section_name_default': None,
        'player_id_default': None,
        'partner_id_default': None,
        'player_name_default': None,
        'partner_name_default': None,
        'player_direction_default': None,
        'partner_direction_default': None,
        'pair_id_default': None,
        'pair_direction_default': None,
        'opponent_pair_direction_default': None,
    }

    for key, value in reset_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # A new game starts from the defaults.
    reset_session_vars = {
        'df': None,
        'game_description': st.session_state.game_description_default,
        'group_id': st.session_state.group_id_default,
        'session_id': st.session_state.session_id_default,
        'section_name': st.session_state.section_name_default,
        'player_id': st.session_state.player_id_default,
        'partner_id': st.session_state.partner_id_default,
        'player_name': st.session_state.player_name_default,
        'partner_name': st.session_state.partner_name_default,
        'player_direction': st.session_state.player_direction_default,
        'partner_direction': st.session_state.partner_direction_default,
        'pair_id': st.session_state.pair_id_default,
        'pair_direction': st.session_state.pair_direction_default,
        'opponent_pair_direction': st.session_state.opponent_pair_direction_default,
        'tournament_date': None,
        'game_url': None,
        'brill_meta': None,
        'brill_pbn_text': None,
        'analysis_started': False,
        'vetted_prompts': [],
        'pdf_assets': [],
        'sql_query_mode': False,
        'sql_queries': [],
        'game_urls_d': {},
        'tournament_session_urls_d': {},
    }

    for key, value in reset_session_vars.items():
        st.session_state[key] = value

    return


def initialize_website_specific():

    st.session_state.assistant_logo = ASSISTANT_LOGO_URL
    st.session_state.guru_logo = 'https://github.com/BSalita/acbl-postmortem/blob/master/assets/logo_guru.png?raw=true' # 🥷todo: put into config file. must have raw=true for github url.
    st.session_state.game_results_url_default = None
    st.session_state.game_name = 'brill'
    st.session_state.organization_name = 'Brill Bridge Challenge'
    st.session_state.game_results_url = st.session_state.game_results_url_default

    streamlit_chat.message(
        "Hi. I'm Morty. Your friendly postmortem chatbot. I only want to chat about the robot competitions at Brill Bridge Challenge.", key='intro_message_1', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "I'm optimized for large screen devices such as a notebook or monitor. Do not use a smartphone.", key='intro_message_2', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "I start with the latest closed tournament. Choose another tournament or entrant, or the whole field, in the left sidebar.", key='intro_message_3', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "I'm just a Proof of Concept so don't double me.", key='intro_message_5', logo=st.session_state.assistant_logo)
    app_info()
    return


class PostmortemBrill(PostmortemBase):
    """postmortem-brill Streamlit application."""

    def __init__(self):
        super().__init__()

    def initialize_session_state(self):
        """Initialize app-specific session state."""
        self.initialize_common_session_state()

        # "Show SQL query" sidebar checkbox defaults to unchecked (the base class
        # default depends on STREAMLIT_ENV); ?show_sql_query=1 overrides below.
        st.session_state.show_sql_query = False

        brill_specific_vars = {
            'brill_tournaments': None,
            'brill_tournaments_error': None,
            'brill_base_df': None,
            'brill_meta': None,
            'brill_pbn_text': None,
            'save_intermediate_files': False,
        }
        for key, value in brill_specific_vars.items():
            if key not in st.session_state:
                st.session_state[key] = value

        self.reset_game_data()
        self.initialize_website_specific()

        # URL params take precedence over defaults (after reset).
        apply_url_params_to_state()
        url_entrant = st.session_state.pop('_url_entrant', None)
        if url_entrant:
            st.session_state[_ENTRANT_KEY] = url_entrant
        return

    def reset_game_data(self):
        """Reset game data."""
        self.reset_common_game_data()
        reset_game_data()

    def initialize_website_specific(self):
        """Initialize app-specific components."""
        initialize_website_specific()

    def process_prompt_macros(self, sql_query):
        """Process app-specific prompt macros."""
        return super().process_prompt_macros(sql_query)

    def create_sidebar(self):
        """Create app-specific sidebar."""
        create_sidebar()

    def create_ui(self):
        """Main UI plus URL sync (same pattern as ffbridge/ACBL)."""
        super().create_ui()
        sync_url_params_from_state()

    def write_report(self):
        """Use the standard report generation from base class."""
        self.write_standard_report()

    def ask_sql_query(self):
        """Use the standard SQL query interface from base class."""
        self.ask_standard_sql_query()

    def main(self):
        """Main entry: on a cold start, load the selected (latest) tournament."""
        st.set_page_config(page_icon=ASSISTANT_LOGO_URL, layout="wide", initial_sidebar_state="expanded")
        if 'first_time' not in st.session_state:
            self.initialize_session_state()
            self.create_sidebar()
            if st.session_state.get(_TOURNAMENT_KEY):
                # ?entrant= survives the load; a sidebar change resets it.
                change_game_state()
                # Rerun so create_ui writes the report instead of the intro.
                if st.session_state.get('df') is not None:
                    st.rerun()
                    return
        else:
            self.create_ui()
        sync_url_params_from_state()


if __name__ == "__main__":
    st.set_page_config(page_icon=ASSISTANT_LOGO_URL, layout="wide", initial_sidebar_state="expanded")
    if 'app' not in st.session_state:
        st.session_state.app = PostmortemBrill()
    st.session_state.app.main()
