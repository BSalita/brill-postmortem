"""Build a field PBN from Brill Bridge Challenge board-result rows.

A Python port of the site's own export (js/pbn-manager.js,
PbnManager.generatePbnFromResults), which is what the "download PBN" button
on a tournament page runs in the browser. One record per entrant per board:
the entrant sits South, the three other seats are the event's robot, and
[Table] names the entrant's table.

Pure functions, no network: brill_client.py supplies the rows.
"""

from __future__ import annotations

import json
import re
from datetime import date as _date
from typing import Any, Dict, List, Optional

_SEATS = ('N', 'E', 'S', 'W')
_OPENING_LEADER = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
_RANK_ORDER = {r: i for i, r in enumerate('23456789TJQKA', start=2)}
_SUIT_SYMBOLS = {'♠': 'S', '♥': 'H', '♦': 'D', '♣': 'C'}
_CARD_RE = re.compile(r'^([SHDC])([AKQJT2-9])$', re.IGNORECASE)
_NOTE_TOKEN_RE = re.compile(r'^=(\d+)=$')
# PBN 2.1 [Vulnerable] values; the site stores several spellings.
_VULNERABLE = {
    'none': 'None', 'love': 'None', '-': 'None', '': 'None',
    'ns': 'NS', 'n-s': 'NS', 'n_s': 'NS',
    'ew': 'EW', 'e-w': 'EW', 'e_w': 'EW',
    'all': 'All', 'both': 'All',
}


def next_seat(seat: str) -> str:
    return _SEATS[(_SEATS.index(seat) + 1) % 4]


def bid_to_pbn(bid: Optional[str]) -> str:
    if not bid:
        return 'Pass'
    upper = str(bid).upper().strip()
    if upper in ('PASS', 'P', 'PA', '--'):
        return 'Pass'
    if upper in ('DOUBLE', 'DBL', 'D', 'X'):
        return 'X'
    if upper in ('REDOUBLE', 'RDBL', 'R', 'XX'):
        return 'XX'
    if re.fullmatch(r'\dN', upper):
        return upper + 'T'
    return upper


def normalize_suit(suit: Optional[str]) -> str:
    if not suit:
        return ''
    return _SUIT_SYMBOLS.get(suit, str(suit).upper())


def contract_to_pbn(contract: Any) -> str:
    """'4H', '5SX', '3NTXX', or 'Pass'. Accepts a string, a JSON string, or a dict."""
    if not contract:
        return 'Pass'
    if isinstance(contract, str):
        text = contract.strip()
        if text.startswith('{'):
            try:
                contract = json.loads(text)
            except ValueError:
                level = re.search(r'"level"\s*:\s*(\d)', text)
                suit = re.search(r'"suit"\s*:\s*"([^"]+)"', text)
                if level and suit:
                    return f"{level.group(1)}{normalize_suit(suit.group(1))}"
                return 'Pass'
        else:
            text = re.sub(r'redouble', 'XX', text, flags=re.IGNORECASE)
            text = re.sub(r'double', 'X', text, flags=re.IGNORECASE)
            if text.upper() in ('PASS', 'PASSOUT', 'P'):
                return 'Pass'
            return text
    if contract.get('suit') in ('PASSOUT', 'Pass'):
        return 'Pass'
    level = contract.get('level') or ''
    suit = normalize_suit(contract.get('suit') or '')
    redoubled = contract.get('redoubled') or contract.get('reXd')
    doubled = contract.get('doubled') or contract.get('Xd')
    penalty = 'XX' if redoubled else ('X' if doubled else '')
    return f"{level}{suit}{penalty}"


def vulnerable_to_pbn(vulnerability: Any) -> str:
    if isinstance(vulnerability, dict):  # an imported board: {NS: bool, EW: bool}
        ns, ew = bool(vulnerability.get('NS')), bool(vulnerability.get('EW'))
        return 'All' if ns and ew else 'NS' if ns else 'EW' if ew else 'None'
    text = str(vulnerability if vulnerability is not None else '').strip()
    return _VULNERABLE.get(text.lower(), text or 'None')


def format_robot_name(challenge: Optional[Dict[str, Any]]) -> str:
    """Name of the robot in the three seats beside the entrant, e.g. 'Robot (Brill)'."""
    challenge = challenge or {}

    def engine(name: Optional[str]) -> str:
        if not name:
            return 'Unknown'
        if name == 'neuralnetwork':
            return 'NN'
        if name == 'brill':
            return 'Brill'
        return name.upper()

    bid = engine(challenge.get('robot_bidding_engine') or 'brill')
    play = engine(challenge.get('robot_play_engine') or 'brill')
    return f"Robot ({bid if bid == play else bid + '/' + play})"


def is_field_challenge(challenge: Optional[Dict[str, Any]]) -> bool:
    challenge = challenge or {}
    return (
        bool(challenge.get('tournament_id'))
        and challenge.get('creator_plays') is False
        and (challenge.get('max_participants') or 0) > 2
    )


def _trick_winner(played: List[Dict[str, str]], trump: Optional[str]) -> str:
    lead_suit = played[0]['suit']
    winner = played[0]
    for current in played[1:]:
        if trump and current['suit'] == trump and winner['suit'] != trump:
            winner = current
        elif current['suit'] == winner['suit'] and winner['suit'] in (trump, lead_suit):
            if _RANK_ORDER[current['rank']] > _RANK_ORDER[winner['rank']]:
                winner = current
    return winner['player']


def play_order_to_pbn_columns(cards: List[str], declarer: Optional[str], contract_str: str) -> str:
    """Cards in true play order -> PBN [Play] rows, one fixed seat per column.

    The stored play runs in the order the cards were played, so each trick
    starts with whoever won the last one. A PBN Play section keeps each seat in
    its own column, starting with the opening leader. A seat with no card in an
    unfinished last trick gets '-'.
    """
    def as_rows() -> str:
        return '\n'.join(' '.join(cards[i:i + 4]) for i in range(0, len(cards), 4))

    if not cards or declarer not in _SEATS:
        return as_rows()
    strain = re.match(r'^\d(NT|[SHDC])', contract_str or '')
    trump = strain.group(1) if strain else 'NT'
    trump = None if trump == 'NT' else trump

    columns = []
    seat = _OPENING_LEADER[declarer]
    for _ in range(4):
        columns.append(seat)
        seat = next_seat(seat)

    rows = []
    leader = columns[0]
    for i in range(0, len(cards), 4):
        played = []
        seat = leader
        for token in cards[i:i + 4]:
            match = _CARD_RE.match(str(token))
            if not match:
                return as_rows()  # not a card list we understand; leave it alone
            played.append({'player': seat, 'suit': match.group(1).upper(), 'rank': match.group(2).upper()})
            seat = next_seat(seat)
        by_seat = {p['player']: p for p in played}
        rows.append(' '.join(
            f"{by_seat[col]['suit']}{by_seat[col]['rank']}" if col in by_seat else '-'
            for col in columns
        ))
        if len(played) != 4:
            break
        leader = _trick_winner(played, trump)
    return '\n'.join(rows)


def _tokens(value: Any) -> List[Any]:
    """A stored list column: a JSON array (possibly already decoded) or space-separated text."""
    if isinstance(value, list):
        return value
    text = str(value or '').strip()
    if text.startswith('['):
        try:
            return json.loads(text)
        except ValueError:
            pass
    return [part for part in text.split() if part]


def _format_auction(auction: Any) -> tuple[str, List[tuple[int, str]]]:
    """PBN auction text (4 calls per line) and its [Note] entries."""
    notes: List[tuple[int, str]] = []
    if not auction:
        return '', notes
    text = auction.strip() if isinstance(auction, str) else ''
    if isinstance(auction, list) or text.startswith('['):
        bids = []
        for entry in _tokens(auction):
            if isinstance(entry, dict):
                bid = bid_to_pbn(entry.get('bid'))
                explanation = (entry.get('explanation') or '').strip()
                if explanation:
                    notes.append((len(notes) + 1, explanation))
                    bid = f"{bid} ={len(notes)}="
                bids.append(bid)
            else:
                bids.append(bid_to_pbn(entry))
    elif '-' in text and ' ' not in text:
        bids = [bid_to_pbn(b) for b in text.split('-') if b]  # "P-P-2C-P-2D-P"
    else:
        bids = [p if _NOTE_TOKEN_RE.match(p) else bid_to_pbn(p) for p in text.split()]
    lines = [' '.join(bids[i:i + 4]) for i in range(0, len(bids), 4)]
    return '\n'.join(lines), notes


def format_timing_tag(timing: Optional[Dict[str, Any]], tag: str = 'DecisionTime') -> str:
    """'\\n[DecisionTime "bid=5:852 declare=10:4500"]', or '' when there is no timing."""
    if not isinstance(timing, dict):
        return ''
    parts = []
    for phase in ('bid', 'lead', 'declare', 'defence'):
        entry = timing.get(phase)
        if isinstance(entry, dict) and (entry.get('n') or entry.get('ms')):
            parts.append(f"{phase}={entry.get('n') or 0}:{round(entry.get('ms') or 0)}")
    return f'\n[{tag} "{" ".join(parts)}"]' if parts else ''


def _tag_text(text: Any) -> str:
    """A PBN tag value: one line, no double quotes."""
    return re.sub(r'\s+', ' ', str(text or '')).replace('"', "'").strip()


def format_board_record(
    board: Dict[str, Any],
    board_num: int,
    player_name: str,
    robot_name: str,
    room: Optional[str],
    scoring_method: Optional[str],
    reference_board: Optional[Dict[str, Any]] = None,
    table: int = 1,
) -> str:
    contract_str = contract_to_pbn(board.get('contract'))
    auction_str, notes = _format_auction(board.get('auction_pbn'))

    # A robot's row often carries placeholder dealer/vulnerability and no deal;
    # the first entrant's record of the same board is the source of truth.
    ref = reference_board or {}
    dealer = ref['dealer'] if ref.get('dealer') and ref['dealer'] != 'N' else (
        board.get('dealer') or ref.get('dealer') or 'N')
    ref_vul = ref.get('vulnerability')
    vulnerability = ref_vul if ref_vul and ref_vul != 'None' else (
        board.get('vulnerability') or ref_vul or 'None')
    deal_str = ref.get('deal_pbn') or board.get('deal_pbn') or ''
    if deal_str and not re.match(r'^[NESW]:', deal_str):
        deal_str = f"{dealer}:{deal_str}"

    declarer = board.get('declarer')
    is_passout = contract_str == 'Pass' or not declarer
    # The site writes '-' for a passed-out board; PBN (and endplay) want an empty value.
    declarer_str = '' if is_passout else declarer
    opening_leader = '' if is_passout else _OPENING_LEADER.get(declarer, 'E')

    play_str = ''
    cards = [str(c) for c in _tokens(board.get('play_pbn'))]
    if cards:
        play_str = play_order_to_pbn_columns(cards, declarer, contract_str)

    notes_section = ''.join(f'\n[Note "{n}:{_tag_text(t)}"]' for n, t in notes)
    timing = board.get('timing')
    timing_tag = format_timing_tag(timing) + format_timing_tag(
        timing.get('robot') if isinstance(timing, dict) else None, 'EngineTime')
    seat_tag = f'[Room "{room}"]' if room else f'[Table "{table}"]'
    result = '0' if is_passout else str(board.get('tricks') or 0)

    return (
        f'[Board "{board_num}"]\n'
        f'[West "{_tag_text(robot_name)}"]\n'
        f'[North "{_tag_text(robot_name)}"]\n'
        f'[East "{_tag_text(robot_name)}"]\n'
        f'[South "{_tag_text(player_name)}"]\n'
        f'[Dealer "{dealer}"]\n'
        f'[Vulnerable "{vulnerable_to_pbn(vulnerability)}"]\n'
        f'[Deal "{deal_str}"]\n'
        f'[Scoring "{"IMP" if scoring_method == "imp" else "MP"}"]\n'
        f'{seat_tag}\n'
        f'[Declarer "{declarer_str}"]\n'
        f'[Contract "{contract_str}"]\n'
        f'[Result "{result}"]{timing_tag}\n'
        f'[Auction "{dealer}"]\n'
        f'{auction_str or "Pass Pass Pass Pass"}{notes_section}\n'
        f'[Play "{opening_leader or "E"}"]\n'
        f'{play_str or "*"}\n'
        '\n'
    )


def group_board_results(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """challenge_board_results rows -> {participant_id: {participant, boards}}, in row order."""
    by_participant: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        entry = by_participant.setdefault(
            row['participant_id'], {'participant': row.get('participant'), 'boards': []})
        entry['boards'].append(row)
    return by_participant


def participant_name(participant: Optional[Dict[str, Any]], robot_name: str) -> str:
    if not participant:
        return 'Unknown'
    if participant.get('participant_type') == 'robot':
        return participant.get('display_name') or robot_name
    return participant.get('display_name') or 'Unknown'


def generate_pbn_from_results(
    challenge: Dict[str, Any],
    board_results: Dict[str, Dict[str, Any]],
    event: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """The whole field as one PBN: board 1 at every table, then board 2, ...

    event/date fill [Event]/[Date] (date as YYYY-MM-DD). The site's own
    download leaves them as 'Challenge Match' and today.
    """
    entrants = sorted(
        board_results.values(),
        key=lambda entry: (entry.get('participant') or {}).get('join_order') or 0,
    )
    use_tables = is_field_challenge(challenge) or len(entrants) > 2
    robot_name = format_robot_name(challenge)
    date_str = (date or _date.today().isoformat())[:10].replace('-', '.')
    out = [
        '% PBN 2.1\n'
        '% EXPORT\n'
        f'[Event "{_tag_text(event or "Challenge Match")}"]\n'
        '[Site "Brill Bridge Challenge"]\n'
        f'[Date "{date_str}"]\n'
    ]
    board_count = challenge.get('board_count') or max(
        (b.get('board_number') or 0 for e in entrants for b in e['boards']), default=0)
    for board_num in range(1, board_count + 1):
        reference = None
        if entrants:
            reference = next(
                (b for b in entrants[0]['boards'] if b.get('board_number') == board_num), None)
        for index, entry in enumerate(entrants):
            board = next((b for b in entry['boards'] if b.get('board_number') == board_num), None)
            if board is None:
                continue
            participant = entry.get('participant') or {}
            room = None if use_tables else ('Open' if index == 0 else 'Closed')
            out.append(format_board_record(
                board, board_num, participant_name(participant, robot_name), robot_name, room,
                challenge.get('scoring_method'), None if index == 0 else reference,
                participant.get('join_order') or index + 1,
            ))
    return ''.join(out)
