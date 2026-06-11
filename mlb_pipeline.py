"""
MLB data pipeline - fetch, normalize, compute, sort, narrate.

Stages:
    1. FETCH       - HTTP with disk cache + retry + backoff
    2. NORMALIZE   - raw JSON -> Pydantic models
    3. COMPUTE     - key plays, importance, OPS/ERA-based form labels
    4. ENRICH      - (optional) pybaseball/Statcast advanced metrics
    5. NARRATE     - (optional) single-model OR 3-model consensus
    6. SORT        - by importance, output to script-folder /output.json

Storage anchored to this file's folder. CLI:
    python mlb_pipeline.py
    python mlb_pipeline.py 2026-06-05 --limit 3 --teams "Yankees,Dodgers"
    python mlb_pipeline.py 2026-06-05 --statcast
    python mlb_pipeline.py 2026-06-05 --narrative single --model qwen2.5:14b
    python mlb_pipeline.py 2026-06-05 --narrative consensus
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Literal, Optional

import requests
from pydantic import BaseModel, Field

# =============================================================================
# CONFIG
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".mlb_cache"
OUTPUT_PATH = SCRIPT_DIR / "output.json"
PLAYERS_DB_PATH = SCRIPT_DIR / "players_db.json"
CACHE_DIR.mkdir(exist_ok=True)

BASE = "https://statsapi.mlb.com/api"
SEASON = date.today().year
REQUEST_SPACING_S = 0.1
DEFAULT_LLM_MODEL = "qwen2.5:14b"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mlb")


# =============================================================================
# HTTP layer
# =============================================================================
def _cache_path(url: str, params: dict | None) -> Path:
    raw = f"{url}?{json.dumps(params or {}, sort_keys=True)}"
    return CACHE_DIR / f"{hashlib.sha1(raw.encode()).hexdigest()}.json"


def _cache_read(path: Path, ttl_s: int) -> dict | None:
    if not path.exists() or (time.time() - path.stat().st_mtime) > ttl_s:
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def http_get(url: str, params: dict | None = None, ttl: int = 3600,
             max_retries: int = 3) -> dict | None:
    cpath = _cache_path(url, params)
    cached = _cache_read(cpath, ttl)
    if cached is not None:
        return cached

    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            cpath.write_text(json.dumps(data))
            time.sleep(REQUEST_SPACING_S)
            return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            sleep_s = 0.5 * (2 ** attempt)
            log.warning(f"GET failed ({attempt + 1}/{max_retries}): {e}; retry in {sleep_s}s")
            time.sleep(sleep_s)

    log.error(f"GET permanently failed: {url}")
    return None


# =============================================================================
# Safe coercion
# =============================================================================
def safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def fmt_slash_line(stat: dict) -> str:
    """Render AVG/OBP/SLG as '.265/.340/.480'. Handles values >= 1.000
    (e.g. 1.050 SLG) and missing/zero values gracefully."""
    def fmt(v):
        if v is None:
            return ".---"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return ".---"
        if f >= 1.0:
            return f"{f:.3f}"          # 1.050
        return f"{f:.3f}"[1:]          # 0.265 -> .265
    return f"{fmt(stat.get('avg'))}/{fmt(stat.get('obp'))}/{fmt(stat.get('slg'))}"


# =============================================================================
# MODELS
# =============================================================================
class KeyPlay(BaseModel):
    inning: str
    description: str
    # None when not provided by MLB (e.g. lagged WPA); avoids misleading 0/1 defaults
    wpa: Optional[float] = None
    leverage: Optional[float] = None
    # How this play was selected and ranked - "wpa", "captivatingIndex", or "scoring_heuristic"
    rank_metric: str = "wpa"
    rank_value: Optional[float] = None


FormLabel = Literal["hot", "warming", "steady", "cooling", "cold", "insufficient_data"]


class PlayerForm(BaseModel):
    person_id: int
    name: str
    role: Literal["hitter", "starting_pitcher", "relief_pitcher"]

    # Human-readable summaries (legacy / for UI)
    season_line: str
    recent_line: str
    delta_summary: str
    form_label: FormLabel
    pa_or_ip: float = 0.0

    # NEW: structured numeric stats for downstream model consumption
    season_stats: Optional[dict] = None    # raw numerics, same keys as recent_stats
    recent_stats: Optional[dict] = None    # raw numerics for the rolling window

    # NEW: handedness — critical for matchup prediction
    bat_side: Optional[str] = None         # "L" | "R" | "S"
    throw_hand: Optional[str] = None       # "L" | "R"
    primary_position: Optional[str] = None
    current_age: Optional[int] = None

    # NEW: platoon splits — biggest single matchup signal
    splits: Optional[dict] = None          # {vs_lhp:{...}, vs_rhp:{...}} (hitter)
                                           # {vs_lhb:{...}, vs_rhb:{...}} (pitcher)

    # Statcast block (now includes pitch_mix for pitchers)
    statcast: Optional[dict] = None


NarrativeMode = Literal["off", "single", "consensus"]


class Game(BaseModel):
    game_pk: int
    date: str
    home: str
    away: str
    home_score: int
    away_score: int
    status: str
    venue: str
    margin: int
    innings_played: int = 9  # from linescore; >9 means extra innings
    # Run-by-run inning breakdown: [{"inning": 1, "away": 0, "home": 2}, ...]
    linescore_innings: list[dict] = Field(default_factory=list)
    # Largest single-inning output: {"side": "home", "inning": 7, "runs": 10}
    biggest_inning: Optional[dict] = None
    # Pitching decisions from the live feed - fixes "losing pitcher unnamed"
    winning_pitcher: Optional[str] = None
    losing_pitcher: Optional[str] = None
    save_pitcher: Optional[str] = None
    # Team form entering this game (last 10 completed games)
    home_team_form: Optional[dict] = None
    away_team_form: Optional[dict] = None
    importance_score: float = 0.0
    importance_reasons: list[str] = Field(default_factory=list)
    key_plays: list[KeyPlay] = Field(default_factory=list)
    home_starter: Optional[PlayerForm] = None
    away_starter: Optional[PlayerForm] = None
    home_lineup: list[PlayerForm] = Field(default_factory=list)
    away_lineup: list[PlayerForm] = Field(default_factory=list)
    # Pitchers who actually appeared (starter + relievers), filled post-game
    # from the boxscore. Lets us include closers/setup men who blew or earned
    # the save and aren't known pre-game.
    home_pitchers: list[PlayerForm] = Field(default_factory=list)
    away_pitchers: list[PlayerForm] = Field(default_factory=list)
    narrative: str = ""
    narrative_metadata: Optional[dict] = None  # populated by consensus mode


# =============================================================================
# FETCH
# =============================================================================
def get_all_teams() -> list[str]:
    data = http_get(f"{BASE}/v1/teams", params={"sportId": 1}, ttl=7 * 86400)
    if not data:
        return []
    return sorted(t["name"] for t in data.get("teams", []) if t.get("active", True))


def get_team_id_map() -> dict[str, int]:
    """{team name: team id} for all active MLB teams."""
    data = http_get(f"{BASE}/v1/teams", params={"sportId": 1}, ttl=7 * 86400)
    if not data:
        return {}
    return {t["name"]: t["id"] for t in data.get("teams", [])
            if t.get("active", True) and "id" in t and "name" in t}


def get_team_roster(team_id: int, season: int) -> list[dict]:
    """Active roster: [{person_id, name, position}, ...]."""
    data = http_get(f"{BASE}/v1/teams/{team_id}/roster",
                    params={"rosterType": "active", "season": season},
                    ttl=6 * 3600)
    if not data:
        return []
    out = []
    for entry in data.get("roster", []):
        person = entry.get("person") or {}
        pos = (entry.get("position") or {}).get("abbreviation", "")
        if "id" in person:
            out.append({"person_id": person["id"],
                        "name": person.get("fullName", "?"),
                        "position": pos})
    return out


def build_team_season(team_name: str, season: int,
                      with_statcast: bool = False,
                      on_progress: "ProgressCB | None" = None) -> dict:
    """
    Season-stat profile for every player on a team's active roster -
    independent of any game. Pitchers (position P) get pitching stats;
    everyone else gets hitting stats. Returns a dict ready to serialize.
    """
    id_map = get_team_id_map()
    team_id = id_map.get(team_name)
    if team_id is None:
        # forgiving substring match
        matches = [tid for name, tid in id_map.items()
                   if team_name.lower() in name.lower()]
        team_id = matches[0] if matches else None
    if team_id is None:
        log.error(f"unknown team: {team_name}")
        return {}

    roster = get_team_roster(team_id, season)
    log.info(f"{team_name}: {len(roster)} players on active roster")
    players: list[dict] = []
    total = len(roster)
    for i, r in enumerate(roster):
        role = "starting_pitcher" if r["position"] == "P" else "hitter"
        window = 30 if role == "starting_pitcher" else 15
        pf = build_player_form(r["person_id"], r["name"], role,
                               season, window_days=window)
        if pf:
            if with_statcast:
                _try_statcast_enrich(pf, window_days=window)
            rec = pf.model_dump()
            rec["roster_position"] = r["position"]
            players.append(rec)
        if on_progress:
            on_progress(i + 1, total, r["name"])

    return {
        "team": team_name,
        "team_id": team_id,
        "season": season,
        "generated": date.today().isoformat(),
        "team_form": get_team_form(team_id, date.today().isoformat()) or None,
        "player_count": len(players),
        "players": players,
    }


def get_team_form(team_id: int, as_of: str, n_games: int = 10) -> dict:
    """
    Team form over the last `n_games` completed games before `as_of`:
    win-loss record, runs scored/allowed per game, current streak.
    Uses the schedule endpoint with a trailing date window.
    """
    end = date.fromisoformat(as_of)
    start = end - timedelta(days=25)  # generous window to find 10 games
    data = http_get(f"{BASE}/v1/schedule", params={
        "sportId": 1, "teamId": team_id,
        "startDate": start.isoformat(), "endDate": as_of,
    }, ttl=6 * 3600)
    if not data:
        return {}

    results = []  # (won: bool, runs_for: int, runs_against: int, date)
    for d_block in data.get("dates", []):
        for g in d_block.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            # Skip the as_of game itself (form = entering the game)
            if g.get("officialDate") == as_of:
                continue
            for side in ("home", "away"):
                side_team = g["teams"][side]["team"]
                if side_team.get("id") == team_id:
                    rf = safe_int(g["teams"][side].get("score"))
                    other = "away" if side == "home" else "home"
                    ra = safe_int(g["teams"][other].get("score"))
                    results.append((rf > ra, rf, ra, g.get("officialDate", "")))

    results.sort(key=lambda x: x[3], reverse=True)
    recent = results[:n_games]
    if not recent:
        return {}

    wins = sum(1 for w, *_ in recent if w)
    # Streak: consecutive same-result games from most recent
    streak_type = "W" if recent[0][0] else "L"
    streak = 0
    for w, *_ in recent:
        if (w and streak_type == "W") or (not w and streak_type == "L"):
            streak += 1
        else:
            break

    n = len(recent)
    return {
        "last_n": n,
        "wins": wins,
        "losses": n - wins,
        "runs_scored_pg": round(sum(r[1] for r in recent) / n, 2),
        "runs_allowed_pg": round(sum(r[2] for r in recent) / n, 2),
        "streak": f"{streak_type}{streak}",
    }


def get_schedule(d: str) -> list[dict]:
    data = http_get(f"{BASE}/v1/schedule", params={
        "sportId": 1, "date": d,
        "hydrate": "probablePitcher,lineups",
    }, ttl=600)
    if not data or not data.get("dates"):
        return []
    return data["dates"][0].get("games", [])


def get_live_feed(game_pk: int, is_final: bool) -> dict | None:
    ttl = 30 * 86400 if is_final else 60
    return http_get(f"{BASE}/v1.1/game/{game_pk}/feed/live", ttl=ttl)


def get_boxscore(game_pk: int, is_final: bool) -> dict | None:
    """Per-game boxscore - includes every player who actually appeared."""
    ttl = 30 * 86400 if is_final else 120
    return http_get(f"{BASE}/v1/game/{game_pk}/boxscore", ttl=ttl)


def get_player_stats(person_id: int, group: str, start: str, end: str,
                     season: int) -> dict:
    season_data = http_get(
        f"{BASE}/v1/people/{person_id}/stats",
        params={"stats": "season", "season": season, "group": group},
        ttl=6 * 3600,
    )
    window_data = http_get(
        f"{BASE}/v1/people/{person_id}/stats",
        params={"stats": "byDateRange", "startDate": start, "endDate": end,
                "group": group, "season": season},
        ttl=6 * 3600,
    )

    def extract(payload):
        if not payload:
            return {}
        try:
            return payload["stats"][0]["splits"][0]["stat"]
        except (IndexError, KeyError, TypeError):
            return {}

    return {"season": extract(season_data), "recent": extract(window_data)}


# --- New structured-stat helpers ---------------------------------------------

# Curated keys to extract from MLB API stat blobs into a flat numeric dict.
# Keys are the API field names; we rename to predictor-friendly shorter names.
_HITTER_KEYS = [
    ("avg", "avg", safe_float),
    ("obp", "obp", safe_float),
    ("slg", "slg", safe_float),
    ("ops", "ops", safe_float),
    ("babip", "babip", safe_float),
    ("homeRuns", "hr", safe_int),
    ("rbi", "rbi", safe_int),
    ("baseOnBalls", "bb", safe_int),
    ("strikeOuts", "so", safe_int),
    ("stolenBases", "sb", safe_int),
    ("atBats", "ab", safe_int),
    ("hits", "hits", safe_int),
    ("doubles", "2b", safe_int),
    ("triples", "3b", safe_int),
    ("plateAppearances", "pa", safe_int),
    ("gamesPlayed", "g", safe_int),
    ("hitByPitch", "hbp", safe_int),
    ("sacFlies", "sf", safe_int),
    ("groundIntoDoublePlay", "gidp", safe_int),
]

_PITCHER_KEYS = [
    ("era", "era", safe_float),
    ("whip", "whip", safe_float),
    ("strikeOutsPer9Inn", "k_per_9", safe_float),
    ("walksPer9Inn", "bb_per_9", safe_float),
    ("homeRunsPer9", "hr_per_9", safe_float),
    ("hitsPer9Inn", "h_per_9", safe_float),
    ("strikeoutWalkRatio", "k_bb_ratio", safe_float),
    ("inningsPitched", "ip", safe_float),
    ("wins", "w", safe_int),
    ("losses", "l", safe_int),
    ("saves", "sv", safe_int),
    ("holds", "hld", safe_int),
    ("gamesStarted", "gs", safe_int),
    ("gamesPlayed", "g", safe_int),
    ("strikeOuts", "k", safe_int),
    ("baseOnBalls", "bb", safe_int),
    ("hits", "h_allowed", safe_int),
    ("homeRuns", "hr_allowed", safe_int),
    ("earnedRuns", "er", safe_int),
    ("battersFaced", "bf", safe_int),
    ("avg", "avg_against", safe_float),
    ("obp", "obp_against", safe_float),
    ("slg", "slg_against", safe_float),
    ("ops", "ops_against", safe_float),
]


def _extract_numeric_stats(stat: dict, role: str) -> dict:
    """Pull a flat dict of numeric stats out of MLB's stat blob."""
    if not stat:
        return {}
    keys = _HITTER_KEYS if role == "hitter" else _PITCHER_KEYS
    return {alias: coerce(stat.get(api_name)) for api_name, alias, coerce in keys}


def get_player_meta(person_id: int) -> dict:
    """Handedness, position, age. Cached for a week (rarely changes)."""
    data = http_get(f"{BASE}/v1/people/{person_id}", ttl=7 * 86400)
    if not data:
        return {}
    try:
        p = data["people"][0]
        return {
            "bat_side": (p.get("batSide") or {}).get("code"),
            "throw_hand": (p.get("pitchHand") or {}).get("code"),
            "primary_position": (p.get("primaryPosition") or {}).get("abbreviation"),
            "current_age": safe_int(p.get("currentAge")) or None,
        }
    except (IndexError, KeyError, TypeError):
        return {}


def get_player_splits(person_id: int, group: str, season: int) -> dict:
    """
    vs-L / vs-R splits via the statSplits stats type.

    For hitters (group='hitting'): returns {vs_lhp, vs_rhp}.
    For pitchers (group='pitching'): returns {vs_lhb, vs_rhb}.

    sitCodes 'vl' = vs Left, 'vr' = vs Right (perspective depends on group).
    """
    data = http_get(
        f"{BASE}/v1/people/{person_id}/stats",
        params={
            "stats": "statSplits",
            "season": season,
            "group": group,
            "sitCodes": "vl,vr",
        },
        ttl=6 * 3600,
    )
    if not data:
        return {}

    role = "hitter" if group == "hitting" else "pitcher"
    key_map = {
        ("hitter", "vl"): "vs_lhp",
        ("hitter", "vr"): "vs_rhp",
        ("pitcher", "vl"): "vs_lhb",
        ("pitcher", "vr"): "vs_rhb",
    }

    out: dict = {}
    try:
        splits = data["stats"][0].get("splits", [])
    except (IndexError, KeyError, TypeError):
        return {}

    for s in splits:
        code = (s.get("split") or {}).get("code")
        stat = s.get("stat") or {}
        key = key_map.get((role, code))
        if key and stat:
            out[key] = _extract_numeric_stats(stat, role)
    return out


# =============================================================================
# NORMALIZE
# =============================================================================
def normalize_game(feed: dict | None) -> Optional[Game]:
    if not feed:
        return None
    try:
        gd = feed["gameData"]
        ld = feed["liveData"]
        line = ld.get("linescore", {})
        home_runs = safe_int(line.get("teams", {}).get("home", {}).get("runs"))
        away_runs = safe_int(line.get("teams", {}).get("away", {}).get("runs"))
        # currentInning on a Final game = total innings played
        innings = safe_int(line.get("currentInning"), 9) or 9
        # Pitching decisions (W/L/S) - available once the game is final
        decisions = ld.get("decisions") or {}
        # Inning-by-inning runs - lets narratives see big innings the top-5
        # key plays can't represent (e.g. a 10-run 7th)
        ls_innings = []
        biggest = None
        for inn in line.get("innings", []) or []:
            row = {
                "inning": safe_int(inn.get("num")),
                "away": safe_int((inn.get("away") or {}).get("runs")),
                "home": safe_int((inn.get("home") or {}).get("runs")),
            }
            ls_innings.append(row)
            for side in ("away", "home"):
                if biggest is None or row[side] > biggest["runs"]:
                    biggest = {"side": side, "inning": row["inning"],
                               "runs": row[side]}
        if biggest and biggest["runs"] < 3:
            biggest = None  # only noteworthy if 3+ runs
        return Game(
            game_pk=gd["game"]["pk"],
            date=gd["datetime"]["officialDate"],
            home=gd["teams"]["home"]["name"],
            away=gd["teams"]["away"]["name"],
            home_score=home_runs,
            away_score=away_runs,
            status=gd["status"]["detailedState"],
            venue=gd["venue"]["name"],
            margin=abs(home_runs - away_runs),
            innings_played=innings,
            linescore_innings=ls_innings,
            biggest_inning=biggest,
            winning_pitcher=(decisions.get("winner") or {}).get("fullName"),
            losing_pitcher=(decisions.get("loser") or {}).get("fullName"),
            save_pitcher=(decisions.get("save") or {}).get("fullName"),
        )
    except (KeyError, TypeError) as e:
        log.error(f"normalize_game failed: {e}")
        return None


def extract_key_plays(feed: dict, top_n: int = 5) -> list[KeyPlay]:
    """
    Three-tier ranking, falling back as needed:
      Tier 1: |homeWinProbabilityAdded| - best signal when populated.
      Tier 2: about.captivatingIndex - MLB's own 0-100 "interesting play"
              score, almost always populated and much better than chronology.
      Tier 3: isScoringPlay sorted by (inning DESC, rbi DESC) - surfaces
              late and big plays first when neither WPA nor CI are present.

    KeyPlay records which metric actually ranked it (`rank_metric`) and the
    value used (`rank_value`). When WPA isn't populated by MLB, we store
    `wpa=None` rather than a misleading 0.0.
    """
    try:
        all_plays = feed["liveData"]["plays"]["allPlays"]
    except (KeyError, TypeError):
        return []

    # Tier 1: WPA
    wpa_scored: list[tuple[float, dict]] = []
    for p in all_plays:
        wpa = p.get("about", {}).get("homeWinProbabilityAdded")
        if wpa is None:
            continue
        try:
            wpa_scored.append((abs(float(wpa)), p))
        except (TypeError, ValueError):
            continue

    if wpa_scored:
        wpa_scored.sort(key=lambda x: x[0], reverse=True)
        ranked: list[tuple[dict, float]] = [(p, score) for score, p in wpa_scored[:top_n]]
        metric = "wpa"
    else:
        # Tier 2: captivatingIndex
        ci_scored: list[tuple[float, dict]] = []
        for p in all_plays:
            ci = p.get("about", {}).get("captivatingIndex")
            if ci is None:
                continue
            try:
                ci_scored.append((float(ci), p))
            except (TypeError, ValueError):
                continue

        if ci_scored:
            log.info("no WPA; ranking by captivatingIndex")
            ci_scored.sort(key=lambda x: x[0], reverse=True)
            ranked = [(p, score) for score, p in ci_scored[:top_n]]
            metric = "captivatingIndex"
        else:
            # Tier 3: scoring plays heuristic - late and big plays first
            log.info("no WPA or captivatingIndex; using scoring-play heuristic")
            scoring = [p for p in all_plays
                       if p.get("about", {}).get("isScoringPlay")]
            scoring.sort(key=lambda p: (
                safe_int(p.get("about", {}).get("inning")),
                safe_int(p.get("result", {}).get("rbi")),
            ), reverse=True)
            ranked = [(p, 0.0) for p in scoring[:top_n]]
            metric = "scoring_heuristic"

    out = []
    for p, rank_value in ranked:
        about = p["about"]
        # Only store WPA/LI when MLB actually populated them
        raw_wpa = about.get("homeWinProbabilityAdded")
        raw_lev = about.get("leverageIndex")
        out.append(KeyPlay(
            inning=f"{'T' if about.get('halfInning') == 'top' else 'B'}{about.get('inning', '?')}",
            description=p.get("result", {}).get("description", ""),
            wpa=float(raw_wpa) if raw_wpa is not None else None,
            leverage=float(raw_lev) if raw_lev is not None else None,
            rank_metric=metric,
            rank_value=rank_value,
        ))
    return out


# =============================================================================
# COMPUTE
# =============================================================================
def score_importance(game: Game) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.3

    if game.margin == 1:
        score += 0.3
        reasons.append("one-run game")
    elif game.margin == 2:
        score += 0.15
    elif game.margin >= 7:
        score -= 0.25
        reasons.append("blowout")

    # Authoritative extra-innings check from the linescore, with key-play
    # innings as a fallback (in case linescore was missing).
    if game.innings_played > 9 or any(
        safe_int(kp.inning[1:]) > 9 for kp in game.key_plays
    ):
        score += 0.25
        reasons.append(f"extra innings ({max(game.innings_played, 10)})")

    # WPA and leverage are Optional now - skip Nones to avoid TypeErrors
    if game.biggest_inning and game.biggest_inning["runs"] >= 5:
        score += 0.15
        bi = game.biggest_inning
        reasons.append(f"{bi['runs']}-run inning ({'T' if bi['side']=='away' else 'B'}{bi['inning']})")

    leverages = [kp.leverage for kp in game.key_plays if kp.leverage is not None]
    max_lev = max(leverages, default=0.0)
    if max_lev > 3.0:
        score += 0.15
        reasons.append(f"high-leverage moment (LI {max_lev:.1f})")

    wpas = [abs(kp.wpa) for kp in game.key_plays if kp.wpa is not None]
    max_wpa = max(wpas, default=0.0)
    if max_wpa > 0.40:
        score += 0.15
        reasons.append(f"major win-probability swing ({max_wpa:+.2f})")

    return max(0.0, min(1.0, score)), reasons


def _classify_hitter(season_ops: float, recent_ops: float, pa: float) -> tuple[FormLabel, str]:
    if pa < 25:
        return "insufficient_data", "insufficient sample"
    delta = recent_ops - season_ops
    if delta > 0.150: label = "hot"
    elif delta > 0.050: label = "warming"
    elif delta < -0.150: label = "cold"
    elif delta < -0.050: label = "cooling"
    else: label = "steady"
    return label, f"OPS {'+' if delta >= 0 else ''}{int(delta * 1000)} pts vs season"


def _classify_pitcher(season_era: float, recent_era: float, ip: float) -> tuple[FormLabel, str]:
    if ip < 5:
        return "insufficient_data", "insufficient sample"
    delta = recent_era - season_era
    if delta < -1.0: label = "hot"
    elif delta < -0.3: label = "warming"
    elif delta > 1.0: label = "cold"
    elif delta > 0.3: label = "cooling"
    else: label = "steady"
    return label, f"ERA {'+' if delta >= 0 else ''}{delta:.2f} vs season"


def build_player_form(person_id: int, name: str, role: str, season: int,
                      window_days: int) -> Optional[PlayerForm]:
    today = date.today()
    start = (today - timedelta(days=window_days)).isoformat()
    group = "hitting" if role == "hitter" else "pitching"
    raw = get_player_stats(person_id, group, start, today.isoformat(), season)
    s, rec = raw["season"], raw["recent"]
    if not s:
        return None

    # Structured numeric versions of both stat blobs (for downstream model)
    role_key = "hitter" if role == "hitter" else "pitcher"
    season_numeric = _extract_numeric_stats(s, role_key)
    recent_numeric = _extract_numeric_stats(rec, role_key) if rec else None

    # Handedness + position + age (cached weekly, cheap after first run)
    meta = get_player_meta(person_id)

    # Platoon splits (vs L / vs R)
    splits = get_player_splits(person_id, group, season) or None

    if role == "hitter":
        pa = safe_float(rec.get("plateAppearances"))
        season_line = f"{fmt_slash_line(s)}, {safe_int(s.get('homeRuns'))} HR"
        if pa < 25:
            recent_line = f"only {int(pa)} PA in last {window_days}d"
        else:
            recent_line = (f"{fmt_slash_line(rec)}, "
                           f"{safe_int(rec.get('homeRuns'))} HR "
                           f"({int(pa)} PA, last {window_days}d)")
        label, delta = _classify_hitter(
            safe_float(s.get("ops")), safe_float(rec.get("ops")), pa,
        )
        return PlayerForm(
            person_id=person_id, name=name, role="hitter",
            season_line=season_line, recent_line=recent_line,
            delta_summary=delta, form_label=label, pa_or_ip=pa,
            season_stats=season_numeric, recent_stats=recent_numeric,
            splits=splits,
            bat_side=meta.get("bat_side"),
            throw_hand=meta.get("throw_hand"),
            primary_position=meta.get("primary_position"),
            current_age=meta.get("current_age"),
        )

    ip = safe_float(rec.get("inningsPitched"))
    season_line = (f"{s.get('era', '---')} ERA, {s.get('whip', '---')} WHIP, "
                   f"{s.get('strikeOutsPer9Inn', '--')} K/9")
    if ip < 5:
        recent_line = f"only {ip} IP in last {window_days}d"
    else:
        recent_line = (f"{rec.get('era', '---')} ERA, "
                       f"{rec.get('whip', '---')} WHIP "
                       f"({ip} IP, last {window_days}d)")
    label, delta = _classify_pitcher(
        safe_float(s.get("era"), 4.5), safe_float(rec.get("era"), 4.5), ip,
    )
    return PlayerForm(
        person_id=person_id, name=name,
        role=role if role != "hitter" else "starting_pitcher",  # preserve relief_pitcher
        season_line=season_line, recent_line=recent_line,
        delta_summary=delta, form_label=label, pa_or_ip=ip,
        season_stats=season_numeric, recent_stats=recent_numeric,
        splits=splits,
        bat_side=meta.get("bat_side"),
        throw_hand=meta.get("throw_hand"),
        primary_position=meta.get("primary_position"),
        current_age=meta.get("current_age"),
    )


def _try_statcast_enrich(form: PlayerForm, window_days: int):
    """Best-effort: enrich with Statcast if pybaseball is installed."""
    try:
        from statcast_form import enrich_form
    except ImportError:
        return
    today = date.today()
    enrich_form(
        form,
        recent_start=(today - timedelta(days=window_days)).isoformat(),
        recent_end=today.isoformat(),
        cache_dir=CACHE_DIR,
    )


def attach_forms(game: Game, raw_schedule_game: dict, season: int,
                 with_statcast: bool = False) -> Game:
    teams = raw_schedule_game.get("teams", {})
    for side in ("home", "away"):
        prob = teams.get(side, {}).get("probablePitcher")
        if prob and "id" in prob:
            pf = build_player_form(prob["id"], prob.get("fullName", "?"),
                                   "starting_pitcher", season, window_days=30)
            if pf:
                if with_statcast:
                    _try_statcast_enrich(pf, window_days=30)
                setattr(game, f"{side}_starter", pf)

    lineups = raw_schedule_game.get("lineups", {})
    for side in ("home", "away"):
        players = lineups.get(f"{side}Players") or []
        forms = []
        for p in players:
            if "id" not in p:
                continue
            pf = build_player_form(p["id"], p.get("fullName", "?"),
                                   "hitter", season, window_days=15)
            if pf:
                if with_statcast:
                    _try_statcast_enrich(pf, window_days=15)
                forms.append(pf)
        setattr(game, f"{side}_lineup", forms)
    return game


def attach_actual_pitchers(game: Game, game_pk: int, is_final: bool,
                           season: int, with_statcast: bool = False) -> Game:
    """
    Post-game enrichment: pull every pitcher who actually appeared from the
    boxscore. The pre-game schedule only tells us probable starters, so
    relievers (closers, setup arms) get missed - including ones critical to
    the outcome (blown saves, walk-off wins). Populates home_pitchers /
    away_pitchers, which OVERLAP with home_starter/away_starter when the
    starter pitched (starter is also a pitcher who appeared).
    """
    box = get_boxscore(game_pk, is_final=is_final)
    if not box:
        return game

    for side in ("home", "away"):
        try:
            team_data = box["teams"][side]
            pitcher_ids = team_data.get("pitchers", []) or []  # ordered by appearance
            player_dict = team_data.get("players", {}) or {}
        except (KeyError, TypeError):
            continue

        pitchers: list[PlayerForm] = []
        for idx, pid in enumerate(pitcher_ids):
            # Boxscore keys players as "ID123456"; pitchers[] is ordered by
            # appearance, so index 0 is the starter, rest are relievers.
            entry = player_dict.get(f"ID{pid}") or {}
            person = entry.get("person") or {}
            full_name = person.get("fullName", "?")
            role = "starting_pitcher" if idx == 0 else "relief_pitcher"
            window = 30 if idx == 0 else 15  # relievers: shorter, denser window
            pf = build_player_form(pid, full_name, role,
                                   season, window_days=window)
            if pf:
                if with_statcast:
                    _try_statcast_enrich(pf, window_days=window)
                pitchers.append(pf)
        setattr(game, f"{side}_pitchers", pitchers)
    return game


# =============================================================================
# NARRATE (single or consensus)
# =============================================================================
def _game_payload_for_llm(game: Game) -> dict:
    """
    Payload sent to the LLM. Spell out home/away explicitly - small models
    routinely confuse which team batted top vs bottom otherwise, and the
    `final` string makes the winner unambiguous (instead of just "5-3").
    """
    winner, wscore, loser, lscore = (
        (game.away, game.away_score, game.home, game.home_score)
        if game.away_score > game.home_score
        else (game.home, game.home_score, game.away, game.away_score)
    )
    return {
        "matchup": f"{game.away} (away) @ {game.home} (home)",
        "home_team": game.home,
        "away_team": game.away,
        "final": f"{winner} won {wscore}-{lscore} over {loser}",
        "home_score": game.home_score,
        "away_score": game.away_score,
        "margin": game.margin,
        "importance_reasons": game.importance_reasons,
        "winning_pitcher": game.winning_pitcher,
        "losing_pitcher": game.losing_pitcher,
        "save_pitcher": game.save_pitcher,
        "innings_played": game.innings_played,
        "runs_by_inning": game.linescore_innings,
        "biggest_inning_MENTION_IF_5PLUS_RUNS": game.biggest_inning,
        "key_plays": [kp.model_dump() for kp in game.key_plays],
    }


def _single_narrative(game: Game, model: str) -> Game:
    try:
        import ollama
    except ImportError:
        log.warning("ollama package not installed; skipping narrative")
        return game

    payload = _game_payload_for_llm(game)
    prompt = (
        "Write a 2-4 sentence factual recap of this baseball game. "
        "Use ONLY facts in the data below. No hype words, no speculation. "
        "Check runs_by_inning: if one team scored 5+ runs in a single inning, "
        "that is the headline - lead with it. Name the winning and losing "
        "pitchers if provided.\n\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        'Return ONLY JSON: {"narrative": "..."}'
    )
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.1},
        )
        game.narrative = json.loads(resp["message"]["content"]).get("narrative", "")
        game.narrative_metadata = {"mode": "single", "model": model}
    except Exception as e:
        log.warning(f"narrative failed for {game.game_pk}: {e}")
    return game


def _consensus_narrative(game: Game, voter_models: list[str],
                         verifier_model: Optional[str] = None) -> Game:
    try:
        from consensus import consensus_narrative
    except ImportError:
        log.warning("consensus module missing")
        return game

    result = consensus_narrative(
        _game_payload_for_llm(game),
        voter_models=voter_models,
        verifier_model=verifier_model,
    )
    if result:
        game.narrative = result.narrative
        game.narrative_metadata = {
            "mode": "consensus",
            "chosen_model": result.chosen_model,
            "voted_fields": result.voted_fields,
            "per_model": result.per_model,
            "disagreements": result.disagreements,
            "failed_models": result.failed_models,
            "verified": result.verified,
            "concerns": result.concerns,
            "verifier_model": result.verifier_model,
        }
    return game


def add_narrative(game: Game, mode: NarrativeMode, model: str,
                  voter_models: Optional[list[str]] = None,
                  verifier_model: Optional[str] = None) -> Game:
    if mode == "off":
        return game

    # Guard: if we have no real play data to ground the narrative, refuse to
    # generate one. Otherwise the LLM invents an inning-by-inning story from
    # the final score alone — which is fabrication, not summary.
    if not game.key_plays:
        log.warning(f"no key plays for {game.game_pk}; skipping narrative generation")
        winner, loser = (
            (game.away, game.home) if game.away_score > game.home_score
            else (game.home, game.away)
        )
        wscore, lscore = max(game.away_score, game.home_score), min(game.away_score, game.home_score)
        game.narrative = (
            f"{winner} defeated {loser} {wscore}–{lscore} at {game.venue}. "
            f"Detailed play-by-play data was not available at fetch time."
        )
        game.narrative_metadata = {"mode": "skipped", "reason": "no_key_plays"}
        return game

    if mode == "consensus":
        return _consensus_narrative(game, voter_models or [], verifier_model)
    return _single_narrative(game, model)


# =============================================================================
# MAIN ENTRY
# =============================================================================
ProgressCB = Callable[[int, int, str], None]


def build_handoff_package(games: list[Game], target_date: str) -> dict:
    """
    Package this run's output for a frontier model (e.g. Claude Opus) to do
    prediction. Includes the structured data plus an instruction preamble so
    the downstream model knows exactly what it's looking at and what the
    data quality caveats are.
    """
    caveats = []
    n_no_wpa = sum(1 for g in games
                   if g.key_plays and g.key_plays[0].rank_metric != "wpa")
    if n_no_wpa:
        caveats.append(
            f"{n_no_wpa}/{len(games)} games lack MLB win-probability data; "
            f"their key plays are ranked by {'captivatingIndex or scoring heuristic'} "
            "instead (see rank_metric per play)."
        )
    n_no_lineup = sum(1 for g in games if not g.home_lineup and not g.away_lineup)
    if n_no_lineup:
        caveats.append(
            f"{n_no_lineup}/{len(games)} games have no lineup data "
            "(processed before lineups posted)."
        )

    return {
        "generated": date.today().isoformat(),
        "target_date": target_date,
        "schema_notes": {
            "player_form.season_stats / recent_stats": "flat numeric dicts; recent window is 15d hitters / 30d pitchers",
            "player_form.splits": "platoon splits - vs_lhp/vs_rhp for hitters, vs_lhb/vs_rhb for pitchers",
            "player_form.statcast": "recent + season; pitch_mix per pitch type for pitchers",
            "key_plays.rank_metric": "wpa (best) | captivatingIndex (MLB's interest score) | scoring_heuristic (fallback)",
            "team_form": "last-10 record, runs per game for/against, streak - entering the game",
            "decisions": "winning_pitcher / losing_pitcher / save_pitcher per game",
            "pitchers lists": "home_pitchers/away_pitchers = every pitcher who appeared (boxscore); index 0 is the starter, rest relievers",
            "form_label thresholds": "hitters: OPS delta +-150/50 pts; pitchers: ERA delta +-1.0/0.3; gated at 25 PA / 5 IP",
        },
        "caveats": caveats,
        "games": [g.model_dump() for g in games],
    }


def update_players_db(games: list[Game], db_path: Path,
                       max_appearances: int = 30) -> int:
    """
    Merge player data from `games` into a persistent player database keyed by
    person_id. Updates stats with latest values; preserves a rolling list of
    recent game appearances per player so we can see who played when.

    Returns the total number of players tracked in the DB after the merge.
    """
    # Load existing DB (a list of player records, keyed by person_id when in memory)
    existing: dict[int, dict] = {}
    if db_path.exists():
        try:
            for rec in json.loads(db_path.read_text()):
                pid = rec.get("person_id")
                if pid is not None:
                    existing[int(pid)] = rec
        except (json.JSONDecodeError, TypeError):
            log.warning(f"players_db at {db_path} unreadable; starting fresh")

    today = date.today().isoformat()

    # Walk every player who appeared in this run's games
    for g in games:
        roster_with_side = [
            (g.home_starter, "home"),
            (g.away_starter, "away"),
        ]
        roster_with_side += [(p, "home") for p in g.home_lineup]
        roster_with_side += [(p, "away") for p in g.away_lineup]
        # Relievers and any post-game pitchers we discovered via boxscore
        roster_with_side += [(p, "home") for p in g.home_pitchers]
        roster_with_side += [(p, "away") for p in g.away_pitchers]

        for player, side in roster_with_side:
            if not player:
                continue
            pid = player.person_id
            record = player.model_dump()
            record["last_updated"] = today

            # Stitch in a rolling appearance history
            appearance = {
                "game_pk": g.game_pk,
                "date": g.date,
                "team_side": side,                       # "home" or "away"
                "team_name": g.home if side == "home" else g.away,
                "opponent": g.away if side == "home" else g.home,
                "venue": g.venue,
                "team_won": (
                    (side == "home" and g.home_score > g.away_score) or
                    (side == "away" and g.away_score > g.home_score)
                ),
                "team_score": g.home_score if side == "home" else g.away_score,
                "opp_score": g.away_score if side == "home" else g.home_score,
            }

            prior_apps = (existing.get(pid) or {}).get("appearances", [])
            # Dedup by game_pk so reprocessing the same date doesn't double-count
            kept = [a for a in prior_apps if a.get("game_pk") != g.game_pk]
            kept.append(appearance)
            # Keep most recent N by date
            kept.sort(key=lambda a: a.get("date", ""), reverse=True)
            record["appearances"] = kept[:max_appearances]

            existing[pid] = record

    # Write back as a sorted list (stable order: by name)
    out_list = sorted(existing.values(), key=lambda r: r.get("name", ""))
    db_path.write_text(json.dumps(out_list, indent=2, default=str))
    return len(out_list)


def run_for_date(
    d: str,
    final_only: bool = True,
    limit: int | None = None,
    teams: list[str] | None = None,
    with_statcast: bool = False,
    narrative_mode: NarrativeMode = "off",
    llm_model: str = DEFAULT_LLM_MODEL,
    consensus_models: Optional[list[str]] = None,
    verifier_model: Optional[str] = None,
    on_progress: ProgressCB | None = None,
) -> list[Game]:
    games_raw = get_schedule(d)

    if teams:
        team_filters = [t.lower() for t in teams]
        games_raw = [
            g for g in games_raw
            if any(
                tf in (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "").lower()
                or tf in (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "").lower()
                for tf in team_filters
            )
        ]
    if limit:
        games_raw = games_raw[:limit]

    log.info(f"{d}: {len(games_raw)} games to process")
    # Season follows the date being processed, not today's calendar year -
    # otherwise running a 2025 date would mix 2026 season stats into it.
    try:
        season_for_run = int(d[:4])
    except (ValueError, TypeError):
        season_for_run = SEASON
    out: list[Game] = []
    total = len(games_raw)

    for i, raw in enumerate(games_raw):
        state = raw.get("status", {}).get("abstractGameState", "")
        is_final = state == "Final"
        if final_only and not is_final:
            continue

        game_pk = raw.get("gamePk")
        if not game_pk:
            log.warning("schedule entry missing gamePk; skipping")
            continue

        feed = get_live_feed(game_pk, is_final=is_final)
        game = normalize_game(feed)
        if not game:
            continue

        game.key_plays = extract_key_plays(feed)
        score, reasons = score_importance(game)
        game.importance_score = score
        game.importance_reasons = reasons
        game = attach_forms(game, raw, season=season_for_run, with_statcast=with_statcast)

        # Team form entering the game (last 10 completed games per team)
        try:
            home_id = raw["teams"]["home"]["team"]["id"]
            away_id = raw["teams"]["away"]["team"]["id"]
            game.home_team_form = get_team_form(home_id, game.date) or None
            game.away_team_form = get_team_form(away_id, game.date) or None
        except (KeyError, TypeError):
            pass

        # Post-game enrichment: pull every pitcher who actually appeared
        # (includes relievers/closers missed by the pre-game schedule).
        game = attach_actual_pitchers(game, game_pk, is_final=is_final,
                                       season=season_for_run, with_statcast=with_statcast)
        game = add_narrative(game, mode=narrative_mode, model=llm_model,
                             voter_models=consensus_models,
                             verifier_model=verifier_model)

        out.append(game)
        log.info(f"  {game.away} @ {game.home}  "
                 f"{game.away_score}-{game.home_score}  imp={score:.2f}")
        if on_progress:
            on_progress(i + 1, total, f"{game.away} @ {game.home}")

    out.sort(key=lambda g: g.importance_score, reverse=True)
    return out


def _parse_args():
    p = argparse.ArgumentParser(description="MLB data pipeline")
    p.add_argument("date", nargs="?",
                   default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--teams", type=str, default=None)
    p.add_argument("--statcast", action="store_true",
                   help="Enrich with pybaseball/Statcast (xwOBA, barrel%, EV)")
    p.add_argument("--narrative", choices=["off", "single", "consensus"],
                   default="off")
    p.add_argument("--model", default=DEFAULT_LLM_MODEL,
                   help=f"Single-mode model (default: {DEFAULT_LLM_MODEL})")
    p.add_argument("--consensus-models", default="qwen2.5:7b,llama3.1:8b,gemma3:4b",
                   help="Comma-separated list of voter models (sequential)")
    p.add_argument("--verifier", default=None,
                   help="Optional verifier model (e.g. qwen2.5:14b). "
                        "If set, runs final audit + narrative pass.")
    p.add_argument("--include-non-final", action="store_true")
    p.add_argument("--team-season", default=None, metavar="TEAM",
                   help="Skip games; build a season-stat profile for every "
                        "player on TEAM's active roster -> team_season.json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.team_season:
        season_for_team = int(args.date[:4]) if args.date else SEASON
        profile = build_team_season(args.team_season, season_for_team,
                                    with_statcast=args.statcast)
        out = SCRIPT_DIR / "team_season.json"
        out.write_text(json.dumps(profile, indent=2, default=str))
        log.info(f"wrote {profile.get('player_count', 0)} players to {out}")
        raise SystemExit(0)
    team_list = [t.strip() for t in args.teams.split(",")] if args.teams else None
    voter_list = [m.strip() for m in args.consensus_models.split(",")]
    games = run_for_date(
        args.date,
        final_only=not args.include_non_final,
        limit=args.limit,
        teams=team_list,
        with_statcast=args.statcast,
        narrative_mode=args.narrative,
        llm_model=args.model,
        consensus_models=voter_list,
        verifier_model=args.verifier,
    )
    OUTPUT_PATH.write_text(json.dumps(
        [g.model_dump() for g in games], indent=2, default=str,
    ))
    log.info(f"wrote {len(games)} games to {OUTPUT_PATH}")

    handoff_path = SCRIPT_DIR / "handoff.json"
    handoff_path.write_text(json.dumps(
        build_handoff_package(games, args.date), indent=2, default=str,
    ))
    log.info(f"wrote prediction handoff package to {handoff_path}")

    total = update_players_db(games, PLAYERS_DB_PATH)
    log.info(f"players_db now tracks {total} players ({PLAYERS_DB_PATH})")