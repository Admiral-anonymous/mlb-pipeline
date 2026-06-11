"""
Statcast enrichment via pybaseball.

Adds advanced metrics to a PlayerForm in a non-destructive way - the existing
OPS/ERA-based form_label stays, but `statcast` dict gets attached with:
  - Hitters: xwOBA, barrel%, hard-hit%, avg/max exit velocity, sample size
  - Pitchers: xwOBA against, whiff%, avg fastball velo, hard-hit% allowed

Cache: per-player per-window JSON in <cache_dir>/statcast/. Cheap to call
repeatedly once warm; first run is slow because pybaseball scrapes
Baseball Savant page by page.

Why this matters: xwOBA is dramatically better than OPS for measuring form
because it controls for batted-ball luck (a hitter can have a .200 OPS over
two weeks with an .350 xwOBA - they're hitting the ball great, just unlucky).
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger("mlb.statcast")

_PYBASEBALL_OK: Optional[bool] = None


def pybaseball_available() -> bool:
    global _PYBASEBALL_OK
    if _PYBASEBALL_OK is None:
        try:
            import pybaseball  # noqa: F401
            _PYBASEBALL_OK = True
        except ImportError:
            _PYBASEBALL_OK = False
    return _PYBASEBALL_OK


def _silent_call(fn, *args, **kwargs):
    """Run pybaseball functions while swallowing their progress chatter."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*args, **kwargs)


def _cache_path(cache_dir: Path, kind: str, person_id: int, start: str, end: str) -> Path:
    out = cache_dir / "statcast"
    out.mkdir(exist_ok=True)
    return out / f"{kind}_{person_id}_{start}_{end}.json"


def fetch_hitter_statcast(person_id: int, start: str, end: str,
                          cache_dir: Path) -> Optional[dict]:
    """Return computed Statcast metrics for a hitter over a date range."""
    if not pybaseball_available():
        return None

    cpath = _cache_path(cache_dir, "hit", person_id, start, end)
    if cpath.exists():
        try:
            return json.loads(cpath.read_text())
        except json.JSONDecodeError:
            pass

    try:
        from pybaseball import statcast_batter
        df = _silent_call(statcast_batter, start, end, person_id)
    except Exception as e:
        log.warning(f"statcast_batter failed for {person_id}: {e}")
        return None

    if df is None or len(df) == 0:
        return None

    # Batted ball events per Savant's definition: type=='X' (in play), with
    # launch_speed populated. Previously we filtered only on launch_speed,
    # which sometimes catches non-batted-ball events with stray measurements,
    # systematically deflating HardHit% and Avg EV.
    in_play = df[df.get("type") == "X"] if "type" in df.columns else df
    batted = in_play[in_play["launch_speed"].notna()]
    sample = int(len(batted))
    if sample < 5:
        result = {"sample_size": sample, "insufficient": True}
    else:
        xwoba = df["estimated_woba_using_speedangle"].dropna()
        barrel_pct = None
        if "barrel" in batted.columns:
            barrel_series = batted["barrel"].dropna()
            if len(barrel_series) > 0:
                barrel_pct = round(float((barrel_series == 1).mean() * 100), 1)
        result = {
            "xwoba": round(float(xwoba.mean()), 3) if len(xwoba) > 0 else None,
            "barrel_pct": barrel_pct,
            "hard_hit_pct": round(float((batted["launch_speed"] >= 95).mean() * 100), 1),
            "avg_ev": round(float(batted["launch_speed"].mean()), 1),
            "max_ev": round(float(batted["launch_speed"].max()), 1),
            "batted_balls": sample,
            "sample_size": sample,
        }

    cpath.write_text(json.dumps(result))
    return result


def _compute_pitch_mix(df) -> dict:
    """
    Aggregate Statcast per-pitch data by pitch_type. Returns a dict keyed by
    pitch type (FF, SL, CU, CH, etc.) sorted by usage descending. Each entry:
        usage_pct, avg_velo, xwoba_against, whiff_pct, pitches
    Pitch types with fewer than 5 pitches are dropped (too noisy).
    """
    if df is None or len(df) == 0:
        return {}
    total = len(df)
    types = [t for t in df["pitch_type"].dropna().unique() if t]
    swing_types = {"swinging_strike", "foul", "hit_into_play",
                   "foul_tip", "swinging_strike_blocked"}

    mix = {}
    for pt in types:
        sub = df[df["pitch_type"] == pt]
        n = len(sub)
        if n < 5:
            continue
        swings = sub[sub["description"].isin(swing_types)]
        whiffs = sub[sub["description"] == "swinging_strike"]
        xwoba = sub["estimated_woba_using_speedangle"].dropna()
        velo = sub["release_speed"].dropna()
        mix[pt] = {
            "usage_pct": round(n / total * 100, 1),
            "avg_velo": round(float(velo.mean()), 1) if len(velo) > 0 else None,
            "xwoba_against": round(float(xwoba.mean()), 3) if len(xwoba) > 0 else None,
            "whiff_pct": round(float(len(whiffs) / max(len(swings), 1)) * 100, 1),
            "pitches": n,
        }
    return dict(sorted(mix.items(), key=lambda x: x[1]["usage_pct"], reverse=True))


def fetch_pitcher_statcast(person_id: int, start: str, end: str,
                           cache_dir: Path) -> Optional[dict]:
    """Return computed Statcast metrics for a pitcher over a date range."""
    if not pybaseball_available():
        return None

    cpath = _cache_path(cache_dir, "pit", person_id, start, end)
    if cpath.exists():
        try:
            return json.loads(cpath.read_text())
        except json.JSONDecodeError:
            pass

    try:
        from pybaseball import statcast_pitcher
        df = _silent_call(statcast_pitcher, start, end, person_id)
    except Exception as e:
        log.warning(f"statcast_pitcher failed for {person_id}: {e}")
        return None

    if df is None or len(df) == 0:
        return None

    pitches = int(len(df))
    if pitches < 20:
        result = {"sample_size": pitches, "insufficient": True}
    else:
        swing_types = {"swinging_strike", "foul", "hit_into_play",
                       "foul_tip", "swinging_strike_blocked"}
        swings = df[df["description"].isin(swing_types)]
        whiffs = df[df["description"] == "swinging_strike"]
        # Match Savant's "batted ball event" definition (in play, type='X')
        in_play = df[df.get("type") == "X"] if "type" in df.columns else df
        batted = in_play[in_play["launch_speed"].notna()]
        fb_types = {"FF", "SI", "FC"}
        fb = df[df["pitch_type"].isin(fb_types)]
        xwoba = df["estimated_woba_using_speedangle"].dropna()

        result = {
            "whiff_pct": round(float(len(whiffs) / max(len(swings), 1)) * 100, 1),
            "avg_fb_velo": round(float(fb["release_speed"].mean()), 1) if len(fb) > 0 else None,
            "hard_hit_allowed_pct": (round(float((batted["launch_speed"] >= 95).mean() * 100), 1)
                                     if len(batted) >= 5 else None),
            "xwoba_against": round(float(xwoba.mean()), 3) if len(xwoba) > 0 else None,
            "pitches": pitches,
            "sample_size": pitches,
            "pitch_mix": _compute_pitch_mix(df),
        }

    cpath.write_text(json.dumps(result))
    return result


def enrich_form(form, recent_start: str, recent_end: str, cache_dir: Path,
                season_start: Optional[str] = None):
    """Attach a `statcast` dict to a PlayerForm with recent + season-to-date stats."""
    if not pybaseball_available():
        return form

    season_start = season_start or f"{date.fromisoformat(recent_end).year}-03-15"

    if form.role == "hitter":
        recent = fetch_hitter_statcast(form.person_id, recent_start, recent_end, cache_dir)
        season = fetch_hitter_statcast(form.person_id, season_start, recent_end, cache_dir)
    else:
        recent = fetch_pitcher_statcast(form.person_id, recent_start, recent_end, cache_dir)
        season = fetch_pitcher_statcast(form.person_id, season_start, recent_end, cache_dir)

    form.statcast = {"recent": recent, "season": season}
    return form