"""
Streamlit UI for the MLB pipeline.

Run with:
    pip install streamlit
    python -m streamlit run mlb_app.py

Opens at http://localhost:8501
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import streamlit as st

from mlb_pipeline import (
    build_team_season,
    DEFAULT_LLM_MODEL,
    Game,
    OUTPUT_PATH,
    PLAYERS_DB_PATH,
    PlayerForm,
    build_handoff_package,
    get_all_teams,
    run_for_date,
    update_players_db,
)

st.set_page_config(page_title="MLB Pipeline", layout="wide", page_icon="⚾")

# ----------------------------------------------------------------------------
FORM_BADGE = {
    "hot": ("🔥", "#ff4b4b"),
    "warming": ("↗️", "#ff9800"),
    "steady": ("➡️", "#9e9e9e"),
    "cooling": ("↘️", "#42a5f5"),
    "cold": ("🥶", "#1e88e5"),
    "insufficient_data": ("❓", "#bdbdbd"),
}


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def cached_team_list() -> list[str]:
    return get_all_teams()


def render_statcast(sc: dict, role: str) -> None:
    if not sc:
        return
    recent = sc.get("recent") or {}
    season = sc.get("season") or {}
    if recent.get("insufficient") or not recent:
        st.caption("Statcast: insufficient sample")
        return

    if role == "hitter":
        fields = [
            ("xwOBA", "xwoba", ".000"),
            ("Barrel%", "barrel_pct", "0.0%"),
            ("HardHit%", "hard_hit_pct", "0.0%"),
            ("Avg EV", "avg_ev", "0.0"),
        ]
    else:
        fields = [
            ("xwOBA vs", "xwoba_against", ".000"),
            ("Whiff%", "whiff_pct", "0.0%"),
            ("FB velo", "avg_fb_velo", "0.0"),
            ("HH%", "hard_hit_allowed_pct", "0.0%"),
        ]

    parts = []
    for label, key, _ in fields:
        r = recent.get(key)
        s = season.get(key) if season else None
        if r is None:
            continue
        if s is not None:
            arrow = "↑" if r > s else ("↓" if r < s else "→")
            parts.append(f"{label} {r}{arrow}{s}")
        else:
            parts.append(f"{label} {r}")
    if parts:
        st.caption("📊 " + " · ".join(parts))


def _splits_caption(splits: dict | None, role: str) -> str | None:
    """Compact one-line summary of vs-L / vs-R splits."""
    if not splits:
        return None
    if role == "hitter":
        l = splits.get("vs_lhp") or {}
        r = splits.get("vs_rhp") or {}
        l_ops = l.get("ops")
        r_ops = r.get("ops")
        if l_ops is None and r_ops is None:
            return None
        return f"🆚 vs LHP: {l_ops or '--'} OPS · vs RHP: {r_ops or '--'} OPS"
    else:
        l = splits.get("vs_lhb") or {}
        r = splits.get("vs_rhb") or {}
        l_ops = l.get("ops_against")
        r_ops = r.get("ops_against")
        if l_ops is None and r_ops is None:
            return None
        return f"🆚 vs LHB: {l_ops or '--'} OPS · vs RHB: {r_ops or '--'} OPS"


def _pitch_mix_caption(statcast: dict | None) -> str | None:
    """Top 4 pitches with usage % and avg velo."""
    if not statcast:
        return None
    recent = statcast.get("recent") or {}
    mix = recent.get("pitch_mix") or {}
    if not mix:
        return None
    parts = []
    for pt, info in list(mix.items())[:4]:
        velo = info.get("avg_velo")
        velo_str = f"@{velo}" if velo else ""
        parts.append(f"{pt} {info.get('usage_pct', 0)}%{velo_str}")
    return "🎯 " + " · ".join(parts)


def render_player(p: PlayerForm) -> None:
    emoji, color = FORM_BADGE[p.form_label]
    role_label = {"starting_pitcher": "SP", "relief_pitcher": "RP"}.get(p.role, "B")

    # Handedness suffix: hitter shows bat side, pitcher shows throw hand
    hand_bits = []
    if p.role == "hitter" and p.bat_side:
        hand_bits.append(f"bats {p.bat_side}")
    if p.role == "starting_pitcher" and p.throw_hand:
        hand_bits.append(f"throws {p.throw_hand}")
    hand_suffix = f" · {', '.join(hand_bits)}" if hand_bits else ""

    st.markdown(
        f"<div style='border-left: 3px solid {color}; padding-left: 8px; "
        f"margin-bottom: 8px;'>"
        f"<b>{emoji} {p.name}</b> "
        f"<span style='color:#888; font-size:0.85em'>"
        f"({role_label} · {p.form_label}{hand_suffix})</span><br>"
        f"<span style='font-size:0.85em'>Season: {p.season_line}<br>"
        f"Recent: {p.recent_line} · <i>{p.delta_summary}</i></span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if p.statcast:
        render_statcast(p.statcast, p.role)

    splits_cap = _splits_caption(p.splits, p.role)
    if splits_cap:
        st.caption(splits_cap)

    if p.role == "starting_pitcher":
        mix_cap = _pitch_mix_caption(p.statcast)
        if mix_cap:
            st.caption(mix_cap)


def render_narrative_metadata(meta: dict | None) -> None:
    if not meta or meta.get("mode") != "consensus":
        return
    with st.expander("🗳️ Consensus details (per-model + voting)", expanded=False):
        chosen = meta.get("chosen_model", "")
        st.caption(f"Narrative source: **{chosen}**")

        # Verifier audit (if a verifier was used)
        verified = meta.get("verified")
        if verified is not None:
            concerns = meta.get("concerns", []) or []
            if verified and not concerns:
                st.success("✅ Verifier: consensus consistent with raw play-by-play")
            else:
                st.warning("⚠️ Verifier flagged concerns:")
                for c in concerns:
                    st.markdown(f"- {c}")

        voted = meta.get("voted_fields", {})
        if voted:
            st.markdown("**Voted fields:**")
            for k, v in voted.items():
                st.markdown(f"- `{k}`: **{v}**")

        disagreements = meta.get("disagreements", [])
        if disagreements:
            st.markdown("**Disagreements (no 2+ majority):**")
            for d in disagreements:
                st.markdown(f"- {d}")
        else:
            st.markdown("_All categorical fields had majority agreement._")

        failed = meta.get("failed_models", [])
        if failed:
            st.warning(f"Failed voters: {', '.join(failed)}")

        per_model = meta.get("per_model", {})
        if per_model:
            st.markdown("**Each voter's analysis:**")
            for m, analysis in per_model.items():
                if analysis is None:
                    st.markdown(f"- **{m}**: _(failed)_")
                else:
                    st.markdown(f"- **{m}**: pace=`{analysis['pace']}` · "
                                f"decisive=`{analysis['decisive_inning']}` by "
                                f"`{analysis['decisive_player']}` "
                                f"({analysis['decisive_team']})")
                    st.caption(f"  _{analysis['narrative']}_")


def render_game(g: Game) -> None:
    header = (
        f"**{g.away} @ {g.home}** · {g.away_score}–{g.home_score} · "
        f"importance **{g.importance_score:.2f}** · {g.status}"
    )
    with st.expander(header, expanded=False):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### Game info")
            st.write(f"**Date:** {g.date}  ·  **Venue:** {g.venue}")
            if g.innings_played > 9:
                st.write(f"**Extra innings:** {g.innings_played}")
            decision_bits = []
            if g.winning_pitcher:
                decision_bits.append(f"W: {g.winning_pitcher}")
            if g.losing_pitcher:
                decision_bits.append(f"L: {g.losing_pitcher}")
            if g.save_pitcher:
                decision_bits.append(f"SV: {g.save_pitcher}")
            if decision_bits:
                st.write("**Decisions:** " + " · ".join(decision_bits))
            # Team form entering the game
            form_bits = []
            for label, tf in ((g.away, g.away_team_form), (g.home, g.home_team_form)):
                if tf:
                    form_bits.append(
                        f"{label}: {tf['wins']}-{tf['losses']} last {tf['last_n']} "
                        f"({tf['streak']}), {tf['runs_scored_pg']} RS/g, "
                        f"{tf['runs_allowed_pg']} RA/g"
                    )
            if form_bits:
                st.write("**Team form (entering):**")
                for fb in form_bits:
                    st.caption(fb)
            if g.importance_reasons:
                st.write("**Why it matters:** " + ", ".join(g.importance_reasons))
            if g.narrative:
                st.info(g.narrative)
                render_narrative_metadata(g.narrative_metadata)

        with col2:
            # Adapt header to the actual ranking metric used
            if g.key_plays:
                metric = g.key_plays[0].rank_metric
                header_metric = {
                    "wpa": "WPA",
                    "captivatingIndex": "MLB captivatingIndex",
                    "scoring_heuristic": "scoring plays (late/big first)",
                }.get(metric, metric)
                st.markdown(f"##### Key plays (ranked by {header_metric})")
            else:
                st.markdown("##### Key plays")
                st.caption("No play-by-play data available")

            for kp in g.key_plays:
                # Show whichever metrics are actually populated
                bits = [f"**{kp.inning}**"]
                if kp.wpa is not None:
                    bits.append(f"WPA {kp.wpa:+.2f}")
                if kp.leverage is not None:
                    bits.append(f"LI {kp.leverage:.1f}")
                if kp.wpa is None and kp.rank_metric != "wpa":
                    bits.append(f"{kp.rank_metric}={kp.rank_value}")
                line = " · ".join(bits)
                st.markdown(
                    f"{line}<br>"
                    f"<span style='font-size:0.9em'>{kp.description}</span>",
                    unsafe_allow_html=True,
                )

        if (g.home_starter or g.away_starter or g.home_lineup or g.away_lineup
                or g.home_pitchers or g.away_pitchers):
            st.divider()
            st.markdown("##### Player form")
            lcol, rcol = st.columns(2)

            def render_team(team_name, starter, lineup, pitchers):
                st.markdown(f"**{team_name}**")
                if starter:
                    render_player(starter)
                for p in lineup:
                    render_player(p)
                # Relievers = pitchers list minus the starter, if known
                starter_id = starter.person_id if starter else None
                relievers = [p for p in pitchers if p.person_id != starter_id]
                if relievers:
                    st.markdown("_Relievers (from boxscore):_")
                    for p in relievers:
                        render_player(p)

            with lcol:
                render_team(g.away, g.away_starter, g.away_lineup, g.away_pitchers)
            with rcol:
                render_team(g.home, g.home_starter, g.home_lineup, g.home_pitchers)


# ----------------------------------------------------------------------------
st.title("⚾ MLB Data Pipeline")
st.caption(f"Source: statsapi.mlb.com · Cache & output in `{OUTPUT_PATH.parent}`")

with st.sidebar:
    st.header("Controls")
    app_mode = st.radio("Mode", ["Game pipeline", "Team season roster"],
                        horizontal=True)
    target_date = st.date_input("Date",
                                 value=date.today() - timedelta(days=1),
                                 max_value=date.today())

    team_options = cached_team_list()
    selected_teams = st.multiselect("Filter by team (empty = all)", options=team_options)

    limit = st.slider("Max games", 1, 30, 5)
    final_only = st.toggle("Final games only", value=True)

    st.divider()
    st.subheader("Statcast enrichment")
    with_statcast = st.toggle(
        "Enable (pybaseball)", value=False,
        help="Adds xwOBA, barrel%, hard-hit%, EV per player. "
             "Slow on first run (~5–10s per player uncached); fast after cache warms.",
    )

    st.divider()
    st.subheader("Narrative")
    narrative_mode = st.radio(
        "Mode", options=["off", "single", "consensus"], index=0,
        captions=[
            "No LLM",
            "One model writes the recap",
            "3 models vote on facts; best-aligned model's recap wins",
        ],
    )
    if narrative_mode == "single":
        llm_model = st.text_input("Model", value=DEFAULT_LLM_MODEL)
        consensus_models = None
        verifier_model = None
    elif narrative_mode == "consensus":
        st.caption("Voters run **sequentially** (one model in VRAM at a time). "
                   "Pick mid-size models from different families.")
        m1 = st.text_input("Voter 1", value="qwen2.5:7b")
        m2 = st.text_input("Voter 2", value="llama3.1:8b")
        m3 = st.text_input("Voter 3", value="gemma3:4b")
        consensus_models = [m1.strip(), m2.strip(), m3.strip()]
        st.caption("Bigger verifiers = better narratives. On 8-12 GB VRAM: "
                   "`qwen2.5:14b` (fast), `gemma3:12b`, `phi4:14b`. "
                   "`qwen2.5:32b` works via CPU offload (~1-2 min/game) - "
                   "worth it if narrative quality is the priority.")
        verifier_input = st.text_input(
            "Verifier (optional)", value="qwen2.5:14b",
            help="Strong model that audits the consensus against raw "
                 "play-by-play and writes the final narrative. Leave empty "
                 "to use the best-aligned voter's narrative instead.",
        )
        verifier_model = verifier_input.strip() or None
        llm_model = DEFAULT_LLM_MODEL
    else:
        llm_model = DEFAULT_LLM_MODEL
        consensus_models = None
        verifier_model = None

    st.divider()
    run_btn = st.button("▶ Run pipeline", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
if not run_btn:
    st.info("Configure in the sidebar and click **Run pipeline**.")
    st.markdown(
        "**Tips**\n"
        "- Start with 1–3 games while warming the cache.\n"
        "- **Statcast** adds rich metrics but slows first runs substantially. "
        "Disk-cached after that.\n"
        "- **Consensus mode** now runs voters **sequentially** — Ollama swaps "
        "each model in/out of VRAM. This lets you use 7–14B voters instead "
        "of 3B ones, at the cost of ~10–15s per model swap.\n"
        "- The optional **verifier** (default `qwen2.5:14b`) audits the voted "
        "consensus against raw play-by-play and writes the final narrative. "
        "It catches errors voters miss — e.g. agreeing on the wrong winner.\n"
        "- The **consensus details** expander shows the verifier's audit "
        "(✅ verified / ⚠️ concerns) plus per-voter disagreements."
    )
    st.stop()

date_str = target_date.isoformat()

# ---- Team season roster mode ----
if app_mode == "Team season roster":
    team_pick = selected_teams[0] if selected_teams else None
    if not team_pick:
        st.warning("Pick exactly one team in the sidebar filter for roster mode.")
        st.stop()
    season_yr = int(date_str[:4])
    progress = st.progress(0.0, text=f"Building {team_pick} {season_yr} roster profile…")

    def on_roster_progress(done: int, total: int, label: str) -> None:
        progress.progress(done / max(total, 1), text=f"[{done}/{total}] {label}")

    try:
        profile = build_team_season(team_pick, season_yr,
                                    with_statcast=with_statcast,
                                    on_progress=on_roster_progress)
    except Exception as e:
        progress.empty()
        st.error(f"Roster build failed: {e}")
        st.stop()
    progress.empty()

    if not profile or not profile.get("players"):
        st.warning("No roster data found for that team/season.")
        st.stop()

    st.subheader(f"{profile['team']} — {profile['season']} season roster "
                 f"({profile['player_count']} players)")
    tf = profile.get("team_form")
    if tf:
        st.caption(f"Form: {tf['wins']}-{tf['losses']} last {tf['last_n']} "
                   f"({tf['streak']}), {tf['runs_scored_pg']} RS/g, "
                   f"{tf['runs_allowed_pg']} RA/g")

    # Split pitchers into SP/RP using the auto-detected role (not roster
    # position, which is just "P" for everyone), and sort each group by form.
    all_pitchers = [p for p in profile["players"] if p["roster_position"] == "P"]
    starters = [p for p in all_pitchers if p.get("role") == "starting_pitcher"]
    relievers = [p for p in all_pitchers if p.get("role") == "relief_pitcher"]
    hitters = [p for p in profile["players"] if p["roster_position"] != "P"]

    def _render(p):
        render_player(PlayerForm(**{k: v for k, v in p.items()
                                    if k != "roster_position"}))

    pcol, hcol = st.columns(2)
    with pcol:
        st.markdown(f"**Starting pitchers ({len(starters)})**")
        for p in starters:
            _render(p)
        st.markdown(f"**Relief pitchers ({len(relievers)})**")
        for p in relievers:
            _render(p)
    with hcol:
        st.markdown(f"**Position players ({len(hitters)})**")
        for p in hitters:
            _render(p)

    team_json = json.dumps(profile, indent=2, default=str)
    (OUTPUT_PATH.parent / "team_season.json").write_text(team_json)
    st.download_button(
        f"⬇ team_season.json ({profile['player_count']} players)",
        data=team_json,
        file_name=f"{profile['team'].replace(' ', '_')}_{profile['season']}.json",
        mime="application/json",
    )
    st.stop()

# ---- Game pipeline mode ----
progress = st.progress(0.0, text="Starting…")


def on_progress(done: int, total: int, label: str) -> None:
    progress.progress(done / max(total, 1), text=f"[{done}/{total}] {label}")


try:
    games = run_for_date(
        date_str,
        final_only=final_only,
        limit=limit,
        teams=selected_teams or None,
        with_statcast=with_statcast,
        narrative_mode=narrative_mode,
        llm_model=llm_model,
        consensus_models=consensus_models,
        verifier_model=verifier_model,
        on_progress=on_progress,
    )
except Exception as e:
    progress.empty()
    st.error(f"Pipeline failed: {e}")
    st.exception(e)
    st.stop()

progress.empty()

if not games:
    st.warning("No games matched your filters for that date.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Games processed", len(games))
m2.metric("Avg importance", f"{sum(g.importance_score for g in games) / len(games):.2f}")
m3.metric("Top importance", f"{games[0].importance_score:.2f}")

st.divider()
for g in games:
    render_game(g)

st.divider()
payload_json = json.dumps([g.model_dump() for g in games], indent=2, default=str)
OUTPUT_PATH.write_text(payload_json)

# Update the persistent player database (accumulates across runs)
db_total = update_players_db(games, PLAYERS_DB_PATH)
players_db_json = PLAYERS_DB_PATH.read_text()

st.markdown("### Outputs")
handoff = build_handoff_package(games, date_str)
handoff_json = json.dumps(handoff, indent=2, default=str)
(OUTPUT_PATH.parent / "handoff.json").write_text(handoff_json)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.download_button(
        "⬇ output.json (games)",
        data=payload_json,
        file_name=f"mlb_games_{date_str}.json",
        mime="application/json", use_container_width=True,
    )
    st.caption(f"This run's games · `{OUTPUT_PATH.name}`")
with col_b:
    st.download_button(
        f"⬇ players_db.json ({db_total})",
        data=players_db_json,
        file_name="players_db.json",
        mime="application/json", use_container_width=True,
    )
    st.caption(f"All players across runs · `{PLAYERS_DB_PATH.name}`")
with col_c:
    st.download_button(
        "⬇ handoff.json (for Opus)",
        data=handoff_json,
        file_name=f"handoff_{date_str}.json",
        mime="application/json", use_container_width=True,
    )
    st.caption("Games + schema notes + data-quality caveats — paste this "
               "into your prediction model")