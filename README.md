# mlb-pipeline

A local data pipeline that pulls MLB game and player data, structures it, and
optionally generates natural-language summaries via locally-hosted LLMs.


## What it does

Given a date, it fetches every MLB game and produces structured JSON containing:

- Game outcomes, key plays (ranked by win-probability when available, with
  graceful fallback to MLB's `captivatingIndex` or a scoring-play heuristic)
- Pitching decisions (W/L/SV) and inning-by-inning linescore
- For every player who appeared:
    - Season and rolling-window numeric stats
    - Platoon splits (vs LHP/RHP for hitters, vs LHB/RHB for pitchers)
    - Statcast metrics: xwOBA, barrel%, hard-hit%, exit velocity (via pybaseball)
    - Pitch mix breakdown for pitchers
    - Handedness, position, age
- Team form entering the game (last-10 record, run differential, streak)
- An optional LLM-generated narrative recap

It produces three outputs per run:
- `output.json` — the games processed in this run
- `players_db.json` — a persistent player database accumulated across runs
- `handoff.json` — the above wrapped with schema notes and data-quality caveats,
  formatted to be pasted into a downstream model

There's also a "team season roster" mode that builds a full-season stat profile
for every player on a team's active roster, independent of any specific game.

## Design notes

The pipeline is deliberately code-first. Numerical work, sorting, classification,
and validation all run in deterministic Python. LLMs are confined to writing
prose at the end, and only after structured data has been validated. This avoids
the failure mode where a small model invents stats from a final score alone.

For multi-model setups, voters run sequentially (one in VRAM at a time via
Ollama's `keep_alive=0`), then a stronger verifier model writes the final
narrative and audits the voted consensus against the raw play-by-play.

## Stack

- Python 3.11+
- `requests`, `pydantic`, `streamlit` (UI)
- `pybaseball` (optional, for Statcast enrichment)
- `ollama` Python client + a running Ollama server (optional, for narratives)
- Data source: MLB's public statsapi at `statsapi.mlb.com`

## Running

```bash
pip install streamlit requests pydantic pybaseball ollama
python -m streamlit run mlb_app.py
```

The UI opens at `http://localhost:8501`. Pick a date, set filters, run.

Or CLI:

```bash
python mlb_pipeline.py 2026-06-05 --limit 3 --statcast
python mlb_pipeline.py --team-season "Los Angeles Dodgers"
```

## Status

Personal project, not maintained for general use. The MLB API endpoints are
public and unofficial — they can change without notice.


## License

MIT — see [LICENSE](LICENSE).
