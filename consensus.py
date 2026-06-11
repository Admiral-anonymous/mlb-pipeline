"""
Multi-model consensus narrative - sequential mode with optional verifier.

Architecture:
  1. Voter models run one at a time (Ollama swaps each in/out of VRAM via
     keep_alive=0). This means VRAM constrains only the *largest* model in
     the chain, not the sum of them - so we can use 7B+ models instead of 3B.

  2. Code votes on each model's categorical fields (pace, decisive_inning,
     decisive_player, decisive_team). Majority wins.

  3. If a verifier_model is set, a final call hands the verifier:
        - the raw game payload (especially key_plays)
        - the voted consensus
        - each voter's own analysis
     The verifier writes the final narrative AND flags any place the
     voted consensus contradicts the raw data. This catches cases like
     two small models agreeing "Yankees won" when key_plays clearly show
     the Red Sox scored more runs.

  4. If no verifier is configured, falls back to the previous behavior:
     narrative comes from whichever voter aligned most with consensus.

Why this beats parallel small models:
  - Small models (3B) routinely fail JSON-schema constraints.
  - With sequential loading, we can afford 7-14B voters.
  - A 14B verifier as the final auditor catches errors the voters miss.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger("mlb.consensus")

# Mid-size voters + strong verifier. Configure via UI as needed.
DEFAULT_VOTER_MODELS = ["qwen2.5:7b", "llama3.1:8b", "gemma3:4b"]
DEFAULT_VERIFIER_MODEL = "qwen2.5:14b"


Pace = Literal["one_sided", "competitive", "close", "comeback", "back_and_forth", "unknown"]
Team = Literal["home", "away", "unknown"]


class GameAnalysis(BaseModel):
    pace: Pace
    decisive_inning: str = Field(description='e.g. "T7", "B9", or "none"')
    decisive_player: str = Field(description='Player name, or "unknown"')
    decisive_team: Team
    narrative: str = Field(description="2-3 sentence factual recap")


class VerifierOutput(BaseModel):
    narrative: str
    verified: bool
    concerns: list[str] = Field(default_factory=list)


class ConsensusResult(BaseModel):
    narrative: str
    chosen_model: str                       # source of the final narrative
    voted_fields: dict                      # winners per categorical field
    per_model: dict[str, Optional[dict]]    # raw output per voter
    disagreements: list[str]                # fields without 2+ majority
    failed_models: list[str]                # voters that didn't return valid JSON

    # Verifier-only fields (None if no verifier was used)
    verified: Optional[bool] = None
    concerns: list[str] = Field(default_factory=list)
    verifier_model: Optional[str] = None


# ---------------------------------------------------------------------------
def _ask_voter(payload: dict, model: str) -> Optional[GameAnalysis]:
    """Run one voter. keep_alive=0 unloads the model after responding,
    freeing VRAM for the next call."""
    import ollama
    prompt = (
        "Analyze this baseball game and return structured JSON. Use the data "
        "below ONLY - no outside knowledge, no guesses. Use 'unknown' or 'none' "
        "when uncertain. Check runs_by_inning for big innings (5+ runs) - a "
        "big inning usually IS the story of the game and should appear in "
        "your narrative.\n\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Schema (return ONLY JSON matching this exactly):\n"
        '{\n'
        '  "pace": one of ["one_sided","competitive","close","comeback","back_and_forth","unknown"],\n'
        '  "decisive_inning": "T7" / "B9" / "none" (T=top, B=bottom),\n'
        '  "decisive_player": player full name or "unknown",\n'
        '  "decisive_team": "home" or "away" or "unknown",\n'
        '  "narrative": "2-3 factual sentences"\n'
        '}'
    )
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.2},
            keep_alive=0,  # unload immediately so the next voter has VRAM
        )
        return GameAnalysis.model_validate_json(resp["message"]["content"])
    except (ValidationError, json.JSONDecodeError, KeyError) as e:
        log.warning(f"voter {model} returned invalid JSON: {e}")
        return None
    except Exception as e:
        log.warning(f"voter {model} call failed: {e}")
        return None


def _vote(values: list, min_majority: int = 2) -> tuple[Any, bool]:
    valid = [v for v in values if v not in (None, "unknown", "none")]
    if not valid:
        for v in values:
            if v is not None:
                return v, False
        return None, False
    counts = Counter(valid)
    winner, count = counts.most_common(1)[0]
    return winner, count >= min_majority


def _run_verifier(payload: dict, voted_fields: dict,
                  voters: list[str], analyses: list[Optional[GameAnalysis]],
                  verifier_model: str) -> Optional[VerifierOutput]:
    """Verifier sees everything: raw data, voted consensus, and each
    voter's analysis. It writes the final narrative AND audits the
    consensus against the raw key_plays."""
    import ollama

    per_voter = {}
    for m, a in zip(voters, analyses):
        per_voter[m] = a.model_dump() if a else None

    prompt = (
        "You are auditing a multi-model consensus on a baseball game.\n\n"
        "Three smaller models analyzed this game and we voted on their "
        "categorical answers. Your job is to (a) write the final 2-4 sentence "
        "factual recap grounded in the RAW data - key_plays AND runs_by_inning "
        "(a 5+ run inning is the headline; lead with it; name W/L pitchers) - "
        "and (b) check whether the voted consensus is consistent with the raw "
        "data.\n\n"
        "If a voted field is 'unknown' but the raw key_plays clearly imply "
        "the answer, prefer the raw data. If a voted field contradicts the "
        "raw data, list it in 'concerns' and write the narrative based on "
        "the data, not the consensus.\n\n"
        "VOTED CONSENSUS:\n"
        f"{json.dumps(voted_fields, indent=2)}\n\n"
        "EACH VOTER'S ANALYSIS:\n"
        f"{json.dumps(per_voter, indent=2)}\n\n"
        "RAW GAME DATA:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Return ONLY JSON:\n"
        '{\n'
        '  "narrative": "2-3 factual sentences",\n'
        '  "verified": true if voted consensus aligns with raw data else false,\n'
        '  "concerns": ["list of short strings naming any contradictions, empty if verified"]\n'
        '}'
    )
    try:
        resp = ollama.chat(
            model=verifier_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.1},
            keep_alive=0,
        )
        return VerifierOutput.model_validate_json(resp["message"]["content"])
    except (ValidationError, json.JSONDecodeError, KeyError) as e:
        log.warning(f"verifier {verifier_model} returned invalid JSON: {e}")
        return None
    except Exception as e:
        log.warning(f"verifier {verifier_model} call failed: {e}")
        return None


# ---------------------------------------------------------------------------
def consensus_narrative(
    payload: dict,
    voter_models: Optional[list[str]] = None,
    verifier_model: Optional[str] = None,
) -> Optional[ConsensusResult]:
    """
    Run sequential consensus. If verifier_model is provided, the verifier
    writes the final narrative and audits the consensus. Otherwise, falls
    back to picking the best-aligned voter's own narrative.
    """
    voter_models = voter_models or DEFAULT_VOTER_MODELS

    # Stage 1: voters run sequentially (Ollama swaps them via keep_alive=0)
    analyses: list[Optional[GameAnalysis]] = []
    for m in voter_models:
        log.info(f"voter: {m}")
        analyses.append(_ask_voter(payload, m))

    failed = [m for m, a in zip(voter_models, analyses) if a is None]
    valid_pairs = [(m, a) for m, a in zip(voter_models, analyses) if a is not None]
    if not valid_pairs:
        log.error("all voter models failed")
        return None

    # Stage 2: vote on categorical fields (deterministic code)
    cat_fields = ["pace", "decisive_inning", "decisive_player", "decisive_team"]
    voted = {}
    disagreements = []
    for field in cat_fields:
        values = [getattr(a, field) for _, a in valid_pairs]
        winner, had_majority = _vote(values)
        voted[field] = winner
        if not had_majority and len(set(values)) > 1:
            disagreements.append(f"{field}: {sorted(set(values))}")

    per_model = {m: (a.model_dump() if a else None)
                 for m, a in zip(voter_models, analyses)}

    # Stage 3: verifier (optional)
    if verifier_model:
        log.info(f"verifier: {verifier_model}")
        v = _run_verifier(payload, voted, voter_models, analyses, verifier_model)
        if v is not None:
            return ConsensusResult(
                narrative=v.narrative,
                chosen_model=f"{verifier_model} (verifier)",
                voted_fields=voted,
                per_model=per_model,
                disagreements=disagreements,
                failed_models=failed,
                verified=v.verified,
                concerns=v.concerns,
                verifier_model=verifier_model,
            )
        log.warning("verifier failed; falling back to best-aligned voter")

    # Fallback: best-aligned voter's narrative
    def agreement_score(a: GameAnalysis) -> int:
        return sum(1 for f in cat_fields if getattr(a, f) == voted[f])

    best_model, best_analysis = max(
        valid_pairs, key=lambda pair: agreement_score(pair[1]),
    )
    return ConsensusResult(
        narrative=best_analysis.narrative,
        chosen_model=best_model,
        voted_fields=voted,
        per_model=per_model,
        disagreements=disagreements,
        failed_models=failed,
    )