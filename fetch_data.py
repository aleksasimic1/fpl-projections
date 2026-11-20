#!/usr/bin/env python3
"""
FPL points projection model.

Pulls data from the official FPL API, projects points per fixture for every
player, and writes the result to data.json (read by index.html).

Run locally:   python fetch_data.py
Needs only the Python standard library (3.9+).
"""

import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# Model settings (tweak these)
# ----------------------------------------------------------------------------
W_SEASON = 0.8          # weight of season-long rates
W_FORM = 0.2            # weight of recent form (last FORM_GAMES appearances)
FORM_GAMES = 5

W_XSTAT = 0.65          # weight of xG / xA ...
W_ACTUAL = 0.35        # ... versus actual goals / assists

APPEARANCE_PTS = 2.0     # we assume everyone plays 90 minutes
FRINGE_MIN_AVG = 60      # avg minutes over the last FORM_GAMES appearances below this -> default points
FRINGE_DEFAULT_PTS = 2.0 # points per fixture for such players

OPP_EXPONENT = 0.6       # square root keeps the opponent effect mild
OPP_MIN, OPP_MAX = 0.5, 1.5
HOME_ATTACK_BOOST = 1.05 # home players: +5% on goal/assist/save rates
HOME_CS_FACTOR = 0.95    # home team concedes ~5% less (higher clean sheet chance)

GOAL_PTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_PTS = 3
CS_PTS = {"GK": 3, "DEF": 3, "MID": 0.65, "FWD": 0}
BONUS_PTS = 0.3             # history bonus is already recorded as actual FPL bonus points
DEFCON_PTS = 2
DEFCON_MULTIPLIER = 0.6
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}  # GKs do not get defcon points

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
HORIZON = 20              # number of gameweeks for the long projection

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-projection-model)"}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
def get_json(url, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
            print(f"Retrying {url} after error: {exc}")


def fetch_history(player_id):
    data = get_json(f"{BASE}/element-summary/{player_id}/")
    return player_id, data.get("history", [])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def clamp(x, lo=OPP_MIN, hi=OPP_MAX):
    return max(lo, min(hi, x))


def blend(season, form):
    if form is None:
        return season
    return W_SEASON * season + W_FORM * form


def expected_half_goals_conceded(lam, kmax=12):
    """E[floor(G/2)] for G ~ Poisson(lam): expected '-1 per 2 goals conceded' penalty."""
    total = 0.0
    p = math.exp(-lam)
    for k in range(kmax + 1):
        total += (k // 2) * p
        p *= lam / (k + 1)
    return total


# ----------------------------------------------------------------------------
# Team strength
# ----------------------------------------------------------------------------
def build_team_stats(teams, players, fixtures):
    games = {t["id"]: 0 for t in teams}
    for fx in fixtures:
        if fx.get("finished") or fx.get("finished_provisional"):
            games[fx["team_h"]] += 1
            games[fx["team_a"]] += 1

    xg_for = {t["id"]: 0.0 for t in teams}
    xgc = {t["id"]: 0.0 for t in teams}
    for p in players:
        xg_for[p["team"]] += f(p["expected_goals"])
        if p["element_type"] == 1:  # sum of goalkeepers' xGC ~ team xGC
            xgc[p["team"]] += f(p["expected_goals_conceded"])

    attack, defence = {}, {}
    for t in teams:
        g = games[t["id"]]
        attack[t["id"]] = xg_for[t["id"]] / g if g else None
        defence[t["id"]] = xgc[t["id"]] / g if g else None

    valid_att = [v for v in attack.values() if v]
    valid_def = [v for v in defence.values() if v]
    avg_att = sum(valid_att) / len(valid_att) if valid_att else 1.4
    avg_def = sum(valid_def) / len(valid_def) if valid_def else 1.4

    for tid in attack:
        if not attack[tid]:
            attack[tid] = avg_att
        if not defence[tid]:
            defence[tid] = avg_def
    return attack, defence, avg_att, avg_def


# ----------------------------------------------------------------------------
# Player profile (per-90 rates, blended season/form)
# ----------------------------------------------------------------------------
def build_profile(p, history):
    pos = POSITIONS[p["element_type"]]
    minutes = p["minutes"]

    # Season rates per 90 (xG/xA mixed with actual goals/assists)
    s90 = 90.0 / minutes
    season_g = (W_XSTAT * f(p["expected_goals"]) + W_ACTUAL * p["goals_scored"]) * s90
    season_a = (W_XSTAT * f(p["expected_assists"]) + W_ACTUAL * p["assists"]) * s90

    # Use actual appearances only for season/form appearance-based statistics.
    played = [h for h in history if h["minutes"] > 0]

    thr = DEFCON_THRESHOLD.get(pos)
    if thr and played:
        season_dc = sum(
            1 for h in played
            if h.get("defensive_contribution", 0) >= thr
        ) / len(played)
    else:
        season_dc = 0.0

    # Bonus points per appearance, using season average and last five appearances.
    season_bonus = (
        sum(f(h.get("bonus", 0)) for h in played) / len(played)
        if played else 0.0
    )

    recent = played[-FORM_GAMES:]
    recent_min = sum(h["minutes"] for h in recent)
    avg_recent_min = recent_min / len(recent) if recent else 0.0

    form_g = form_a = form_saves = form_dc = form_bonus = None
    if recent_min > 0:
        s = 90.0 / recent_min
        form_g = (W_XSTAT * sum(f(h["expected_goals"]) for h in recent)
                  + W_ACTUAL * sum(h["goals_scored"] for h in recent)) * s
        form_a = (W_XSTAT * sum(f(h["expected_assists"]) for h in recent)
                  + W_ACTUAL * sum(h["assists"] for h in recent)) * s
        form_saves = sum(h.get("saves", 0) for h in recent) * s

        recent_played = [h for h in recent if h["minutes"] > 0]
        if thr and recent_played:
            form_dc = sum(
                1 for h in recent_played
                if h.get("defensive_contribution", 0) >= thr
            ) / len(recent_played)

        if recent_played:
            form_bonus = (
                sum(f(h.get("bonus", 0)) for h in recent_played)
                / len(recent_played)
            )

    return {
        "pos": pos,
        "fringe": avg_recent_min < FRINGE_MIN_AVG,
        "g90": blend(season_g, form_g),
        "a90": blend(season_a, form_a),
        "saves90": blend(season_saves := p["saves"] * s90, form_saves),
        "bonus": blend(season_bonus, form_bonus),
        "dc": blend(season_dc, form_dc if thr else None),
    }


# ----------------------------------------------------------------------------
# Single-fixture projection
# ----------------------------------------------------------------------------
def fixture_components(prof, team_id, opp_id, home, attack, defence, avg_att, avg_def):
    """Return (points_by_component, diagnostics) for one fixture."""
    if prof["fringe"]:
        return {"fringe_default": FRINGE_DEFAULT_PTS}, {}

    pos = prof["pos"]
    # Opponent defensive weakness (more xGC per game -> easier to score against)
    att_mult = clamp((defence[opp_id] / avg_def) ** OPP_EXPONENT)
    # Opponent attacking strength (more xG per game -> harder to keep a clean sheet)
    def_mult = clamp((attack[opp_id] / avg_att) ** OPP_EXPONENT)
    boost = HOME_ATTACK_BOOST if home else 1.0

    goals = prof["g90"] * att_mult * boost * GOAL_PTS[pos]
    assists = prof["a90"] * att_mult * boost * ASSIST_PTS

    lam = defence[team_id] * def_mult * (HOME_CS_FACTOR if home else 1.0)
    # All players are assumed to play 90 minutes, so the FPL 60-minute
    # clean-sheet eligibility condition is automatically satisfied.
    cs = math.exp(-lam) * CS_PTS[pos]
    gc_penalty = -expected_half_goals_conceded(lam) if pos in ("GK", "DEF") else 0.0

    defcon = prof["dc"] * DEFCON_PTS * DEFCON_MULTIPLIER if pos != "GK" else 0.0
    saves = (prof["saves90"] * att_mult * boost / 3.0) if pos == "GK" else 0.0
    bonus = prof["bonus"]

    points = {
        "appearance": APPEARANCE_PTS,
        "goals": goals,
        "assists": assists,
        "clean_sheet": cs,
        "goals_conceded": gc_penalty,
        "defcon": defcon,
        "saves": saves,
        "bonus": bonus,
    }
    info = {
        "att_mult": att_mult,
        "def_mult": def_mult,
        "boost": boost,
        "lambda_conceded": lam,
        "p_clean_sheet": math.exp(-lam),
    }
    return points, info


def project_fixture(prof, team_id, opp_id, home, attack, defence, avg_att, avg_def):
    points, _ = fixture_components(prof, team_id, opp_id, home, attack, defence, avg_att, avg_def)
    return sum(points.values())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def availability(p):
    """Return chance of playing next round in %, treating missing value sensibly."""
    chance = p.get("chance_of_playing_next_round")
    if chance is not None:
        return int(chance)
    return 0 if p.get("status") in ("i", "s", "u", "n") else 100


def main():
    print("Fetching bootstrap-static and fixtures...")
    bootstrap = get_json(f"{BASE}/bootstrap-static/")
    fixtures = get_json(f"{BASE}/fixtures/")

    teams = bootstrap["teams"]
    events = bootstrap["events"]
    team_by_id = {t["id"]: t for t in teams}

    # Ignore players with 0 minutes in the whole season
    players = [p for p in bootstrap["elements"] if p["minutes"] > 0]
    print(f"{len(players)} players with minutes this season")

    next_event = next((e["id"] for e in events if e.get("is_next")), None)
    if next_event is None:
        next_event = next((e["id"] for e in events if not e.get("finished")), None)
    if next_event is None:
        raise SystemExit("No upcoming gameweek found (season over?).")
    horizon_events = list(range(next_event, min(next_event + HORIZON, 38) + 1))[:HORIZON]

    attack, defence, avg_att, avg_def = build_team_stats(teams, bootstrap["elements"], fixtures)

    # Upcoming fixtures per team: {team_id: {event: [(opp_id, home), ...]}}
    upcoming = {t["id"]: {} for t in teams}
    for fx in fixtures:
        ev = fx.get("event")
        if ev is None or ev not in horizon_events or fx.get("finished"):
            continue
        upcoming[fx["team_h"]].setdefault(ev, []).append((fx["team_a"], True))
        upcoming[fx["team_a"]].setdefault(ev, []).append((fx["team_h"], False))

    # Per-player history (needed for form, bonus and defcon rates)
    print("Fetching player histories...")
    histories = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (pid, hist) in enumerate(pool.map(fetch_history, [p["id"] for p in players]), 1):
            histories[pid] = hist
            if i % 100 == 0:
                print(f"  {i}/{len(players)}")

    rows = []
    for p in players:
        prof = build_profile(p, histories[p["id"]])
        tid = p["team"]
        team_fx = upcoming[tid]

        def gw_points(ev):
            return sum(project_fixture(prof, tid, opp, home, attack, defence, avg_att, avg_def)
                       for opp, home in team_fx.get(ev, []))

        next1 = gw_points(next_event)
        next5 = sum(gw_points(ev) for ev in horizon_events)
        opp_label = ", ".join(
            f"{team_by_id[opp]['short_name']} ({'H' if home else 'A'})"
            for opp, home in team_fx.get(next_event, [])
        ) or "-"

        rows.append({
            "name": p["web_name"],
            "team": team_by_id[tid]["short_name"],
            "pos": prof["pos"],
            "opp": opp_label,
            "next1": round(next1, 1),
            "next5": round(next5, 1),
            "chance": availability(p),
            "news": p.get("news") or "",
            "fringe": prof["fringe"],
        })

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "next_gw": next_event,
        "horizon": horizon_events,
        "teams": sorted(t["short_name"] for t in teams),
        "players": rows,
    }
    with open("data.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote data.json with {len(rows)} players (next GW: {next_event})")


if __name__ == "__main__":
    main()
