"""Week-by-week points and how the standings moved.

Everything here is recomputed from the immutable game results on every build
rather than accumulated into snapshot files. That means a fresh clone
reproduces the entire history exactly, there is no state to drift, and a
scoring fix retroactively corrects every past week instead of leaving old
snapshots wrong.

nflverse numbers the postseason as a continuation of the regular season —
weeks 1-18 are ``REG``, then 19 ``WC``, 20 ``DIV``, 21 ``CON``, 22 ``SB`` — so
one week axis covers the whole season.
"""

from __future__ import annotations

import pandas as pd

from .scoring import entrant_scores, score_teams
from .season import Season
from .standings import final_seeds


def played_weeks(games: pd.DataFrame) -> list[int]:
    """Every week that has at least one completed game, in order."""
    played = games[games["played"]]
    return sorted(int(w) for w in played["week"].dropna().unique())


def weekly_frame(season: Season, games: pd.DataFrame) -> pd.DataFrame:
    """Cumulative entrant totals after each week.

    Returns a frame indexed by entrant name with one column per played week,
    plus ``rank_<week>`` columns. Berth bonuses only appear once the regular
    season is complete, which is the week they are actually earned.
    """
    weeks = played_weeks(games)
    names = [e.name for e in season.entrants]
    totals = pd.DataFrame(index=names, dtype=float)
    ranks = pd.DataFrame(index=names, dtype=float)

    for week in weeks:
        upto = games.copy()
        upto.loc[upto["week"] > week, "played"] = False
        seeds = final_seeds(season, upto)
        scores = entrant_scores(season, score_teams(season, upto, seeds))
        totals[week] = scores.set_index("name")["total"].reindex(names)
        ranks[week] = scores.set_index("name")["rank"].reindex(names)

    totals.columns = [f"w{w}" for w in weeks]
    ranks.columns = [f"rank_w{w}" for w in weeks]
    out = totals.join(ranks)
    out.index.name = "name"
    out.attrs["weeks"] = weeks
    return out


def weekly_deltas(history: pd.DataFrame) -> pd.DataFrame:
    """Points scored *in* each week, rather than cumulative totals."""
    weeks = history.attrs.get("weeks", [])
    cols = [f"w{w}" for w in weeks]
    if not cols:
        return pd.DataFrame(index=history.index)
    deltas = history[cols].diff(axis=1)
    deltas[cols[0]] = history[cols[0]]
    deltas.attrs["weeks"] = weeks
    return deltas.round(2)


def series_for(history: pd.DataFrame, name: str) -> list[float]:
    """One entrant's cumulative totals, for a sparkline."""
    weeks = history.attrs.get("weeks", [])
    if not weeks or name not in history.index:
        return []
    return [float(history.loc[name, f"w{w}"]) for w in weeks]


def rank_series_for(history: pd.DataFrame, name: str) -> list[int]:
    weeks = history.attrs.get("weeks", [])
    if not weeks or name not in history.index:
        return []
    return [int(history.loc[name, f"rank_w{w}"]) for w in weeks]


def movers(history: pd.DataFrame, top: int = 3) -> dict[str, list[tuple[str, float]]]:
    """Biggest risers and fallers by rank over the most recent week."""
    weeks = history.attrs.get("weeks", [])
    if len(weeks) < 2:
        return {"risers": [], "fallers": []}

    prev, last = f"rank_w{weeks[-2]}", f"rank_w{weeks[-1]}"
    # A lower rank number is better, so improvement is previous minus current.
    change = (history[prev] - history[last]).sort_values(ascending=False)

    risers = [(n, float(v)) for n, v in change.items() if v > 0][:top]
    # Reversed so the biggest fall leads, mirroring the risers — the previous
    # tail-slice kept the right names but presented them smallest drop first.
    fallers = [(n, float(v)) for n, v in change.items() if v < 0][-top:][::-1]
    return {"risers": risers, "fallers": fallers}


def week_leaders(season: Season, history: pd.DataFrame, top: int = 3) -> list[tuple[str, float]]:
    """Who scored the most points in the most recent completed week."""
    deltas = weekly_deltas(history)
    weeks = deltas.attrs.get("weeks", [])
    if not weeks:
        return []
    last = deltas[f"w{weeks[-1]}"].sort_values(ascending=False)
    # A thin week must not crown a 0.00 "leader" — an empty panel is honest.
    last = last[last > 0]
    return [(n, float(v)) for n, v in last.head(top).items()]
