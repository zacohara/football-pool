"""The scoring engine — pure functions from games to points.

This is the part that must never be wrong, so it is deliberately boring: no I/O,
no network, no global state. Everything takes a :class:`~football_pool.season.Season`
and a games DataFrame and returns numbers.

Scoring, in full:

* regular-season win  -> the team's leveling factor (a tie pays half)
* division winner     -> +3.0        } exactly one of these two, decided by the
* wild-card berth     -> +1.5        } final seeding
* wild-card upset     -> +1.0 flat, no LF, and only to a wild card that beats a
                         division winner
* divisional win      -> +1.0 + LF
* conference win      -> +1.5 + LF
* Super Bowl win      -> +2.0 + LF
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .season import Season

# Point values carry at most two decimals, so summing floats can drift a hair
# (0.1 + 0.2 territory). Every total is rounded at the boundary so that two
# entrants who genuinely tie compare equal instead of differing by 1e-16.
POINT_DP = 2


def _round(x):
    return np.round(x, POINT_DP)


def score_regular_season(season: Season, games: pd.DataFrame) -> np.ndarray:
    """(32,) points from completed regular-season games.

    A win pays ``win_multiplier * LF``; a tie pays ``tie_multiplier * LF`` to
    both teams.
    """
    pts = np.zeros(season.n_teams)
    reg = games[games["played"] & (games["game_type"] == "REG")]

    for row in reg.itertuples():
        hi, ai = season.idx[row.home_team], season.idx[row.away_team]
        if row.is_tie:
            pts[hi] += season.tie_multiplier * season.lf[hi]
            pts[ai] += season.tie_multiplier * season.lf[ai]
        elif row.home_won:
            pts[hi] += season.win_multiplier * season.lf[hi]
        else:
            pts[ai] += season.win_multiplier * season.lf[ai]
    return pts


def score_playoffs(season: Season, games: pd.DataFrame) -> np.ndarray:
    """(32,) points from completed playoff games.

    The wild-card rule needs no seeding lookup. Seed 1 has a bye, so wild-card
    games are 2v7, 3v6 and 4v5 — the home team is always a division winner and
    the away team always a wild card. (Verified against all 30 wild-card games
    from 2021-2025.) So the flat upset bonus is awarded exactly when the away
    team wins, and a division winner that wins its own wild-card game scores
    nothing for it.
    """
    b = season.bonuses
    stages = {
        "WC": (b.wild_card_upset_flat, b.wild_card_upset_add_lf),
        "DIV": (b.divisional_flat, b.divisional_add_lf),
        "CON": (b.conference_flat, b.conference_add_lf),
        "SB": (b.super_bowl_flat, b.super_bowl_add_lf),
    }

    pts = np.zeros(season.n_teams)
    po = games[games["played"] & games["game_type"].isin(stages)]

    for row in po.itertuples():
        flat, add_lf = stages[row.game_type]
        if row.game_type == "WC":
            # The shortcut below only holds when the higher seed actually
            # hosts. A neutral-site or relocated wild-card game breaks it
            # silently, so refuse to guess: the build fails loudly and
            # yesterday's correct site stays live until the scoring is fixed.
            location = getattr(row, "location", "Home")
            if pd.notna(location) and str(location) != "Home":
                raise ValueError(
                    f"wild-card game {row.game_id} is at a neutral site "
                    f"({location!r}) — the home-team-is-a-division-winner "
                    f"shortcut no longer holds; score this round from seeds."
                )
            # Only an away (wild-card) win scores.
            if not row.away_won:
                continue
            wi = season.idx[row.away_team]
        else:
            if row.is_tie:  # cannot happen in the playoffs, but never silently guess
                continue
            wi = season.idx[row.home_team if row.home_won else row.away_team]
        pts[wi] += flat + (season.lf[wi] if add_lf else 0.0)
    return pts


def score_berths(season: Season, seeds: dict[str, int] | None) -> np.ndarray:
    """(32,) division-winner and wild-card berth bonuses.

    ``seeds`` maps team -> conference seed 1-7. Seeds 1-4 are division winners,
    5-7 are wild cards; the two bonuses are mutually exclusive by construction.
    Pass ``None`` before the field is settled and no berth points are awarded.
    """
    pts = np.zeros(season.n_teams)
    if not seeds:
        return pts
    for team, seed in seeds.items():
        i = season.idx[team]
        pts[i] += (
            season.bonuses.division_winner if seed <= 4 else season.bonuses.wild_card_berth
        )
    return pts


def score_teams(
    season: Season,
    games: pd.DataFrame,
    seeds: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Full per-team points breakdown, indexed by team code.

    Columns: ``lf``, ``w``, ``l``, ``t``, ``win_points``, ``berth_points``,
    ``playoff_points``, ``total``.
    """
    from .nflverse import team_records

    rec = team_records(games, season.teams)
    win_pts = score_regular_season(season, games)
    berth_pts = score_berths(season, seeds)
    po_pts = score_playoffs(season, games)

    df = pd.DataFrame(
        {
            "lf": season.lf,
            "w": rec["w"].to_numpy(),
            "l": rec["l"].to_numpy(),
            "t": rec["t"].to_numpy(),
            "win_points": _round(win_pts),
            "berth_points": _round(berth_pts),
            "playoff_points": _round(po_pts),
            "total": _round(win_pts + berth_pts + po_pts),
        },
        index=list(season.teams),
    )
    df.index.name = "team"
    if seeds:
        df["seed"] = pd.Series(seeds, dtype="Int64").reindex(df.index)
    return df


def entrant_scores(season: Season, team_points: pd.DataFrame) -> pd.DataFrame:
    """Per-entrant totals and ranks, plus each entrant's team contributions.

    Ranking is competition-style: equal totals share a rank, and the next rank
    skips. Sorted best-first.
    """
    totals = team_points["total"].reindex(list(season.teams)).to_numpy()
    picks = season.picks_matrix()
    entrant_totals = _round(picks @ totals)

    df = pd.DataFrame(
        {
            "name": [e.name for e in season.entrants],
            "slug": [e.slug for e in season.entrants],
            "teams": [list(e.teams) for e in season.entrants],
            "total": entrant_totals,
        }
    )
    df["contributions"] = [
        {t: float(team_points.loc[t, "total"]) for t in e.teams} for e in season.entrants
    ]
    df = df.sort_values("total", ascending=False, kind="stable").reset_index(drop=True)
    df["rank"] = df["total"].rank(ascending=False, method="min").astype(int)
    return df


def payout_for_rank(season: Season, rank: int) -> float:
    """Dollars a given finishing rank pays, before splitting ties."""
    if 1 <= rank <= len(season.payout_split):
        return season.payouts[rank - 1]
    return 0.0


def money_if_season_ended(season: Season, standings: pd.DataFrame) -> pd.Series:
    """Payout each entrant would collect if the season ended right now.

    Entrants tied on points split the sum of the ranks they jointly occupy,
    which is how a real pool settles a tie rather than paying both in full.
    """
    out = pd.Series(0.0, index=standings.index)
    for _, group in standings.groupby("rank", sort=True):
        rank = int(group["rank"].iloc[0])
        n = len(group)
        shared = sum(payout_for_rank(season, r) for r in range(rank, rank + n))
        if shared:
            out.loc[group.index] = _round(shared / n)
    return out
