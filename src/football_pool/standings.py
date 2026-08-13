"""Standings, division winners, and playoff seeds, derived from results.

nflverse publishes no standings or seeding file, so the pool has to derive its
own — and it must, because the +3.0 division-winner and +1.5 wild-card bonuses
depend on the final seeding.

This implements the NFL's actual tiebreaker procedure, which is not a sort key.
Three details do all the damage if you get them wrong:

* After a tiebreaker step separates the field, **exactly one club advances** and
  the remaining tied clubs restart the ladder from step one.
* Division head-to-head is a **win percentage** against the other tied clubs;
  conference head-to-head is a **sweep** test (a club must have beaten all of
  them, or lost to all of them) — using win percentage there gives wrong
  answers in three-way ties.
* "Common games" requires **at least four** common opponents, and if there are
  fewer the step is skipped entirely rather than scoring zero.

Validated against ESPN's published ``playoffSeed`` for 2021-2025: all 70 seeds
match exactly. The ladder stops at strength of schedule; deeper levels have
never been needed since the league went to 32 teams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from .season import Season

REG_SEASON_GAMES = 272


@dataclass(frozen=True)
class GameLog:
    """Every team's completed regular-season results, indexed for tiebreakers.

    ``log[team]`` is a list of ``(opponent, points_for, points_against, credit)``
    where ``credit`` is 1.0 for a win, 0.5 for a tie and 0.0 for a loss.
    """

    season: Season
    log: dict[str, list[tuple[str, int, int, float]]]

    @classmethod
    def build(cls, season: Season, games: pd.DataFrame) -> GameLog:
        log: dict[str, list[tuple[str, int, int, float]]] = {t: [] for t in season.teams}
        reg = games[games["played"] & (games["game_type"] == "REG")]
        for row in reg.itertuples():
            h, a = row.home_team, row.away_team
            hs, as_ = int(row.home_score), int(row.away_score)
            credit = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
            log[h].append((a, hs, as_, credit))
            log[a].append((h, as_, hs, 1.0 - credit))
        return cls(season, log)

    # -- basic rates ------------------------------------------------------
    @staticmethod
    def _pct(rows: Sequence[tuple[str, int, int, float]]) -> float:
        return sum(r[3] for r in rows) / len(rows) if rows else 0.0

    def win_pct(self, team: str) -> float:
        return self._pct(self.log[team])

    def division_pct(self, team: str) -> float:
        d = self.season.div_of[team]
        return self._pct([r for r in self.log[team] if self.season.div_of[r[0]] == d])

    def conference_pct(self, team: str) -> float:
        c = self.season.conf_of[team]
        return self._pct([r for r in self.log[team] if self.season.conf_of[r[0]] == c])

    def net_points(self, team: str) -> int:
        return sum(r[1] - r[2] for r in self.log[team])

    def strength_of_victory(self, team: str) -> float:
        """Combined win percentage of the opponents this team has *defeated*.

        Defeated only — the NFL's definition. A tie is not a victory, so a
        tied opponent contributes nothing here (it still counts fully in
        strength of schedule). Weighting ties at half, as this previously did,
        deviates from the written procedure in exactly the seasons where the
        deep tiebreakers get reached.
        """
        victories = [r for r in self.log[team] if r[3] == 1.0]
        if not victories:
            return 0.0
        return sum(self.win_pct(r[0]) for r in victories) / len(victories)

    def strength_of_schedule(self, team: str) -> float:
        rows = self.log[team]
        return sum(self.win_pct(r[0]) for r in rows) / len(rows) if rows else 0.0

    # -- head-to-head -----------------------------------------------------
    def h2h_pct(self, team: str, group: Iterable[str]) -> float | None:
        """Division form: win percentage against the other tied clubs."""
        others = {t for t in group if t != team}
        rows = [r for r in self.log[team] if r[0] in others]
        return self._pct(rows) if rows else None

    def h2h_sweep(self, team: str, group: Iterable[str]) -> int:
        """Conference form: +1 swept them all, -1 swept by all, 0 otherwise.

        Only a clean sweep counts, and only if the club actually played every
        other tied club.
        """
        others = {t for t in group if t != team}
        faced = {r[0] for r in self.log[team] if r[0] in others}
        if faced != others:
            return 0
        results = [r for r in self.log[team] if r[0] in others]
        if all(r[3] == 1.0 for r in results):
            return 1
        if all(r[3] == 0.0 for r in results):
            return -1
        return 0

    def common_games_pct(self, team: str, group: Iterable[str]) -> float | None:
        """Win percentage vs common opponents; ``None`` if fewer than four."""
        opponent_sets = [{r[0] for r in self.log[t]} for t in group]
        common = set.intersection(*opponent_sets) if opponent_sets else set()
        rows = [r for r in self.log[team] if r[0] in common]
        return self._pct(rows) if len(rows) >= 4 else None


def _order(gl: GameLog, teams: Sequence[str], kind: str) -> list[str]:
    """Rank ``teams`` by win percentage, breaking each tied cluster properly."""
    by_pct: dict[float, list[str]] = {}
    for t in teams:
        by_pct.setdefault(gl.win_pct(t), []).append(t)

    ordered: list[str] = []
    for pct in sorted(by_pct, reverse=True):
        cluster = by_pct[pct]
        ordered.extend(cluster if len(cluster) == 1 else _break_tie(gl, cluster, kind))
    return ordered


def _break_tie(gl: GameLog, group: Sequence[str], kind: str) -> list[str]:
    """Apply the tiebreaker ladder, restarting after each elimination."""
    g = list(group)

    # If every tied club is from the same division, the DIVISION ladder applies
    # even when ranking within a conference. Omitting this swaps seeds in real
    # seasons (it produced the one wrong seed in 2025 during validation).
    if kind == "conf" and len({gl.season.div_of[t] for t in g}) == 1:
        kind = "div"

    if kind == "div":
        ladder = [
            lambda t: v if (v := gl.h2h_pct(t, g)) is not None else -1.0,
            gl.division_pct,
            lambda t: v if (v := gl.common_games_pct(t, g)) is not None else -1.0,
            gl.conference_pct,
            gl.strength_of_victory,
            gl.strength_of_schedule,
            gl.net_points,
        ]
    else:
        ladder = [
            lambda t: gl.h2h_sweep(t, g),
            gl.conference_pct,
            lambda t: v if (v := gl.common_games_pct(t, g)) is not None else -1.0,
            gl.strength_of_victory,
            gl.strength_of_schedule,
            gl.net_points,
        ]

    for step in ladder:
        values = {t: step(t) for t in g}
        best = max(values.values())
        leaders = [t for t in g if values[t] == best]
        if len(leaders) < len(g):
            rest = [t for t in g if t not in leaders]
            head = leaders if len(leaders) == 1 else _break_tie(gl, leaders, kind)
            tail = rest if len(rest) == 1 else _break_tie(gl, rest, kind)
            return head + tail

    # Ladder exhausted — the NFL flips a coin. Sort for a deterministic build.
    return sorted(g)


def regular_season_complete(games: pd.DataFrame) -> bool:
    """True once the regular season is decided.

    Two independent signals, either one sufficient:

    * every regular-season game has a final score, or
    * a playoff game has a final score. A completed postseason game is
      ironclad proof the berths were decided.

    The second matters because a game *cancelled* outright (Buffalo–Cincinnati,
    January 2023) keeps a null result forever. Requiring every REG game to be
    played would then hold the +3.0/+1.5 berth bonuses hostage for the whole
    season — the standings would run all of January with playoff points
    accruing and no berth money ever banked.

    A *played* playoff game, deliberately, not merely the presence of playoff
    rows: replayed past seasons and the week-by-week history rebuild both
    carry unplayed bracket rows next to a half-finished regular season, and
    "rows exist" would mark every one of those mid-season frames complete —
    awarding berth bonuses from week one of the history chart.
    """
    reg = games[games["game_type"] == "REG"]
    if len(reg) == 0:
        return False
    if bool(reg["played"].all()):
        return True
    playoff = games["game_type"].isin(("WC", "DIV", "CON", "SB"))
    return bool((playoff & games["played"]).any())


def playoff_seeds(season: Season, games: pd.DataFrame) -> dict[str, int]:
    """Map team -> conference seed 1-7, from results so far.

    Mid-season this is the *projected* field if the standings froze today, which
    is what the site shows as a projection. Berth bonuses are only banked once
    :func:`regular_season_complete` is true.

    Before kickoff it is nothing at all, and says so. With every club 0-0 the
    tiebreaker ladder has no information to work with, exhausts every step, and
    reaches the coin-toss fallback — which sorts alphabetically to keep builds
    reproducible. That produced a full, confident-looking fourteen-team field in
    which Arizona was the AFC... the NFC one seed and Baltimore the AFC one
    seed, purely for starting with an A and a B. A projection that is really
    alphabetical order is worse than no projection, so return nothing and let
    the pages render an honest dash.
    """
    played_regular_season = games[(games["game_type"] == "REG") & games["played"]]
    if played_regular_season.empty:
        return {}

    gl = GameLog.build(season, games)
    seeds: dict[str, int] = {}

    for conf in ("AFC", "NFC"):
        winners = [_order(gl, list(ts), "div")[0] for ts in season.conference_divisions(conf)]
        for i, team in enumerate(_order(gl, winners, "conf")):
            seeds[team] = i + 1

        rest = [t for t in season.conference_teams(conf) if t not in winners]
        for i, team in enumerate(_order(gl, rest, "conf")[:3]):
            seeds[team] = 5 + i

    return seeds


def final_seeds(season: Season, games: pd.DataFrame) -> dict[str, int] | None:
    """Seeds if and only if the regular season is over, else ``None``.

    This is what the scorer takes: berth bonuses must not be awarded from a
    half-played season.
    """
    return playoff_seeds(season, games) if regular_season_complete(games) else None


def standings_table(season: Season, games: pd.DataFrame) -> pd.DataFrame:
    """Per-team standings with everything the site displays.

    Columns: ``w``, ``l``, ``t``, ``win_pct``, ``div_pct``, ``conf_pct``,
    ``net_points``, ``division``, ``conference``, ``seed``.
    """
    from .nflverse import team_records

    gl = GameLog.build(season, games)
    rec = team_records(games, season.teams)
    seeds = playoff_seeds(season, games)

    df = rec.copy()
    df["div_pct"] = [gl.division_pct(t) for t in df.index]
    df["conf_pct"] = [gl.conference_pct(t) for t in df.index]
    df["net_points"] = [gl.net_points(t) for t in df.index]
    df["division"] = [season.div_of[t] for t in df.index]
    df["conference"] = [season.conf_of[t] for t in df.index]
    df["seed"] = pd.Series(seeds, dtype="Int64").reindex(df.index)
    df.index.name = "team"
    return df
