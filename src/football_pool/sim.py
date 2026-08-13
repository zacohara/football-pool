"""Monte Carlo season simulation, conditioned on games already played.

Adapted from the owner's preseason pick optimizer. That script answered "which
four teams should I take?"; this one answers "given what has already happened,
how is this likely to end?" — so the half that modelled ~29 unknown opponents
and searched every four-team combination is gone. The field is known now: it is
whoever is in ``picks.yaml``.

What carries over is the season engine: Elo fitted so expected wins match the
market, a full 272-game regular season, real playoff seeding, and the pool's
exact scoring rules — all vectorised across simulations.

The one structural change is conditioning. Games with a final score are frozen
to what actually happened; only the rest are sampled. So on week 12 the model
is simulating six weeks and a bracket, not a season.

Nothing here affects anybody's score. Every number this module produces is a
projection, and the site renders it in a colour reserved for modelled values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .season import Season

N_PLAYOFF_GAMES = 13  # 2 * (3 wild card + 2 divisional + 1 conference) + Super Bowl


@dataclass(frozen=True)
class SimConfig:
    """Model constants, loaded from a season's ``forecast.yaml``."""

    simulations: int = 25_000
    seed: int = 7
    home_field_elo: float = 48.0
    playoff_elo_multiplier: float = 1.2
    elo_scale: float = 400.0
    rating_shock_sd: float = 70.0
    qb_out_season_prob: float = 0.35
    qb_missed_low: int = 2
    qb_missed_high: int = 9
    qb_elo_drop: float = 110.0

    @classmethod
    def from_forecast(cls, forecast: Mapping | None) -> SimConfig:
        if not forecast:
            return cls()
        model = forecast.get("model", {})
        lo, hi = model.get("qb_missed_games", [2, 9])
        return cls(
            simulations=int(forecast.get("simulations", 25_000)),
            seed=int(forecast.get("seed", 7)),
            home_field_elo=float(model.get("home_field_elo", 48.0)),
            playoff_elo_multiplier=float(model.get("playoff_elo_multiplier", 1.2)),
            elo_scale=float(model.get("elo_scale", 400.0)),
            rating_shock_sd=float(model.get("rating_shock_sd", 70.0)),
            qb_out_season_prob=float(model.get("qb_out_season_prob", 0.35)),
            qb_missed_low=int(lo),
            qb_missed_high=int(hi),
            qb_elo_drop=float(model.get("qb_elo_drop", 110.0)),
        )


@dataclass(frozen=True)
class Schedule:
    """The regular season as index arrays, plus what has already happened."""

    home: np.ndarray  # (272,) team index
    away: np.ndarray  # (272,)
    team_games: np.ndarray  # (32, 17) game indices per team
    decided: np.ndarray  # (272,) bool — has a final score
    home_won: np.ndarray  # (272,) bool, meaningful where decided
    tied: np.ndarray  # (272,) bool

    @property
    def n_games(self) -> int:
        return len(self.home)

    @property
    def games_left(self) -> int:
        return int((~self.decided).sum())


def build_schedule(season: Season, games: pd.DataFrame) -> Schedule:
    """Index the regular season for simulation."""
    reg = games[games["game_type"] == "REG"]
    home = reg["home_team"].map(season.idx).to_numpy()
    away = reg["away_team"].map(season.idx).to_numpy()

    team_games = np.stack(
        [np.where((home == i) | (away == i))[0] for i in range(season.n_teams)]
    )
    return Schedule(
        home=home,
        away=away,
        team_games=team_games,
        decided=reg["played"].to_numpy(dtype=bool),
        home_won=reg["home_won"].to_numpy(dtype=bool),
        tied=reg["is_tie"].to_numpy(dtype=bool),
    )


def win_prob(elo_diff, scale: float = 400.0):
    """Win probability from an Elo difference (the 538 logistic)."""
    return 1.0 / (10.0 ** (-np.asarray(elo_diff) / scale) + 1.0)


def _qb_out_game_prob(cfg: SimConfig) -> float:
    """Marginal per-game probability that a team's starting QB is unavailable."""
    mean_missed = (cfg.qb_missed_low + cfg.qb_missed_high - 1) / 2
    return cfg.qb_out_season_prob * mean_missed / 17.0


def fit_elo(
    season: Season,
    schedule: Schedule,
    cfg: SimConfig,
    win_totals: Mapping[str, float],
    qual_sd: np.ndarray,
    iterations: int = 300,
    learning_rate: float = 18.0,
    quadrature_nodes: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ratings so expected wins over the real schedule match the market.

    Expected wins are marginalised over both sources of noise the simulation
    will later apply — QB availability and preseason rating uncertainty — using
    Gauss-Hermite quadrature. Skipping that marginalisation makes the fit
    systematically wrong: the logistic is concave in the tails, so extreme teams
    get compressed toward .500 in simulation even though the fit looked exact.

    The deliberate qualitative shift stays out of the fit. It is a considered
    deviation from the market, not an error to be calibrated away.
    """
    home, away = schedule.home, schedule.away
    shock_sd = np.sqrt(cfg.rating_shock_sd**2 + qual_sd**2)
    qb_drop = np.full(season.n_teams, cfg.qb_elo_drop)

    elo = np.full(season.n_teams, 1500.0)
    target = np.array([win_totals[t] for t in season.teams])
    p_qb_out = _qb_out_game_prob(cfg)

    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_nodes)
    weights = weights / np.sqrt(np.pi)
    game_sd = np.sqrt(shock_sd[home] ** 2 + shock_sd[away] ** 2)
    offsets = np.sqrt(2.0) * game_sd[:, None] * nodes[None, :]

    drop_h, drop_a = qb_drop[home][:, None], qb_drop[away][:, None]
    # Four joint QB-availability states: both up, home out, away out, both out.
    states = np.array(
        [
            (1 - p_qb_out) ** 2,
            p_qb_out * (1 - p_qb_out),
            (1 - p_qb_out) * p_qb_out,
            p_qb_out**2,
        ]
    )

    def expected_wins(ratings: np.ndarray) -> np.ndarray:
        diff = (ratings[home] - ratings[away] + cfg.home_field_elo)[:, None] + offsets
        p = (
            states[0] * win_prob(diff, cfg.elo_scale)
            + states[1] * win_prob(diff - drop_h, cfg.elo_scale)
            + states[2] * win_prob(diff + drop_a, cfg.elo_scale)
            + states[3] * win_prob(diff - drop_h + drop_a, cfg.elo_scale)
        )
        p = p @ weights
        return np.bincount(home, p, season.n_teams) + np.bincount(
            away, 1 - p, season.n_teams
        )

    for _ in range(iterations):
        elo += learning_rate * (target - expected_wins(elo))
        elo -= elo.mean() - 1500.0

    # Recomputed from the *final* ratings. The loop variable was one update
    # stale — the returned diagnostics did not correspond to the returned elo.
    return elo, expected_wins(elo)


def playoff_field(season: Season, wins: np.ndarray, jitter: np.ndarray) -> np.ndarray:
    """Seed every simulated season at once.

    Returns ``(n, 2, 7)`` team indices — conference on axis 1, seeds 1-7 on
    axis 2, division winners in the first four slots. Ties are broken by small
    random jitter rather than the NFL's full procedure, which is the right
    trade for a forecast (the real ladder is used for actual standings).
    """
    n = wins.shape[0]
    score = wins + jitter
    rows = np.arange(n)
    seeds = np.empty((n, 2, 7), dtype=np.int64)

    for ci, conf in enumerate(("AFC", "NFC")):
        conf_idx = np.array([season.idx[t] for t in season.conference_teams(conf)])
        divisions = [
            np.array([season.idx[t] for t in ts]) for ts in season.conference_divisions(conf)
        ]
        position = np.full(season.n_teams, -1)
        position[conf_idx] = np.arange(len(conf_idx))

        winners = np.stack(
            [div[np.argmax(score[:, div], axis=1)] for div in divisions], axis=1
        )
        order = np.argsort(
            -np.take_along_axis(score, winners, axis=1), axis=1, kind="stable"
        )
        winners = np.take_along_axis(winners, order, axis=1)

        conf_scores = score[:, conf_idx].copy()
        conf_scores[rows[:, None], position[winners]] = -np.inf  # remove champions
        wildcards = np.argsort(-conf_scores, axis=1, kind="stable")[:, :3]

        seeds[:, ci, :4] = winners
        seeds[:, ci, 4:] = conf_idx[wildcards]

    return seeds


def sim_playoffs(
    season: Season,
    seeds: np.ndarray,
    elo: np.ndarray,
    points: np.ndarray,
    uniforms: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Play every bracket in parallel, adding pool points into ``points``.

    Mirrors the scoring engine exactly: wild-card wins pay a flat bonus and only
    to the visiting wild card, later rounds add the team's leveling factor, and
    the Super Bowl is played at a neutral site.
    """
    b = season.bonuses
    lf = season.lf
    n = seeds.shape[0]
    rows = np.arange(n)

    def play(home, away, u, neutral=False):
        # Home-field Elo goes to the first argument — the higher seed, which
        # every caller passes first. (The parameter previously named
        # ``home_side`` was actually the away team; the maths was right, the
        # name was a landmine.)
        diff = (
            elo[rows, home] - elo[rows, away] + (0.0 if neutral else cfg.home_field_elo)
        ) * cfg.playoff_elo_multiplier
        return np.where(u < win_prob(diff, cfg.elo_scale), home, away)

    col = 0
    champions = []

    for ci in range(2):
        s = seeds[:, ci, :]
        alive = [s[:, 0]]  # the top seed has a bye
        alive_seed = [np.full(n, 1)]

        for high, low in ((1, 6), (2, 5), (3, 4)):  # 2v7, 3v6, 4v5
            winner = play(s[:, high], s[:, low], uniforms[:, col])
            col += 1
            # The host is always a division winner, so only a visiting win pays.
            upset = winner == s[:, low]
            bonus = b.wild_card_upset_flat + (
                lf[winner] if b.wild_card_upset_add_lf else 0.0
            )
            np.add.at(points, (rows[upset], winner[upset]), np.asarray(bonus)[upset] if np.ndim(bonus) else bonus)
            alive.append(winner)
            alive_seed.append(np.where(upset, low + 1, high + 1))

        teams = np.stack(alive, axis=1)
        ranks = np.stack(alive_seed, axis=1)
        order = np.argsort(ranks, axis=1, kind="stable")  # reseed
        teams = np.take_along_axis(teams, order, axis=1)
        ranks = np.take_along_axis(ranks, order, axis=1)

        w1 = play(teams[:, 0], teams[:, 3], uniforms[:, col])
        col += 1
        w2 = play(teams[:, 1], teams[:, 2], uniforms[:, col])
        col += 1
        s1 = np.where(w1 == teams[:, 0], ranks[:, 0], ranks[:, 3])
        s2 = np.where(w2 == teams[:, 1], ranks[:, 1], ranks[:, 2])
        for w in (w1, w2):
            points[rows, w] += b.divisional_flat + (lf[w] if b.divisional_add_lf else 0.0)

        high = np.where(s1 <= s2, w1, w2)
        low = np.where(s1 <= s2, w2, w1)
        champion = play(high, low, uniforms[:, col])
        col += 1
        points[rows, champion] += b.conference_flat + (
            lf[champion] if b.conference_add_lf else 0.0
        )
        champions.append(champion)

    winner = play(champions[0], champions[1], uniforms[:, col], neutral=True)
    points[rows, winner] += b.super_bowl_flat + (
        lf[winner] if b.super_bowl_add_lf else 0.0
    )
    return winner


def simulate(
    season: Season,
    schedule: Schedule,
    elo: np.ndarray,
    cfg: SimConfig,
    qual_mean: np.ndarray,
    qual_sd: np.ndarray,
    n: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Play the rest of the season ``n`` times.

    Returns ``(points, stats)`` — an ``(n, 32)`` array of pool points per team
    per simulated season, and a per-team summary.

    Decided games are frozen to their real result, so only what is genuinely
    unknown is sampled. A real tie is split deterministically across
    simulations — half assign the win to each side — which keeps expected points
    exactly right without needing to represent draws in the model.
    """
    n = n or cfg.simulations
    rng = np.random.default_rng(cfg.seed)
    n_games = schedule.n_games
    n_teams = season.n_teams
    rows = np.arange(n)

    # Per-simulation ratings: preseason uncertainty plus the researched shift.
    ratings = (
        elo
        + cfg.rating_shock_sd * rng.standard_normal((n, n_teams))
        + qual_mean
        + qual_sd * rng.standard_normal((n, n_teams))
    )

    # QB availability, drawn per (simulation, team), then spread over that
    # team's games by ranking uniforms — cheaper than sampling game indices.
    injured = rng.random((n, n_teams)) < cfg.qb_out_season_prob
    missed = rng.integers(cfg.qb_missed_low, cfg.qb_missed_high, size=(n, n_teams))
    order = np.argsort(rng.random((n, n_teams, 17)), axis=2)
    out = (order < missed[:, :, None]) & injured[:, :, None]

    penalty_home = np.zeros((n, n_games))
    penalty_away = np.zeros((n, n_games))
    is_home = schedule.home[schedule.team_games] == np.arange(n_teams)[:, None]
    for i in range(n_teams):
        g = schedule.team_games[i]
        drop = out[:, i, :] * cfg.qb_elo_drop
        penalty_home[:, g[is_home[i]]] -= drop[:, is_home[i]]
        penalty_away[:, g[~is_home[i]]] -= drop[:, ~is_home[i]]

    p_home = win_prob(
        ratings[:, schedule.home]
        + penalty_home
        - (ratings[:, schedule.away] + penalty_away)
        + cfg.home_field_elo,
        cfg.elo_scale,
    )
    home_wins = rng.random((n, n_games)) < p_home

    # Freeze what already happened.
    settled = schedule.decided & ~schedule.tied
    home_wins[:, settled] = schedule.home_won[settled]
    # A drawn game pays half to each side. Alternating the assignment across
    # simulations reproduces that expectation exactly.
    if schedule.tied.any():
        alternating = (rows[:, None] % 2 == 0)
        home_wins[:, schedule.tied] = alternating

    H = np.zeros((n_games, n_teams))
    H[np.arange(n_games), schedule.home] = 1.0
    A = np.zeros((n_games, n_teams))
    A[np.arange(n_games), schedule.away] = 1.0
    wins = home_wins @ H + (~home_wins) @ A

    if not np.all(wins.sum(axis=1) == n_games):
        raise RuntimeError("simulated wins are not zero-sum")

    points = wins * season.lf
    seeds = playoff_field(season, wins, rng.random((n, n_teams)) * 1e-3)
    np.add.at(points, (rows[:, None, None], seeds[:, :, :4]), season.bonuses.division_winner)
    np.add.at(points, (rows[:, None, None], seeds[:, :, 4:]), season.bonuses.wild_card_berth)
    champion = sim_playoffs(
        season, seeds, ratings, points, rng.random((n, N_PLAYOFF_GAMES)), cfg
    )

    stats = pd.DataFrame(
        {
            "team": list(season.teams),
            "lf": season.lf,
            "elo": np.round(elo, 0),
            "sim_wins": np.round(wins.mean(0), 2),
            "p_division": np.round(
                np.bincount(seeds[:, :, :4].ravel(), minlength=n_teams) / n, 3
            ),
            "p_wildcard": np.round(
                np.bincount(seeds[:, :, 4:].ravel(), minlength=n_teams) / n, 3
            ),
            "p_super_bowl": np.round(np.bincount(champion, minlength=n_teams) / n, 3),
            "mean_points": np.round(points.mean(0), 2),
            "p90_points": np.round(np.percentile(points, 90, axis=0), 2),
        }
    ).set_index("team")

    return points, stats
