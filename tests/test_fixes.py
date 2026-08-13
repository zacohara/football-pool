"""Regression tests for the bug-fix pass.

One test per fixed behaviour, each written to fail against the pre-fix code.
Grouped by module, referencing the January-window failures the pass was
mostly about: four separate ways the site used to degrade exactly during the
weeks the money gets decided.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from football_pool import cli
from football_pool.nflverse import GameData, parse_games
from football_pool.render import _pool_state, _team_rows, build_context, make_environment
from football_pool.scoring import entrant_scores, score_teams
from football_pool.season import ConfigError, load_season
from football_pool.standings import GameLog, final_seeds
from football_pool import history as history_mod

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def games_2025():
    return parse_games(FIXTURES / "games_2025.csv", 2025)


def _gd(games):
    return GameData(games, 2025, datetime(2026, 1, 20, tzinfo=timezone.utc), None, "cache")


# -- cli: berth bonuses on the terminal leaderboard -------------------------
def test_cli_standings_awards_berth_bonuses(monkeypatch, season, games_2025, capsys):
    """The CLI and the site must not disagree in January."""
    monkeypatch.setattr(cli, "load_season", lambda year: season)
    monkeypatch.setattr(
        "football_pool.nflverse.fetch_games", lambda *a, **k: _gd(games_2025)
    )
    assert cli.main(["standings"]) == 0
    out = capsys.readouterr().out

    seeded = entrant_scores(
        season, score_teams(season, games_2025, final_seeds(season, games_2025))
    )
    assert f"{float(seeded['total'].iloc[0]):.2f}" in out


# -- nflverse: a cancelled game is not staleness ----------------------------
def test_a_hole_behind_newer_results_is_not_staleness(games_2025):
    """One cancelled game used to make `pool build` refuse to publish from two
    days after the cancellation until the end of the season."""
    cancelled = games_2025.copy()
    reg = cancelled[cancelled["game_type"] == "REG"]
    one = reg.index[40]
    cancelled.loc[one, "played"] = False

    sb = pd.Timestamp(cancelled[cancelled["game_type"] == "SB"]["gameday"].max())
    later = (sb + pd.Timedelta(days=3)).to_pydatetime().replace(tzinfo=timezone.utc)
    gd = GameData(cancelled, 2025, later, None, "network")
    assert gd.days_behind == 0


def test_a_stalled_feed_is_still_caught(games_2025):
    """The frontier rule must not weaken the guard's real job."""
    frozen = games_2025.copy()
    sunday = pd.Timestamp("2025-11-16")
    frozen.loc[pd.to_datetime(frozen["gameday"]) > sunday, "played"] = False
    asof = (sunday + pd.Timedelta(days=5)).to_pydatetime().replace(tzinfo=timezone.utc)
    assert GameData(frozen, 2025, asof, None, "network").days_behind >= 2


# -- standings: SOV counts victories only -----------------------------------
def test_strength_of_victory_excludes_ties(season):
    gl = GameLog(season, {t: [] for t in season.teams})
    gl.log["KC"] = [("DEN", 20, 10, 1.0), ("LV", 14, 14, 0.5)]
    gl.log["DEN"] = [("KC", 10, 20, 0.0)]
    gl.log["LV"] = [("KC", 14, 14, 0.5), ("DEN", 30, 0, 1.0)]
    # Only the defeated opponent (DEN, .000) counts; the tied one (LV, .750
    # here) previously leaked in at half weight and lifted the number.
    assert gl.strength_of_victory("KC") == pytest.approx(0.0)


# -- scoring: neutral-site wild-card guard ----------------------------------
def test_a_neutral_site_wild_card_game_refuses_to_guess(season, games_2025):
    g = games_2025.copy()
    wc = g[g["game_type"] == "WC"].index[0]
    g.loc[wc, "location"] = "Neutral"
    with pytest.raises(ValueError, match="neutral site"):
        score_teams(season, g, final_seeds(season, g))


# -- history: mover ordering and zero-point leaders --------------------------
def test_fallers_lead_with_the_biggest_drop():
    hist = pd.DataFrame(
        {"rank_w1": [1, 2, 3, 4], "rank_w2": [1, 4, 2, 3]},
        index=["a", "b", "c", "d"],
    )
    hist.attrs["weeks"] = [1, 2]
    out = history_mod.movers(hist)
    assert out["fallers"][0] == ("b", -2.0)


def test_week_leaders_never_crowns_a_zero(season):
    hist = pd.DataFrame({"w1": [3.0, 0.0], "rank_w1": [1, 2]}, index=["a", "b"])
    hist.attrs["weeks"] = [1]
    leaders = history_mod.week_leaders(season, hist)
    assert leaders == [("a", 3.0)]


# -- render: the January window ---------------------------------------------
def _ctx(season, games):
    return build_context(season, _gd(games))


def test_phase_is_playoffs_before_the_bracket_rows_exist(season, games_2025):
    """Week 18 done, nflverse yet to publish the bracket: zero unplayed rows
    used to render the headline as 'Final' with the whole postseason left."""
    gap = games_2025[games_2025["game_type"] == "REG"].copy()
    state = _pool_state(_ctx(season, gap), [])
    assert state["phase"] == "Playoffs"


def test_phase_is_final_only_after_a_played_super_bowl(season, games_2025):
    assert _pool_state(_ctx(season, games_2025), [])["phase"] == "Final"


def test_best_of_week_uses_the_scored_week(season, games_2025):
    """January points must not sit under a week-18 headline."""
    mid = games_2025.copy()
    mid.loc[mid["game_type"].isin({"DIV", "CON", "SB"}), "played"] = False
    state = _pool_state(_ctx(season, mid), [])
    assert state["week"] == 18  # max REG week, unchanged
    assert state["scored_week"] == 19  # what the panel must show


def test_reg_games_remaining_is_zero_during_the_playoffs(season, games_2025):
    mid = games_2025.copy()
    mid.loc[mid["game_type"].isin({"DIV", "CON", "SB"}), "played"] = False
    state = _pool_state(_ctx(season, mid), [])
    assert state["reg_games_remaining"] == 0
    assert state["games_remaining"] > 0  # the pair the forecast page branches on


def test_ordinal_filter_survives_the_twenties(season):
    ordinal = make_environment().filters["ordinal"]
    assert [ordinal(n) for n in (1, 2, 3, 11, 21, 22, 23)] == [
        "1st", "2nd", "3rd", "11th", "21st", "22nd", "23rd",
    ]


def test_team_rows_carry_a_win_pct_sort_key(season, games_2025):
    rows = _team_rows(_ctx(season, games_2025))
    assert all(0.0 <= r["win_pct"] <= 1.0 for r in rows)


# -- season: loud config errors for the money -------------------------------
def _rewrite(season_writer, tmp_path, overrides):
    season_writer(
        tmp_path, 2031,
        [{"name": "Solo", "teams": ["KC", "SEA", "DAL", "NE"]}],
        rules_overrides=overrides, forecast=False,
    )
    return load_season(2031, root=tmp_path)


def test_an_overfull_payout_split_is_rejected(season_writer, tmp_path):
    with pytest.raises(ConfigError, match="more than 100%"):
        _rewrite(season_writer, tmp_path, {"payout_split": [0.5, 0.3, 0.3]})


def test_a_negative_entry_fee_is_rejected(season_writer, tmp_path):
    with pytest.raises(ConfigError, match="entry_fee is negative"):
        _rewrite(season_writer, tmp_path, {"entry_fee": -10})


def test_a_negative_payout_share_is_rejected(season_writer, tmp_path):
    with pytest.raises(ConfigError, match="negative share"):
        _rewrite(season_writer, tmp_path, {"payout_split": [1.0, -0.5, 0.3]})


def test_a_missing_flat_bonus_is_a_config_error_not_a_traceback(
    season_writer, tmp_path
):
    import yaml

    sdir = season_writer(
        tmp_path, 2032,
        [{"name": "Solo", "teams": ["KC", "SEA", "DAL", "NE"]}],
        forecast=False,
    )
    rules = yaml.safe_load((sdir / "rules.yaml").read_text())
    del rules["scoring"]["division_winner"]
    (sdir / "rules.yaml").write_text(yaml.safe_dump(rules))
    with pytest.raises(ConfigError, match="scoring.division_winner"):
        load_season(2032, root=tmp_path)


def test_the_placeholder_hint_does_not_teach_the_yaml_no_trap(
    season_writer, tmp_path
):
    """The error's own example used to be `teams: [CIN, WAS, TEN, NO]` —
    unquoted NO, the exact boolean trap the loader exists to catch."""
    season_writer(
        tmp_path, 2033,
        [{"name": "Solo", "teams": ["TODO", "SEA", "DAL", "NE"]}],
        forecast=False,
    )
    with pytest.raises(ConfigError, match=r'\["CIN", "WAS", "TEN", "NO"\]'):
        load_season(2033, root=tmp_path)
