"""Standings and playoff seeding.

The headline test replays five real seasons and asserts all 70 seeds match
ESPN's published playoff seeding exactly. That golden test is what makes the
+3.0 / +1.5 berth bonuses trustworthy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_pool.nflverse import parse_games
from football_pool.standings import (
    GameLog,
    final_seeds,
    playoff_seeds,
    regular_season_complete,
    standings_table,
)

FIXTURES = Path(__file__).parent / "fixtures"
SEASONS = ["2021", "2022", "2023", "2024", "2025"]


@pytest.fixture(scope="module")
def espn_seeds():
    return json.loads((FIXTURES / "espn_seeds.json").read_text())


@pytest.fixture(scope="module")
def multi_season_games():
    return {y: parse_games(FIXTURES / "games_2021_2025.csv", int(y)) for y in SEASONS}


# -- the golden test --------------------------------------------------------
@pytest.mark.parametrize("year", SEASONS)
def test_seeds_match_espn_exactly(season, multi_season_games, espn_seeds, year):
    derived = playoff_seeds(season, multi_season_games[year])
    expected = espn_seeds[year]

    assert len(derived) == 14, "seven seeds per conference"
    assert derived == expected, (
        f"{year}: derived seeding diverges from ESPN — "
        f"{ {t: (derived.get(t), expected.get(t)) for t in set(derived) | set(expected) if derived.get(t) != expected.get(t)} }"
    )


def test_every_conference_has_seeds_one_through_seven(season, multi_season_games):
    for year in SEASONS:
        seeds = playoff_seeds(season, multi_season_games[year])
        for conf in ("AFC", "NFC"):
            got = sorted(s for t, s in seeds.items() if season.conf_of[t] == conf)
            assert got == list(range(1, 8))


def test_division_winners_take_the_top_four_seeds(season, multi_season_games):
    """Seeds 1-4 must be exactly the four division champions."""
    for year in SEASONS:
        games = multi_season_games[year]
        seeds = playoff_seeds(season, games)
        for conf in ("AFC", "NFC"):
            top4 = {t for t, s in seeds.items() if s <= 4 and season.conf_of[t] == conf}
            assert {season.div_of[t] for t in top4} == {
                d for d in season.divisions if d.startswith(conf)
            }


# -- berth bonuses must not be awarded early --------------------------------
def test_final_seeds_are_withheld_until_the_season_ends(season, games_2025):
    assert regular_season_complete(games_2025)
    assert final_seeds(season, games_2025) is not None

    # Mid-season the way it actually looks upstream: week 18 unplayed AND no
    # playoff rows yet — nflverse does not publish the bracket until the field
    # is set, so a file cannot carry played playoff games ahead of week 18.
    partial = games_2025[games_2025["game_type"] == "REG"].copy()
    partial.loc[partial["week"] == 18, "played"] = False
    assert not regular_season_complete(partial)
    assert final_seeds(season, partial) is None


def test_a_cancelled_game_does_not_hold_the_berth_bonuses_hostage(season, games_2025):
    """Buffalo–Cincinnati, January 2023: one REG game keeps a null result
    forever. Once the bracket rows exist the berths are decided by
    definition, and the +3.0/+1.5 bonuses must bank."""
    cancelled = games_2025.copy()
    one = cancelled[cancelled["game_type"] == "REG"].index[40]
    cancelled.loc[one, "played"] = False
    assert regular_season_complete(cancelled)  # bracket rows exist
    assert final_seeds(season, cancelled) is not None


def test_seeds_are_still_projectable_mid_season(season, games_2025):
    """The site shows a projected field before week 18 — just not banked."""
    partial = games_2025.copy()
    partial.loc[partial["week"] > 10, "played"] = False
    seeds = playoff_seeds(season, partial)
    assert len(seeds) == 14


def test_no_games_played_projects_no_field_at_all(season, games_2025):
    """Preseason must project nothing rather than project alphabetical order.

    With every club 0-0 the ladder has nothing to separate anyone on, runs out
    of steps, and hits the coin-toss fallback — which sorts by name so builds
    stay reproducible. That used to hand back a complete fourteen-team field
    with Arizona and Baltimore holding the one seeds on the strength of their
    initials, and the Teams page displayed it as fact.
    """
    empty = games_2025.copy()
    empty["played"] = False

    assert playoff_seeds(season, empty) == {}
    assert final_seeds(season, empty) is None


def test_one_played_game_is_enough_to_start_projecting(season, games_2025):
    """The cutoff is "nothing has happened yet", not "not much has happened".

    Once any result exists the projection is informed by it, however thinly, so
    it is a real if volatile answer rather than a stand-in for the alphabet.
    """
    one = games_2025.copy()
    one["played"] = False
    first = one.index[(one["game_type"] == "REG").to_numpy()][0]
    one.loc[first, "played"] = True

    assert len(playoff_seeds(season, one)) == 14


def test_seeds_appear_the_moment_the_season_starts(season, games_2025):
    """Guards the boundary itself: empty, then not empty, same fixture."""
    empty = games_2025.copy()
    empty["played"] = False
    week_one = empty.copy()
    week_one.loc[week_one["week"] == 1, "played"] = True

    assert playoff_seeds(season, empty) == {}
    assert len(playoff_seeds(season, week_one)) == 14


# -- tiebreaker components --------------------------------------------------
def test_game_log_credits_a_tie_as_half(season, games_2025):
    gl = GameLog.build(season, games_2025)
    gb = [r for r in gl.log["GB"] if r[0] == "DAL"]
    assert any(r[3] == 0.5 for r in gb)


def test_win_pct_matches_the_record(season, games_2025):
    from football_pool.nflverse import team_records

    gl = GameLog.build(season, games_2025)
    rec = team_records(games_2025, season.teams)
    for t in season.teams:
        assert gl.win_pct(t) == pytest.approx(rec.loc[t, "win_pct"])


def test_common_games_needs_four_opponents(season, games_2025):
    gl = GameLog.build(season, games_2025)
    # Two clubs across a full season always share at least four opponents.
    assert gl.common_games_pct("KC", ["KC", "DEN"]) is not None
    # A single-game log cannot reach the threshold.
    tiny = GameLog(season, {t: [] for t in season.teams})
    tiny.log["KC"] = [("DEN", 20, 10, 1.0)]
    tiny.log["DEN"] = [("KC", 10, 20, 0.0)]
    assert tiny.common_games_pct("KC", ["KC", "DEN"]) is None


def test_head_to_head_sweep_semantics(season):
    """Conference ties need a clean sweep; a split counts for neither club."""
    gl = GameLog(season, {t: [] for t in season.teams})
    gl.log["KC"] = [("DEN", 30, 0, 1.0), ("DEN", 27, 3, 1.0)]
    gl.log["DEN"] = [("KC", 0, 30, 0.0), ("KC", 3, 27, 0.0)]
    assert gl.h2h_sweep("KC", ["KC", "DEN"]) == 1
    assert gl.h2h_sweep("DEN", ["KC", "DEN"]) == -1

    split = GameLog(season, {t: [] for t in season.teams})
    split.log["KC"] = [("DEN", 30, 0, 1.0), ("DEN", 3, 27, 0.0)]
    split.log["DEN"] = [("KC", 0, 30, 0.0), ("KC", 27, 3, 1.0)]
    assert split.h2h_sweep("KC", ["KC", "DEN"]) == 0


def test_sweep_requires_playing_everyone(season):
    """Never played them -> not a sweep, regardless of results."""
    gl = GameLog(season, {t: [] for t in season.teams})
    gl.log["KC"] = [("DEN", 30, 0, 1.0)]
    assert gl.h2h_sweep("KC", ["KC", "DEN", "LV"]) == 0


def test_h2h_pct_is_none_when_they_never_met(season):
    gl = GameLog(season, {t: [] for t in season.teams})
    assert gl.h2h_pct("KC", ["KC", "DEN"]) is None


def test_rate_helpers_are_safe_on_an_empty_log(season):
    gl = GameLog(season, {t: [] for t in season.teams})
    assert gl.win_pct("KC") == 0.0
    assert gl.division_pct("KC") == 0.0
    assert gl.conference_pct("KC") == 0.0
    assert gl.net_points("KC") == 0
    assert gl.strength_of_victory("KC") == 0.0
    assert gl.strength_of_schedule("KC") == 0.0


def test_strength_of_victory_ignores_losses(season):
    gl = GameLog(season, {t: [] for t in season.teams})
    gl.log["KC"] = [("DEN", 20, 10, 1.0), ("LV", 3, 30, 0.0)]
    gl.log["DEN"] = [("KC", 10, 20, 0.0)]
    gl.log["LV"] = [("KC", 30, 3, 1.0)]
    # Only the beaten opponent (DEN, 0.000) counts.
    assert gl.strength_of_victory("KC") == pytest.approx(0.0)
    assert gl.strength_of_schedule("KC") == pytest.approx(0.5)


# -- the display table ------------------------------------------------------
def test_standings_table_shape_and_content(season, games_2025):
    df = standings_table(season, games_2025)
    assert len(df) == 32
    assert set(df["conference"]) == {"AFC", "NFC"}
    assert (df["w"] + df["l"] + df["t"] == 17).all()
    assert df["seed"].notna().sum() == 14
    assert df["net_points"].sum() == 0  # points scored are zero-sum league-wide
