"""NFL schedule and results, from the nflverse published CSV.

One HTTPS GET of a single file gives every game from 1999 through the current
season, with scores filled in as they go final. Upstream (Lee Sharpe's nfldata)
commits every 15-90 minutes year round and lands within minutes of a game
ending, so a once-a-day build always has the previous night's finals.

Deliberately does not depend on ``nflreadpy``: it would add six transitive
dependencies and a pre-1.0 API with no retry logic to fetch the same bytes.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from .season import TEAM_ALIASES

GAMES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
)

# Everything the pool needs; the upstream file has 46 columns.
COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
    "result",
    "overtime",
    "location",
]

# Playoff rounds, in bracket order. These rows do not exist until the bracket is
# actually set — during the regular season the file has only REG rows.
PLAYOFF_ROUNDS = ("WC", "DIV", "CON", "SB")

# nflverse dates every game by its US Eastern calendar date — including the
# 09:30 kickoffs played in London, which carry the Sunday they are watched on
# rather than a UK date. Reading the clock in any other zone puts the day
# boundary in the middle of a slate: once the clocks go back, the 19:37 Eastern
# job runs at 00:37 UTC, so a UTC date would call that evening's games late
# before they had kicked off.
SCHEDULE_TZ = ZoneInfo("America/New_York")

# How many days of results may be missing before publishing does more harm than
# not publishing. Measured, not guessed: replaying a real season at both cron
# times, a current file scores 0 at every build instant, a file up to thirty
# hours old scores at most 1, and thirty-six hours is the first to reach 2.
#
# So 2 keeps the degradation this was always meant to allow — an outage falls
# back to yesterday's numbers — while refusing anything that has lost a slate.
# Two consecutive failed runs is not an ordinary day.
#
# Days rather than games, deliberately. A count cannot span a sixteen-game
# Sunday and a one-game Super Bowl with a single threshold, and a threshold set
# for a slate keeps publishing a wrong leaderboard for a week and a half after a
# Sunday freeze. Days treat one missed Super Bowl exactly as one missed Sunday,
# with no second rule and no bracket calendar to keep in step.
STALE_DAYS_LIMIT = 2


class DataError(Exception):
    """Upstream data was unreachable or did not look like what we expect."""


@dataclass(frozen=True)
class GameData:
    """Season-filtered games plus provenance for the site's freshness display."""

    games: pd.DataFrame
    season: int
    fetched_at: datetime
    upstream_modified: datetime | None
    # "network"  — fetched live, the normal case
    # "cache"    — the committed copy, asked for deliberately with --offline
    # "fallback" — the network was tried and failed, so the committed copy is
    #              standing in. Distinct from "cache" on purpose: one is a
    #              choice and the other is a degraded state, and only the
    #              degraded one should be able to stop a publish.
    source: str

    @property
    def played(self) -> pd.DataFrame:
        return self.games[self.games["played"]]

    @property
    def unplayed(self) -> pd.DataFrame:
        return self.games[~self.games["played"]]

    @property
    def last_completed(self) -> datetime | None:
        """Kickoff date of the most recent completed game."""
        p = self.played
        if p.empty:
            return None
        return pd.to_datetime(p["gameday"]).max().to_pydatetime()

    @property
    def current_week(self) -> int | None:
        """Highest regular-season week with a completed game (None preseason)."""
        reg = self.played[self.played["game_type"] == "REG"]
        return None if reg.empty else int(reg["week"].max())

    @property
    def next_week(self) -> int | None:
        """Lowest regular-season week with games still to play."""
        reg = self.unplayed[self.unplayed["game_type"] == "REG"]
        return None if reg.empty else int(reg["week"].min())

    @property
    def days_behind(self) -> int:
        """How many days of results this file is missing. 0 when it is current.

        The oldest game that ought to have a result and does not, measured
        against the day the data was fetched. One number, and it calibrates
        itself the way no clock can: the committed copy can be six months old in
        August and still be perfectly correct, because nothing has been played.

        Measured from ``fetched_at`` rather than a live call to now(), so the
        answer is a property of this object and a test can construct any
        situation it likes.

        A rescheduled game is not counted — nflverse moves ``gameday`` forward
        when a game is postponed, so it stops being in the past.

        A *cancelled* game is not counted either, and cannot be, by a
        different route: a game that is never made up (Buffalo–Cincinnati,
        January 2023) keeps its date and a null result forever. Counting it
        overdue would have this guard refusing to publish from two days after
        the cancellation until the end of the season — including all of
        January. The tell is the frontier: staleness means the feed stopped
        delivering, so a missing result *behind* newer completed games is a
        hole in the schedule, not a stall in the feed. Only games overdue past
        the most recent completed game count.
        """
        asof = pd.Timestamp(self.fetched_at.astimezone(SCHEDULE_TZ).date())
        gameday = pd.to_datetime(self.games["gameday"])
        outstanding = gameday[~self.games["played"]]

        overdue = outstanding[outstanding < asof]
        played_days = gameday[self.games["played"]]
        if not played_days.empty:
            overdue = overdue[overdue > played_days.max()]
        if not overdue.empty:
            return int((asof - overdue.min()).days)

        if outstanding.empty:
            # Nothing outstanding is either a finished season or a file missing
            # rows it cannot know it is missing. Playoff rows do not exist
            # upstream until the bracket is set, so a copy frozen at the end of
            # week 18 has every game it knows about played and looks perfectly
            # current for the whole of January — the window that decides the
            # money. A truncated copy looks identical, because the rows it loses
            # are the most recent ones. Absence cannot be counted, so ask
            # instead how long we have been in the dark. A season that really
            # ended ran to a played Super Bowl; one that did not is still owed
            # rows that never arrived.
            finished = bool(
                (self.games["game_type"] == "SB").any() and self.games["played"].all()
            )
            if not finished:
                return int((asof - gameday.max()).days)

        return 0


def fetch_games(
    season: int,
    cache_path: Path | None = None,
    *,
    offline: bool = False,
    retries: int = 3,
    timeout: float = 30.0,
) -> GameData:
    """Fetch games for ``season``, falling back to the committed copy.

    The cache is a fallback, never a shortcut: an unreachable upstream degrades
    the site to yesterday's numbers instead of failing the build, but a
    reachable upstream is always preferred. Caching results and serving them
    without a network attempt is the one failure this site must never have.
    """
    raw: bytes | None = None
    upstream_modified: datetime | None = None
    # Initialised to the guarded state, not the exempt one. Every path below
    # reassigns this, so the value is dead today — but "cache" is what waives
    # the staleness check, and a default that waives it means any future path
    # that forgets to set it publishes unguarded while claiming to be a
    # deliberate offline build. Failing closed costs nothing here.
    source = "fallback"

    if not offline:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                # follow_redirects is required: GitHub release URLs 302 to a CDN
                # host, and httpx does not follow redirects by default.
                resp = httpx.get(
                    GAMES_URL,
                    follow_redirects=True,
                    timeout=timeout,
                    headers={"Accept-Encoding": "gzip"},
                )
                resp.raise_for_status()
                raw = resp.content
                lm = resp.headers.get("last-modified")
                if lm:
                    upstream_modified = parsedate_to_datetime(lm)
                source = "network"
                break
            except Exception as e:  # noqa: BLE001 - retried, then reported
                last_err = e
                if attempt < retries - 1:
                    time.sleep(2**attempt)
        if raw is None and cache_path is None:
            raise DataError(f"could not fetch {GAMES_URL} and no cache available") from last_err

    if raw is None:
        if cache_path is None or not cache_path.exists():
            raise DataError(f"no cached games file at {cache_path}")
        raw = cache_path.read_bytes()
        # Reading the committed copy because it was asked for is a different
        # event from reading it because the network failed, even though the
        # bytes are identical. Only the caller can tell them apart, and only
        # the second one is a reason to refuse to publish.
        source = "cache" if offline else "fallback"

    games = parse_games(raw, season)
    data = GameData(
        games=games,
        season=season,
        fetched_at=datetime.now(timezone.utc),
        upstream_modified=upstream_modified,
        source=source,
    )

    # A 200 is not the same as good data. A regenerated release asset or a stale
    # CDN object can answer perfectly while carrying no results at all, and
    # writing that over the committed copy destroys the very thing the fallback
    # exists to be. So the cache is only refreshed from data that is current.
    if source == "network" and cache_path is not None and data.days_behind < STALE_DAYS_LIMIT:
        # Cache only this season's rows. The upstream file carries every season
        # back to 1999 (~2 MB), and the daily job commits its cache — storing
        # the whole thing would add megabytes of near-identical history to the
        # repository every day for the sake of ~20 KB that changes.
        # Written to a sibling and renamed, because rename is atomic on POSIX
        # and a direct write is not. A run cancelled or killed mid-write would
        # otherwise leave a truncated file that pandas reads back perfectly
        # happily — the columns are all present and the season filter still
        # matches, so nothing raises. Because rows are sorted by week, the part
        # it loses is the most recent one, which is exactly the part that would
        # make the staleness check say everything is fine.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_path.with_suffix(cache_path.suffix + ".partial")
        games[COLUMNS].to_csv(partial, index=False)
        partial.replace(cache_path)

    return data


def parse_games(raw: bytes | str | Path, season: int) -> pd.DataFrame:
    """Parse the games CSV, filter to ``season``, and normalize team codes.

    This is the one boundary where upstream codes become pool codes. Everything
    downstream speaks pool codes only.
    """
    if isinstance(raw, Path):
        df = pd.read_csv(raw, usecols=lambda c: c in COLUMNS, low_memory=False)
    else:
        data = raw.encode() if isinstance(raw, str) else raw
        df = pd.read_csv(
            io.BytesIO(data), usecols=lambda c: c in COLUMNS, low_memory=False
        )

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            f"upstream games file is missing expected column(s) {missing} — "
            "the schema changed and the scoring code needs review"
        )

    df = df[df["season"] == season].copy()
    if df.empty:
        raise DataError(f"no games found for season {season}")

    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(TEAM_ALIASES)

    # `result` is the only correct played/unplayed test. Not `home_score > 0`
    # (a 0-score row is not the signal) and not `gameday < today` (games move).
    # Note a tie is result == 0, which is falsy — always test for null, never
    # truthiness.
    df["played"] = df["result"].notna()

    # pandas reads the nullable score columns as float64/NaN, which renders as
    # "24.0" on the page. Int64 (capital I) keeps them integral and nullable.
    for col in ("home_score", "away_score", "week"):
        df[col] = df[col].astype("Int64")

    df["is_tie"] = df["played"] & (df["result"] == 0)
    df["home_won"] = df["played"] & (df["result"] > 0)
    df["away_won"] = df["played"] & (df["result"] < 0)

    return df.sort_values(["week", "gameday", "game_id"]).reset_index(drop=True)


def validate_teams(games: pd.DataFrame, teams: tuple[str, ...]) -> None:
    """Assert the data's team set matches the season config exactly.

    Catches a missing alias, which would otherwise silently drop one team's
    points for the entire season rather than raising.
    """
    seen = set(games["home_team"]) | set(games["away_team"])
    expected = set(teams)
    if seen != expected:
        raise DataError(
            f"team codes in the data do not match rules.yaml. "
            f"In data only: {sorted(seen - expected)}. "
            f"In rules only: {sorted(expected - seen)}. "
            f"A missing entry in TEAM_ALIASES is the usual cause."
        )


def team_records(games: pd.DataFrame, teams: tuple[str, ...]) -> pd.DataFrame:
    """Win/loss/tie record per team from completed regular-season games."""
    rec = pd.DataFrame(
        0, index=list(teams), columns=["w", "l", "t"], dtype=int
    )
    reg = games[games["played"] & (games["game_type"] == "REG")]
    for row in reg.itertuples():
        h, a = row.home_team, row.away_team
        if row.is_tie:
            rec.loc[h, "t"] += 1
            rec.loc[a, "t"] += 1
        elif row.home_won:
            rec.loc[h, "w"] += 1
            rec.loc[a, "l"] += 1
        else:
            rec.loc[a, "w"] += 1
            rec.loc[h, "l"] += 1
    rec["gp"] = rec[["w", "l", "t"]].sum(axis=1)
    rec["win_pct"] = (rec["w"] + 0.5 * rec["t"]) / rec["gp"].where(rec["gp"] > 0)
    return rec.fillna({"win_pct": 0.0})
