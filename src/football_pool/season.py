"""Season configuration — everything that changes from one year to the next.

The whole point of this module is that no pool rule, leveling factor, entrant,
or dollar figure is ever a constant in code. A :class:`Season` is loaded from
``seasons/<year>/`` and threaded through every other module, so building a
different year is a different argument, not a different program.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

# Upstream data uses a few different codes than the pool does. Normalization
# happens at exactly one boundary (see nflverse.parse_games); everything
# downstream speaks pool codes.
TEAM_ALIASES: dict[str, str] = {"LA": "LAR", "WSH": "WAS", "JAC": "JAX"}


def _find_repo_root() -> Path:
    """Locate the directory holding config.yaml, seasons/, and templates/.

    ``Path(__file__).parents[2]`` is only the repo root when running from a
    source checkout. Installed as a wheel it resolves to site-packages'
    grandparent (e.g. ``/usr/local/lib/python3.12``), where none of the data
    files exist — which made the packaged ``pool`` console script dead on
    arrival. Resolution order:

    1. ``POOL_ROOT`` environment variable, set explicitly.
    2. Walking up from this file — finds the checkout root when running from
       source, exactly as before.
    3. The current working directory — so an installed ``pool`` works when run
       from inside a checkout.
    """
    env = os.environ.get("POOL_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.yaml").is_file() and (parent / "seasons").is_dir():
            return parent
    return Path.cwd()


REPO_ROOT = _find_repo_root()


class ConfigError(Exception):
    """Raised when the season config is wrong in a way a human must fix.

    These are deliberately loud. A typo in picks.yaml would otherwise silently
    score someone zero for a team all season, and nobody would notice until
    Thanksgiving.
    """


@dataclass(frozen=True)
class Bonuses:
    """Point values for everything that is not a regular-season win."""

    division_winner: float
    wild_card_berth: float
    wild_card_upset_flat: float
    wild_card_upset_add_lf: bool
    divisional_flat: float
    divisional_add_lf: bool
    conference_flat: float
    conference_add_lf: bool
    super_bowl_flat: float
    super_bowl_add_lf: bool


@dataclass(frozen=True)
class Entrant:
    """One person's entry: a name and the teams they picked."""

    name: str
    teams: tuple[str, ...]

    @property
    def slug(self) -> str:
        """URL-safe id, stable across builds (used for /entrant/<slug>/)."""
        s = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return s or "entrant"


@dataclass(frozen=True)
class Season:
    """A fully-resolved season: rules, teams, entrants, and derived arrays."""

    year: int
    teams: tuple[str, ...]
    lf: np.ndarray  # (32,) leveling factor, indexed by team position
    divisions: Mapping[str, tuple[str, ...]]
    bonuses: Bonuses
    win_multiplier: float
    tie_multiplier: float
    entry_fee: float
    payout_split: tuple[float, ...]
    picks_per_entrant: int
    entrants: tuple[Entrant, ...]
    forecast: Mapping[str, Any] | None = None

    # Derived lookups, built in __post_init__.
    idx: Mapping[str, int] = field(default_factory=dict)
    div_of: Mapping[str, str] = field(default_factory=dict)
    conf_of: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "idx", {t: i for i, t in enumerate(self.teams)})
        div_of = {t: d for d, ts in self.divisions.items() for t in ts}
        object.__setattr__(self, "div_of", div_of)
        object.__setattr__(
            self, "conf_of", {t: d.split()[0] for t, d in div_of.items()}
        )

    # -- convenience ------------------------------------------------------
    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def pot(self) -> float:
        """Total money in the pool.

        Derived from the actual entrant count rather than an assumed 30, so
        every dollar figure on the site stays correct as picks trickle in.
        """
        return len(self.entrants) * self.entry_fee

    @property
    def payouts(self) -> tuple[float, ...]:
        """Dollar payout for 1st, 2nd, 3rd."""
        return tuple(self.pot * s for s in self.payout_split)

    def lf_of(self, team: str) -> float:
        return float(self.lf[self.idx[team]])

    def picks_matrix(self) -> np.ndarray:
        """(n_entrants, n_teams) 0/1 matrix — entrant totals are a matmul away."""
        m = np.zeros((len(self.entrants), self.n_teams))
        for i, e in enumerate(self.entrants):
            m[i, [self.idx[t] for t in e.teams]] = 1.0
        return m

    def conference_teams(self, conf: str) -> list[str]:
        return [t for t in self.teams if self.conf_of[t] == conf]

    def conference_divisions(self, conf: str) -> list[tuple[str, ...]]:
        return [ts for d, ts in self.divisions.items() if d.startswith(conf)]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def team_code(value: Any, where: str) -> str:
    """Coerce a YAML scalar to a team code, catching the YAML 1.1 boolean trap.

    YAML 1.1 reads a bare ``NO`` as the boolean ``false``, which would silently
    erase New Orleans from the pool. Every team code in the config files is
    quoted for that reason; this catches it if one ever isn't.
    """
    if isinstance(value, bool):
        raise ConfigError(
            f"{where}: found the boolean {str(value).lower()!r} where a team code "
            f"was expected. YAML reads a bare NO as false — quote it as \"NO\"."
        )
    return str(value).strip().upper()


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} did not parse to a mapping")
    return data


def active_season(root: Path | None = None) -> int:
    """The season the site builds by default (from config.yaml)."""
    root = root or REPO_ROOT
    return int(_read_yaml(root / "config.yaml")["active_season"])


def load_season(year: int | None = None, root: Path | None = None) -> Season:
    """Load ``seasons/<year>/`` into a :class:`Season`.

    Raises:
        ConfigError: on any malformed rules or picks. Loud on purpose.
    """
    root = root or REPO_ROOT
    year = year if year is not None else active_season(root)
    sdir = root / "seasons" / str(year)
    if not sdir.is_dir():
        raise ConfigError(
            f"no config for season {year} (expected {sdir}). "
            f"To start a new season: cp -r seasons/<prev> {sdir}"
        )

    rules = _read_yaml(sdir / "rules.yaml")
    teams, lf, divisions = _parse_teams(rules, sdir)
    bonuses, win_mult, tie_mult = _parse_scoring(rules, sdir)
    _validate_money(rules, sdir)
    entrants = _parse_picks(_read_yaml(sdir / "picks.yaml"), set(teams), rules, sdir)

    # forecast.yaml is optional and never affects scoring.
    forecast = None
    fpath = sdir / "forecast.yaml"
    if fpath.exists():
        f = _read_yaml(fpath)
        if f.get("enabled", True):
            forecast = f

    return Season(
        year=year,
        teams=teams,
        lf=lf,
        divisions=divisions,
        bonuses=bonuses,
        win_multiplier=win_mult,
        tie_multiplier=tie_mult,
        entry_fee=float(rules["entry_fee"]),
        payout_split=tuple(float(x) for x in rules["payout_split"]),
        picks_per_entrant=int(rules.get("picks_per_entrant", 4)),
        entrants=entrants,
        forecast=forecast,
    )


def _parse_teams(
    rules: dict, sdir: Path
) -> tuple[tuple[str, ...], np.ndarray, dict[str, tuple[str, ...]]]:
    """Teams, leveling factors, and divisions — cross-checked against each other."""
    rules_path = sdir / "rules.yaml"
    try:
        lf_map = {
            team_code(k, f"{rules_path} leveling_factors"): v
            for k, v in rules["leveling_factors"].items()
        }
        divisions = {
            d: tuple(team_code(t, f"{rules_path} divisions.{d}") for t in ts)
            for d, ts in rules["divisions"].items()
        }
    except KeyError as e:
        raise ConfigError(f"{rules_path} is missing section {e}") from e

    teams = tuple(sorted(lf_map))
    all_division_slots = [t for ts in divisions.values() for t in ts]
    in_divisions = set(all_division_slots)

    # Checked before the set comparison: a team listed twice always shows up as
    # a set mismatch too, and "KC is in two divisions" is the message that
    # actually names the mistake.
    dupes = sorted({t for t in all_division_slots if all_division_slots.count(t) > 1})
    if dupes:
        raise ConfigError(f"team(s) listed in more than one division: {dupes}")

    if set(teams) != in_divisions:
        only_lf = sorted(set(teams) - in_divisions)
        only_div = sorted(in_divisions - set(teams))
        raise ConfigError(
            f"{sdir / 'rules.yaml'}: leveling_factors and divisions disagree. "
            f"Only in leveling_factors: {only_lf or 'none'}. "
            f"Only in divisions: {only_div or 'none'}."
        )
    if len(teams) != 32:
        raise ConfigError(f"expected 32 teams in {rules_path}, got {len(teams)}")

    lf = np.array([float(lf_map[t]) for t in teams])
    if np.any(lf <= 0):
        bad = [t for t, v in zip(teams, lf) if v <= 0]
        raise ConfigError(f"leveling factors must be positive; bad: {bad}")
    return teams, lf, divisions


def _parse_scoring(rules: dict, sdir: Path) -> tuple[Bonuses, float, float]:
    try:
        sc = rules["scoring"]
    except KeyError as e:
        raise ConfigError(f"{sdir / 'rules.yaml'} is missing section {e}") from e

    def stage(key: str) -> tuple[float, bool]:
        try:
            d = sc[key]
            return float(d["flat"]), bool(d["add_lf"])
        except (KeyError, TypeError) as e:
            raise ConfigError(
                f"{sdir / 'rules.yaml'}: scoring.{key} must be "
                f"{{flat: <number>, add_lf: <bool>}}"
            ) from e

    wc_flat, wc_lf = stage("wild_card_upset")
    dv_flat, dv_lf = stage("divisional_win")
    cf_flat, cf_lf = stage("conference_win")
    sb_flat, sb_lf = stage("super_bowl_win")

    def flat(key: str) -> float:
        # The staged bonuses already fail with a legible ConfigError; a missing
        # flat bonus used to leak a raw KeyError traceback instead. Same
        # loudness for both.
        try:
            return float(sc[key])
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigError(
                f"{sdir / 'rules.yaml'}: scoring.{key} must be a number"
            ) from e

    bonuses = Bonuses(
        division_winner=flat("division_winner"),
        wild_card_berth=flat("wild_card_berth"),
        wild_card_upset_flat=wc_flat,
        wild_card_upset_add_lf=wc_lf,
        divisional_flat=dv_flat,
        divisional_add_lf=dv_lf,
        conference_flat=cf_flat,
        conference_add_lf=cf_lf,
        super_bowl_flat=sb_flat,
        super_bowl_add_lf=sb_lf,
    )
    return bonuses, float(sc.get("win_multiplier", 1.0)), float(sc.get("tie_multiplier", 0.5))


def _validate_money(rules: dict, sdir: Path) -> None:
    """The dollars get the same hard validation the picks do.

    A payout_split summing past 1.0 pays out more than the pot; a negative
    entry fee produces negative payouts. Both previously loaded silently and
    would have been published as-is.
    """
    path = sdir / "rules.yaml"
    try:
        fee = float(rules["entry_fee"])
        split = [float(x) for x in rules["payout_split"]]
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(
            f"{path}: entry_fee must be a number and payout_split a list of numbers"
        ) from e
    if fee < 0:
        raise ConfigError(f"{path}: entry_fee is negative ({fee})")
    if not split:
        raise ConfigError(f"{path}: payout_split is empty — nobody gets paid")
    if any(s < 0 for s in split):
        raise ConfigError(f"{path}: payout_split has a negative share: {split}")
    if sum(split) > 1.0 + 1e-9:
        raise ConfigError(
            f"{path}: payout_split sums to {sum(split):.2f} — that pays out "
            f"more than 100% of the pot"
        )


def _parse_picks(
    picks: dict, valid: set[str], rules: dict, sdir: Path
) -> tuple[Entrant, ...]:
    """Validate and load entrants. Every failure mode here is loud."""
    path = sdir / "picks.yaml"
    raw = picks.get("entrants")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}: needs a non-empty `entrants:` list")

    n_required = int(rules.get("picks_per_entrant", 4))
    entrants: list[Entrant] = []
    seen_names: set[str] = set()
    seen_slugs: dict[str, str] = {}

    for i, row in enumerate(raw):
        where = f"{path} entry #{i + 1}"
        if not isinstance(row, dict) or "name" not in row or "teams" not in row:
            raise ConfigError(f"{where}: needs both `name:` and `teams:`")

        name = str(row["name"]).strip()
        if not name:
            raise ConfigError(f"{where}: name is empty")
        if name.lower() in seen_names:
            raise ConfigError(f"{path}: duplicate entrant name {name!r}")
        seen_names.add(name.lower())

        teams_raw = row["teams"]
        if not isinstance(teams_raw, Sequence) or isinstance(teams_raw, str):
            raise ConfigError(f"{where} ({name}): `teams:` must be a list")
        teams = [team_code(t, f"{where} ({name})") for t in teams_raw]

        placeholders = [t for t in teams if t in {"TODO", "TBD", "???", ""}]
        if placeholders:
            raise ConfigError(
                f"{path}: {name} still has placeholder picks {placeholders}. "
                f'Replace them with {n_required} real team codes, quoted '
                f'(e.g. teams: ["CIN", "WAS", "TEN", "NO"] — a bare NO is '
                f"YAML for false)."
            )
        if len(teams) != n_required:
            raise ConfigError(
                f"{where} ({name}): has {len(teams)} teams, needs exactly {n_required}"
            )
        if len(set(teams)) != len(teams):
            dupe = sorted({t for t in teams if teams.count(t) > 1})
            raise ConfigError(f"{where} ({name}): duplicate team(s) {dupe} in one entry")

        unknown = [t for t in teams if t not in valid]
        if unknown:
            raise ConfigError(
                f"{where} ({name}): unknown team code(s) {unknown}. "
                f"Valid codes are the 32 in rules.yaml — note LAR (not LA), "
                f"KC (not KAN), WAS (not WSH), JAX (not JAC)."
            )

        e = Entrant(name=name, teams=tuple(teams))
        if e.slug in seen_slugs:
            raise ConfigError(
                f"{path}: {name!r} and {seen_slugs[e.slug]!r} both produce the URL "
                f"slug {e.slug!r}; make the names more distinct."
            )
        seen_slugs[e.slug] = name
        entrants.append(e)

    return tuple(entrants)
