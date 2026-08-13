"""Site generation.

Structural assertions, not golden HTML files. Eighty-odd golden pages would
break on every CSS class rename, and a suite that gets regenerated without
being read is worse than no suite at all.

The base-path tests matter most: a root-absolute asset URL works perfectly on a
local preview server and 404s only once deployed to a project page.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from football_pool.history import weekly_frame
from football_pool.nflverse import GameData, parse_games
from football_pool.render import ASSET_DIR, build_context, make_environment, render_site

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def game_data(games_2025):
    return GameData(
        games_2025,
        2025,
        datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 2, 9, 14, 30, tzinfo=timezone.utc),
        "network",
    )


@pytest.fixture
def mid_season(games_2025):
    """Week 11 of 18 — the state the site spends most of its life in."""
    g = games_2025.copy()
    g.loc[g["week"] > 11, "played"] = False
    g.loc[g["game_type"] != "REG", "played"] = False
    return GameData(g, 2025, datetime.now(timezone.utc), None, "cache")


@pytest.fixture
def pool(make_season):
    return make_season(
        [
            {"name": "Aunt Carol", "teams": ["SEA", "NE", "PHI", "LAR"]},
            {"name": "Cousin Mike", "teams": ["KC", "CIN", "DAL", "NO"]},
            {"name": "Brandon", "teams": ["ARI", "NYJ", "TEN", "CLE"]},
        ],
        year=2025,
    )


# -- the url filter, the deployment footgun ---------------------------------
# Non-asset paths only: assets additionally carry a cache-busting fingerprint,
# which is asserted separately below so that an edit to site.css cannot turn
# this test into a tripwire on an unrelated commit.
@pytest.mark.parametrize(
    "base, path, expected",
    [
        ("", "/rules/", "/rules/"),
        ("/football-pool", "/rules/", "/football-pool/rules/"),
        ("/football-pool/", "/rules/", "/football-pool/rules/"),
        ("/football-pool", "/", "/football-pool/"),
        ("/football-pool", "/data/standings.json", "/football-pool/data/standings.json"),
    ],
)
def test_url_filter_applies_the_deployment_base(base, path, expected):
    assert make_environment(base).filters["url"](path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "https://example.com/x",
        "http://example.com/x",
        "#scoring",
        "mailto:a@b.c",
        # An external URL that happens to contain /assets/ must not be
        # fingerprinted: the passthrough has to run before the asset check.
        "https://cdn.example.com/assets/site.css",
    ],
)
def test_url_filter_leaves_absolute_and_anchor_links_alone(path):
    """Rewriting '#scoring' would turn an in-page jump into a navigation."""
    assert make_environment("/football-pool").filters["url"](path) == path


# -- asset fingerprinting, the cache footgun --------------------------------
# GitHub Pages pins every response to max-age=600 and offers no way to change
# it, so for ten minutes after a deploy a returning visitor can hold the old
# stylesheet against new markup. That skew looks exactly like a layout bug and
# was once reported as one.
def expected_digest(name: str) -> str:
    """Recompute the fingerprint the filter should produce for an asset.

    Derived rather than hardcoded on purpose: a hardcoded digest would have to
    be updated on every unrelated CSS edit, and a test people routinely
    regenerate without reading stops being a test.
    """
    return hashlib.sha256((ASSET_DIR / name).read_bytes()).hexdigest()[:8]


@pytest.mark.parametrize("name", ["site.css", "site.js", "favicon.svg"])
def test_every_asset_carries_its_content_hash(name):
    url = make_environment("").filters["url"](f"/assets/{name}")
    assert url == f"/assets/{name}?v={expected_digest(name)}"


def test_the_fingerprint_composes_with_the_deployment_base():
    """The stamp goes after the path, never before the prefix."""
    url = make_environment("/football-pool/").filters["url"]("/assets/site.css")
    assert url == f"/football-pool/assets/site.css?v={expected_digest('site.css')}"


def test_only_assets_are_fingerprinted():
    """Page URLs get pasted into the family group chat; keep them clean."""
    url = make_environment("").filters["url"]
    for path in ("/", "/rules/", "/entrant/brandon/", "/data/standings.json"):
        assert "?v=" not in url(path)


@pytest.mark.parametrize(
    "path",
    [
        "/assets/nope.css",  # no such file
        "/assets/fonts/",  # a directory, not a file
        "/assets/",  # the directory itself
    ],
)
def test_an_unhashable_asset_degrades_to_a_plain_url(path):
    """A deleted favicon must not take the whole scoreboard down with it."""
    url = make_environment("").filters["url"](path)
    assert url == path
    assert "?v=" not in url


def test_a_changed_asset_changes_the_url(tmp_path, monkeypatch):
    """The actual cache-busting behaviour, not merely that a hash exists."""
    assets = tmp_path / "assets"
    shutil.copytree(ASSET_DIR, assets)
    monkeypatch.setattr("football_pool.render.ASSET_DIR", assets)

    before = make_environment("").filters["url"]("/assets/site.css")

    (assets / "site.css").write_text((assets / "site.css").read_text() + "\n/* x */")
    # A fresh environment, because the digest cache is per build by design.
    after = make_environment("").filters["url"]("/assets/site.css")

    assert before != after
    assert before.split("?v=")[0] == after.split("?v=")[0]


def test_an_unchanged_asset_keeps_its_url():
    """The guard against anyone reaching for a timestamp later."""
    first = make_environment("").filters["url"]("/assets/site.css")
    second = make_environment("").filters["url"]("/assets/site.css")
    assert first == second


def test_no_root_absolute_urls_survive_a_based_build(pool, game_data, tmp_path):
    """The failure that only ever shows up in production."""
    render_site(pool, game_data, tmp_path, base="/football-pool")

    for page in tmp_path.rglob("*.html"):
        html = page.read_text()
        for attr, value in re.findall(r'(href|src)="(/[^"]*)"', html):
            assert value.startswith("/football-pool/"), (
                f"{page.name}: {attr}={value} is root-absolute and will 404 "
                f"on a project page"
            )


def test_base_tag_is_never_used(pool, game_data, tmp_path):
    """`<base href>` would silently break every in-page anchor."""
    render_site(pool, game_data, tmp_path, base="/football-pool")
    assert "<base " not in (tmp_path / "index.html").read_text()


# -- pages -------------------------------------------------------------------
def test_builds_every_expected_page(pool, game_data, tmp_path):
    render_site(pool, game_data, tmp_path)

    for rel in (
        "index.html",
        "rules/index.html",
        "forecast/index.html",
        "teams/index.html",
        "weeks/index.html",
        "season/index.html",
        "404.html",
        "entrant/aunt-carol/index.html",
        "entrant/cousin-mike/index.html",
        "entrant/brandon/index.html",
        "data/standings.json",
    ):
        assert (tmp_path / rel).exists(), rel


def test_assets_are_copied(pool, game_data, tmp_path):
    render_site(pool, game_data, tmp_path)
    assert (tmp_path / "assets" / "site.css").exists()
    assert (tmp_path / "assets" / "site.js").exists()


def test_leaderboard_is_ordered_and_linked(pool, game_data, tmp_path):
    render_site(pool, game_data, tmp_path)
    html = (tmp_path / "index.html").read_text()

    order = [m for m in re.findall(r'href="/entrant/([a-z-]+)/"', html)]
    assert order[0] == "aunt-carol"  # highest scorer in 2025
    assert len(order) == 3


def test_every_entrant_page_is_reachable_and_names_its_teams(pool, game_data, tmp_path):
    render_site(pool, game_data, tmp_path)
    page = (tmp_path / "entrant" / "brandon" / "index.html").read_text()
    assert "Brandon" in page
    for team in ("ARI", "NYJ", "TEN", "CLE"):
        assert team in page


def test_rules_page_renders_from_the_rules_file(pool, game_data, tmp_path):
    """Posted rules and scoring maths come from one source, so they can't drift."""
    render_site(pool, game_data, tmp_path)
    html = (tmp_path / "rules" / "index.html").read_text()
    assert "+3.0" in html  # division winner
    assert "+1.5" in html  # wild card
    assert "0.5 × LF" in html  # the tie policy from rules.yaml
    assert "2.60" in html  # ARI's leveling factor


def test_freshness_stamps_are_machine_readable(pool, game_data, tmp_path):
    """Rendered as UTC in a datetime attribute so JS can localise them."""
    render_site(pool, game_data, tmp_path)
    html = (tmp_path / "index.html").read_text()
    stamps = re.findall(r'<time datetime="([^"]+)" data-ts>', html)
    assert len(stamps) == 2  # data-through and site-built
    for iso in stamps:
        datetime.fromisoformat(iso)


def test_timezone_picker_defaults_to_eastern(pool, game_data, tmp_path):
    render_site(pool, game_data, tmp_path)
    html = (tmp_path / "index.html").read_text()
    first_option = re.search(r"<select data-tz-select>\s*<option value=\"([^\"]+)\"", html)
    assert first_option.group(1) == "America/New_York"


def test_theme_toggle_is_present(pool, game_data, tmp_path):
    render_site(pool, game_data, tmp_path)
    assert "data-theme-toggle" in (tmp_path / "index.html").read_text()


# -- preseason, the state the site launches in ------------------------------
def test_preseason_builds_with_no_games_played(pool, games_2025, tmp_path):
    empty = games_2025.copy()
    empty[["played", "home_won", "away_won", "is_tie"]] = False
    data = GameData(empty, 2025, datetime.now(timezone.utc), None, "cache")

    render_site(pool, data, tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert "Preseason" in html
    assert "Nothing has kicked off yet" in html
    assert "Nothing has been played yet" in (tmp_path / "season" / "index.html").read_text()


def test_weeks_and_trends_forward_to_season(pool, game_data, tmp_path):
    """A year of group-chat links points at the old pages; they must forward."""
    render_site(pool, game_data, tmp_path)
    for rel in ("weeks", "trends"):
        html = (tmp_path / rel / "index.html").read_text()
        assert 'url=/season/' in html


def test_a_pool_with_a_single_entrant_builds(make_season, game_data, tmp_path):
    solo = make_season([{"name": "Solo", "teams": ["KC", "SEA", "DAL", "NE"]}], year=2025)
    render_site(solo, game_data, tmp_path)
    assert "Solo" in (tmp_path / "index.html").read_text()


# -- mid-season -------------------------------------------------------------
def test_mid_season_shows_the_current_week(pool, mid_season, tmp_path):
    render_site(pool, mid_season, tmp_path)
    assert "Week 11" in (tmp_path / "index.html").read_text()


def test_weeks_page_has_a_column_per_played_week(pool, mid_season, tmp_path):
    render_site(pool, mid_season, tmp_path)
    html = (tmp_path / "season" / "index.html").read_text()
    header = re.search(r"<thead>.*?</thead>", html, re.S).group(0)
    assert len(re.findall(r'class="num">\d+</th>', header)) == 11


# -- the json sidecar -------------------------------------------------------
def test_standings_json_is_valid_and_complete(pool, game_data, tmp_path):
    render_site(pool, game_data, tmp_path)
    payload = json.loads((tmp_path / "data" / "standings.json").read_text())

    assert payload["season"] == 2025
    assert len(payload["entrants"]) == 3
    assert payload["entrants"][0]["rank"] == 1
    # Rendered markup does not belong in the data file.
    assert "spark" not in payload["entrants"][0]


def test_context_totals_agree_with_the_rendered_page(pool, game_data, tmp_path):
    ctx = build_context(pool, game_data)
    render_site(pool, game_data, tmp_path)
    payload = json.loads((tmp_path / "data" / "standings.json").read_text())

    for row in payload["entrants"]:
        expected = float(ctx.outlook.loc[ctx.outlook["name"] == row["name"], "banked"].iloc[0])
        assert row["banked"] == pytest.approx(expected)


# -- rebuild stability ------------------------------------------------------
def test_rebuilding_is_deterministic(pool, mid_season, tmp_path):
    """A day with no new games must not produce a diff."""
    a, b = tmp_path / "a", tmp_path / "b"
    render_site(pool, mid_season, a)
    render_site(pool, mid_season, b)

    for page in sorted(a.rglob("*.html")):
        other = b / page.relative_to(a)
        # The build timestamp is the only thing allowed to differ.
        strip = lambda s: re.sub(r'<time datetime="[^"]+" data-ts>[^<]+</time>', "", s)
        assert strip(page.read_text()) == strip(other.read_text()), page.name


def test_playoff_phase_is_labelled(pool, games_2025, tmp_path):
    """Regular season done, bracket still running."""
    g = games_2025.copy()
    g.loc[g["game_type"].isin({"DIV", "CON", "SB"}), "played"] = False
    data = GameData(g, 2025, datetime.now(timezone.utc), None, "cache")

    render_site(pool, data, tmp_path)
    assert "Playoffs" in (tmp_path / "index.html").read_text()


def test_final_phase_is_labelled(pool, game_data, tmp_path):
    render_site(pool, game_data, tmp_path)
    assert "Final" in (tmp_path / "index.html").read_text()


def test_money_filter_hides_zero_and_missing(pool):
    """An entrant out of the money shows nothing, not "$0"."""
    money = make_environment().filters["money"]
    assert money(0) == ""
    assert money(None) == ""
    assert money(37.5) == "$38"


def test_points_filter_pads_and_handles_missing(pool):
    points = make_environment().filters["points"]
    assert points(3.5) == "3.50"
    assert points(None) == "—"


def test_assets_missing_does_not_break_the_build(pool, game_data, tmp_path, monkeypatch):
    """Rendering must still work if the asset directory is absent."""
    out = tmp_path / "out"
    monkeypatch.setattr("football_pool.render.ASSET_DIR", tmp_path / "nope")
    written = render_site(pool, game_data, out)

    assert (out / "index.html").exists()
    assert not (out / "assets").exists()
    assert not any(p.relative_to(out).parts[0] == "assets" for p in written)


def test_a_removed_entrant_stops_being_published(make_season, game_data, tmp_path):
    """Their page must not stay live and reachable after they leave the pool."""
    before = make_season(
        [
            {"name": "Stays", "teams": ["KC", "SEA", "DAL", "NE"]},
            {"name": "Leaves", "teams": ["ARI", "BUF", "GB", "MIA"]},
        ],
        year=2025,
    )
    render_site(before, game_data, tmp_path)
    assert (tmp_path / "entrant" / "leaves" / "index.html").exists()

    after = make_season([{"name": "Stays", "teams": ["KC", "SEA", "DAL", "NE"]}], year=2025)
    render_site(after, game_data, tmp_path)
    assert (tmp_path / "entrant" / "stays" / "index.html").exists()
    assert not (tmp_path / "entrant" / "leaves").exists()


def test_a_broken_model_does_not_take_the_site_down(pool, mid_season, tmp_path, monkeypatch):
    """Actual standings are the point; a projection failure must not cost them."""
    def boom(*a, **kw):
        raise RuntimeError("simulation exploded")

    monkeypatch.setattr("football_pool.render.project", boom)
    with pytest.warns(RuntimeWarning, match="projections unavailable"):
        render_site(pool, mid_season, tmp_path)

    html = (tmp_path / "index.html").read_text()
    assert "Standings" in html
    assert "Projected finish" not in html


def test_projections_appear_when_available(pool, mid_season, tmp_path):
    """A summary on the standings page, the full table on /forecast/.

    The standings page is a scoreboard — what has happened — so it carries the
    headline only, and the distributional detail lives on its own page.
    """
    render_site(pool, mid_season, tmp_path, simulations=200)
    index = (tmp_path / "index.html").read_text()
    assert "Projected finish" in index
    assert "modelled" in index

    forecast = (tmp_path / "forecast" / "index.html").read_text()
    assert "P(1st)" in forecast
    assert "P(cash)" in forecast


def test_projections_are_absent_once_the_season_ends(pool, game_data, tmp_path):
    """Nothing left to simulate, so the panels are dropped rather than faked."""
    render_site(pool, game_data, tmp_path)
    assert "Projected finish" not in (tmp_path / "index.html").read_text()


def test_an_entrant_missing_from_the_model_is_tolerated(pool, mid_season, monkeypatch):
    """A name mismatch must yield no projection, not a crash."""
    from football_pool import render as render_mod

    ctx = render_mod.build_context(pool, mid_season, simulations=200)
    assert render_mod._projection_for(ctx, "Nobody At All") is None


def test_history_and_render_agree_on_week_count(pool, mid_season):
    ctx = build_context(pool, mid_season)
    assert ctx.history.attrs["weeks"] == list(range(1, 12))
    assert weekly_frame(pool, mid_season.games).attrs["weeks"] == ctx.history.attrs["weeks"]


# -- theme, identity and the morph -------------------------------------------
def test_the_markup_itself_is_dark(season, game_data, tmp_path):
    """Dark is the default in the document, not merely in the stylesheet.

    Anything that reads the page before CSS lands — or with scripting off —
    must already be on the dark theme rather than flashing light first.
    """
    render_site(season, game_data, tmp_path)
    assert 'data-theme="dark"' in (tmp_path / "index.html").read_text()


def test_the_row_and_the_hero_share_a_transition_name(season, game_data, tmp_path):
    """This pairing is what makes clicking a row morph instead of cross-fade.

    The names live in two separate templates, so nothing but a test notices
    when one of them is renamed and the effect silently degrades.
    """
    render_site(season, game_data, tmp_path)
    index = (tmp_path / "index.html").read_text()

    for entrant in season.entrants:
        slug = _slug_of(tmp_path, entrant.name)
        page = (tmp_path / "entrant" / slug / "index.html").read_text()
        for name in (f"name-{slug}", f"total-{slug}"):
            assert f"view-transition-name: {name}" in index, f"{name} missing from the board"
            assert f"view-transition-name: {name}" in page, f"{name} missing from the hero"


def test_transition_names_are_unique_within_a_page(season, game_data, tmp_path):
    """A duplicate name makes the browser drop the transition for both elements."""
    render_site(season, game_data, tmp_path)
    names = re.findall(r"view-transition-name: ([\w-]+)", (tmp_path / "index.html").read_text())
    assert len(names) == len(set(names))


# -- the link preview ---------------------------------------------------------
def test_the_share_card_is_generated_and_copied_out(season, game_data, tmp_path):
    render_site(season, game_data, tmp_path)
    assert (tmp_path / "assets" / "card.png").exists()


def test_og_image_is_absolute_and_only_present_with_a_site_url(season, game_data, tmp_path):
    """Link scrapers do not resolve relative URLs, so a preview without an
    origin would be worse than none. The tag also carries the card's content
    fingerprint, so a changed leaderboard is an image no chat app has cached."""
    render_site(season, game_data, tmp_path, site_url="https://example.github.io/")
    html = (tmp_path / "index.html").read_text()
    m = re.search(r'property="og:image" content="([^"]+)"', html)
    assert m, "og:image missing despite a site_url"
    assert m.group(1).startswith("https://example.github.io/assets/card.png?v=")

    render_site(season, game_data, tmp_path)
    assert 'og:image' not in (tmp_path / "index.html").read_text()


def _real_pages(written):
    """Every rendered page, minus the chrome-less forwarding stubs."""
    for page in written:
        if page.suffix != ".html":
            continue
        text = page.read_text()
        if 'http-equiv="refresh"' in text:
            continue
        yield page, text


def test_every_page_offers_the_identity_picker(season, game_data, tmp_path):
    written = render_site(season, game_data, tmp_path)
    for page, text in _real_pages(written):
        assert "data-me-select" in text, page


def test_rows_carry_the_slug_the_picker_matches_on(season, game_data, tmp_path):
    """The picker's option values and the rows' data-slug must be the same set."""
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "index.html").read_text()

    row_slugs = set(re.findall(r'class="row[^"]*"\s+style="[^"]*"\s+data-slug="([\w-]+)"', html))
    # Scope to the identity picker: the time zone control is a <select> too, and
    # matching every <option> on the page would sweep up "UTC".
    picker = re.search(r"<select data-me-select>(.*?)</select>", html, re.S).group(1)
    option_slugs = set(re.findall(r'<option value="([\w-]+)">', picker))
    assert row_slugs
    assert row_slugs == option_slugs


# -- team colours -------------------------------------------------------------
def test_team_chips_wear_their_club_colours(season, game_data, tmp_path):
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert "--team-bg:" in html and "--team-fg:" in html and "--team-edge:" in html


def test_every_chip_that_names_a_team_is_coloured(season, game_data, tmp_path):
    """A chip with no colour block would render as an unexplained grey outlier."""
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "teams" / "index.html").read_text()

    chips = re.findall(
        r'<span class="team-chip[^"]*"\s*\n?\s*style="([^"]*)">(?:<img[^>]*>)?([A-Z]{2,3})</span>',
        html,
    )
    assert len(chips) >= 32
    for style, team in chips:
        assert "--team-bg:" in style, f"{team} chip has no colour"


# -- sortable tables ----------------------------------------------------------
def test_the_teams_table_is_sortable_with_real_sort_keys(season, game_data, tmp_path):
    """Every sortable column needs a key that sorts correctly as a number.

    Displayed text would sort "10-2" before "3-1" and put unseeded teams above
    the top seed, so the sort keys are emitted separately.
    """
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "teams" / "index.html").read_text()

    assert html.count("data-sort") == 6
    assert 'class="table-scroll is-tall"' in html
    # An unseeded team sorts to the bottom rather than sorting as empty.
    assert 'data-value="99"' in html


def test_the_seed_sort_key_orders_playoff_teams_first(season, game_data, tmp_path):
    """Seeds 1..7 must come before the 99 that stands in for "did not qualify"."""
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "teams" / "index.html").read_text()
    seeds = [int(v) for v in re.findall(r'<td data-value="(\d+)">\s*\n?\s*(?:<span class="badge money">|—)', html)]
    assert sorted(s for s in seeds if s != 99)[:1] == [1]
    assert 99 in seeds


# -- the comparison chart -----------------------------------------------------
def test_the_compare_chart_offers_every_entrant(season, game_data, tmp_path):
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "season" / "index.html").read_text()

    picks = set(re.findall(r'data-pick="([\w-]+)"', html))
    lines = set(re.findall(r'<polyline class="cmp-line" data-entrant="([\w-]+)"', html))
    assert picks
    assert picks == lines, "a name you can pick must have a line to light up"


def test_the_compare_chart_ships_unpicked(season, game_data, tmp_path):
    """The server does not choose for the viewer; the browser does."""
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "season" / "index.html").read_text()
    assert "is-picked" not in html
    assert html.count("data-compare>") == 2  # points and rank


def test_the_compare_chart_is_absent_before_any_games(season, tmp_path, games_2025):
    """Nothing to compare in the preseason, so it must not render an empty frame."""
    g = games_2025.copy()
    g[["played", "home_won", "away_won", "is_tie"]] = False
    preseason = GameData(g, 2025, datetime.now(timezone.utc), None, "cache")

    render_site(season, preseason, tmp_path)
    html = (tmp_path / "season" / "index.html").read_text()
    assert "data-pick=" not in html


def _slug_of(out_dir: Path, name: str) -> str:
    """Find the slug the site actually generated for an entrant."""
    data = json.loads((out_dir / "data" / "standings.json").read_text())
    return next(e["slug"] for e in data["entrants"] if e["name"] == name)


# -- preseason honesty --------------------------------------------------------
def test_no_playoff_seeds_are_shown_before_any_game(season, tmp_path, games_2025):
    """The Teams page must not claim a playoff field in the preseason.

    Every club is 0-0, so the tiebreaker ladder exhausted and fell through to
    its alphabetical coin-toss fallback — and the page printed the result as
    fourteen seeded teams, with the one seeds going to whoever came first in
    the alphabet. This is the page-level guard for that.
    """
    g = games_2025.copy()
    g[["played", "home_won", "away_won", "is_tie"]] = False
    preseason = GameData(g, 2025, datetime.now(timezone.utc), None, "cache")

    render_site(season, preseason, tmp_path)
    html = (tmp_path / "teams" / "index.html").read_text()

    seeded = re.findall(r'<span class="badge money">(\d)</span>', html)
    assert seeded == [], f"{len(seeded)} teams shown as seeded before kickoff"
    # All 32 rows sort to the bottom of the seed column instead.
    assert html.count('<td data-value="99">') == 32


def test_entrant_cards_omit_the_seed_before_any_game(season, tmp_path, games_2025):
    g = games_2025.copy()
    g[["played", "home_won", "away_won", "is_tie"]] = False
    preseason = GameData(g, 2025, datetime.now(timezone.utc), None, "cache")

    written = render_site(season, preseason, tmp_path)
    for page in (p for p in written if p.parent.parent.name == "entrant"):
        assert "seed " not in page.read_text()


def test_seeds_return_once_the_season_is_under_way(season, mid_season, tmp_path):
    """The projection is a feature — this must not have thrown it away."""
    render_site(season, mid_season, tmp_path)
    html = (tmp_path / "teams" / "index.html").read_text()
    assert len(re.findall(r'<span class="badge money">(\d)</span>', html)) == 14


# -- fingerprints in the built site -------------------------------------------
ASSET_URL_RE = re.compile(r'(?:href|src)="([^"]*/assets/[^"]*)"')


def test_every_asset_url_on_every_page_is_fingerprinted(season, game_data, tmp_path):
    """Guards assets that do not exist yet.

    The point of doing this inside the one filter every URL already passes
    through is that a new asset cannot forget to opt in — this asserts it.
    """
    render_site(season, game_data, tmp_path, base="/football-pool")

    seen = 0
    for page in tmp_path.rglob("*.html"):
        for url in ASSET_URL_RE.findall(page.read_text()):
            seen += 1
            assert re.search(r"\?v=[0-9a-f]{8}$", url), f"{page.name}: {url}"
    assert seen >= 3, "expected at least the stylesheet, the script and the favicon"


def test_the_fingerprint_matches_the_bytes_that_shipped(season, game_data, tmp_path):
    """The URL the browser asks for must name the file that was published.

    This is the real invariant. It is also what would catch a minifier being
    slipped in between hashing the source and copying it to the output.
    """
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "index.html").read_text()

    checked = 0
    for url in ASSET_URL_RE.findall(html):
        path, _, stamp = url.partition("?v=")
        shipped = tmp_path / path.lstrip("/")
        assert shipped.is_file(), f"{url} does not exist in the output"
        assert stamp == hashlib.sha256(shipped.read_bytes()).hexdigest()[:8]
        checked += 1
    assert checked >= 3


def test_page_links_are_never_fingerprinted(season, game_data, tmp_path):
    """A stamp on /entrant/x/ would break every in-page link regex, and look daft."""
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "index.html").read_text()

    links = [u for u in re.findall(r'href="(/[^"]*)"', html) if "/assets/" not in u]
    assert links, "expected at least the nav links"
    for url in links:
        assert "?v=" not in url, url


def test_the_stylesheet_still_finds_its_fonts(season, game_data, tmp_path):
    """A relative url() inside the CSS resolves against the path, not the query.

    RFC 3986 §5.3: a non-empty relative reference does not inherit the base's
    query, so `fonts/x.woff2` resolves correctly under `site.css?v=...`. This is
    also the regression guard against anyone rewriting the CSS to rename files.
    """
    render_site(season, game_data, tmp_path)
    css = (tmp_path / "assets" / "site.css").read_text()

    assert "url('fonts/big-shoulders.woff2')" in css
    assert sorted(p.name for p in (tmp_path / "assets" / "fonts").glob("*.woff2"))


def test_the_ci_contract_holds(season, game_data, tmp_path):
    """Mirrors the greps in ci.yml so this breaks in pytest, not a required check."""
    root = tmp_path / "root"
    project = tmp_path / "project"
    render_site(season, game_data, root, base="")
    render_site(season, game_data, project, base="/football-pool")

    assert re.search(
        r'href="/assets/site\.css\?v=[0-9a-f]{8}"', (root / "index.html").read_text()
    )
    assert re.search(
        r'href="/football-pool/assets/site\.css\?v=[0-9a-f]{8}"',
        (project / "index.html").read_text(),
    )


# -- viewer preferences sit at the top ----------------------------------------
def test_the_viewer_controls_come_before_the_content(season, game_data, tmp_path):
    """Both settings change what you are looking at, so they precede it.

    They used to live in the footer, where nobody scrolled to find them. This
    pins the position rather than merely their existence — the previous test
    passed perfectly well while they were invisible at the bottom of the page.
    """
    written = render_site(season, game_data, tmp_path)

    for page, html in _real_pages(written):
        main_at = html.index("<main")
        assert html.index("data-me-select") < main_at, page
        assert html.index("data-tz-select") < main_at, page
        assert html.index('class="viewer-bar"') < main_at, page


def test_the_controls_appear_exactly_once(season, game_data, tmp_path):
    """A duplicate would silently break the wiring.

    initMe and initTimeZone both use querySelector, so a second copy left
    behind in the footer would never be wired up and would sit there showing
    a stale value.
    """
    render_site(season, game_data, tmp_path)
    html = (tmp_path / "index.html").read_text()

    assert html.count("data-me-select") == 1
    assert html.count("data-tz-select") == 1


def test_the_footer_keeps_the_freshness_stamps(season, game_data, tmp_path):
    """Moving the controls out must not take the timestamps with them."""
    render_site(season, game_data, tmp_path)
    footer = (tmp_path / "index.html").read_text().split("<footer")[1]

    assert "Data through" in footer
    assert "Site built" in footer
    assert len(re.findall(r"<time datetime=", footer)) == 2


# -- the forecast page --------------------------------------------------------
def test_the_forecast_page_carries_all_three_charts(pool, mid_season, tmp_path):
    render_site(pool, mid_season, tmp_path, simulations=200)
    html = (tmp_path / "forecast" / "index.html").read_text()

    assert 'class="finish"' in html  # finishing-place bars
    assert 'class="ridge"' in html  # score distributions
    assert 'class="heat"' in html  # head to head
    assert "modelled" in html


def test_the_forecast_page_has_a_row_per_entrant(pool, mid_season, tmp_path):
    render_site(pool, mid_season, tmp_path, simulations=200)
    html = (tmp_path / "forecast" / "index.html").read_text()

    slugs = re.findall(r'class="forecast-row"[^>]*data-slug="([\w-]+)"', html)
    assert len(slugs) == len(pool.entrants)
    assert len(set(slugs)) == len(slugs)


def test_the_forecast_is_ordered_by_chance_of_winning(pool, mid_season, tmp_path):
    """The model's own ranking, not the current standings — they differ."""
    ctx = build_context(pool, mid_season, simulations=200)
    render_site(pool, mid_season, tmp_path, simulations=200)
    html = (tmp_path / "forecast" / "index.html").read_text()

    shown = re.findall(r'class="forecast-row"[^>]*data-slug="([\w-]+)"', html)
    expected = (
        ctx.projections.entrants.sort_values("p_first", ascending=False)["slug"].tolist()
    )
    assert shown == expected


def test_the_forecast_tab_is_in_the_nav_on_every_page(pool, mid_season, tmp_path):
    written = render_site(pool, mid_season, tmp_path, simulations=200)
    for page, text in _real_pages(written):
        assert "/forecast/" in text, page


def test_the_forecast_page_explains_itself_when_there_is_nothing_to_say(
    pool, game_data, tmp_path
):
    """Once the bracket starts there is nothing left to simulate."""
    render_site(pool, game_data, tmp_path)
    html = (tmp_path / "forecast" / "index.html").read_text()

    assert 'class="finish"' not in html
    assert "The season is over" in html


def test_the_standings_page_links_to_the_full_forecast(pool, mid_season, tmp_path):
    """The scoreboard keeps a headline; the distributions live on their own page."""
    render_site(pool, mid_season, tmp_path, simulations=200)
    index = (tmp_path / "index.html").read_text()

    assert 'class="finish"' in index  # the summary bars
    assert 'class="ridge"' not in index  # but not the full picture
    assert 'href="/forecast/"' in index


def test_an_entrant_page_shows_their_own_odds(pool, mid_season, tmp_path):
    render_site(pool, mid_season, tmp_path, simulations=200)
    html = (tmp_path / "entrant" / "brandon" / "index.html").read_text()

    assert 'class="finish"' in html
    assert "Favoured against" in html
    # Every rival but themselves.
    assert html.count("<tbody>") >= 1
    for other in ("Aunt Carol", "Cousin Mike"):
        assert other in html


def test_short_names_fall_back_when_first_names_collide():
    """Two Sarahs must not both become "Sarah" on a chart axis."""
    from football_pool.render import _short_names

    assert _short_names(["Brian Moore", "Paul Moore"]) == {
        "Brian Moore": "Brian",
        "Paul Moore": "Paul",
    }
    clash = _short_names(["Sarah Jones", "Sarah Smith"])
    assert clash == {"Sarah Jones": "Sarah Jones", "Sarah Smith": "Sarah Smith"}


def test_the_data_feed_carries_data_not_markup(pool, mid_season, tmp_path):
    """standings.json is a machine-readable interface, so no SVG in it.

    Every chart is a string of markup. One leaking into the feed tripled its
    size, and anyone parsing it wants the numbers.
    """
    render_site(pool, mid_season, tmp_path, simulations=200)
    raw = (tmp_path / "data" / "standings.json").read_text()

    assert "<svg" not in raw
    data = json.loads(raw)
    projection = data["entrants"][0]["projection"]
    # The numbers behind the charts are still there.
    assert {"p_first", "p_cash", "mean_points", "rivals"} <= set(projection)
