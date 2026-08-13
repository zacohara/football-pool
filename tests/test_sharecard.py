"""The link-preview card: it must draw, and it must never lie about a model."""

from PIL import Image

from football_pool.sharecard import card_content, draw_card


def _read(path):
    with Image.open(path) as img:
        return img.size, img.format


def test_in_season_card_draws_the_leaderboard(tmp_path):
    out = tmp_path / "card.png"
    draw_card(
        out,
        year=2026,
        headline="The standings",
        sub="Week 11 · $60 pot · 6 in",
        lines=[("1", "Aunt Carol", "24.10"), ("2", "Brandon", "22.45")],
    )
    size, fmt = _read(out)
    assert size == (1200, 630)
    assert fmt == "PNG"


def test_an_empty_card_still_draws(tmp_path):
    """Before picks exist there is nothing to list; the card must not care."""
    out = tmp_path / "card.png"
    draw_card(out, year=2026, headline="Kickoff soon", sub="$0 pot · 0 in", lines=[])
    assert _read(out)[0] == (1200, 630)


def test_in_season_content_is_the_top_five_actuals():
    state = {"week": 11, "phase": "Week 11", "pot": 60.0, "entrants": 6}
    rows = [{"rank": i + 1, "name": f"P{i}", "banked": 30.0 - i} for i in range(7)]
    c = card_content(state, rows, None)
    assert c["modelled"] is False
    assert len(c["lines"]) == 5
    assert c["lines"][0] == ("1", "P0", "30.00")


def test_preseason_content_is_the_model_and_says_so():
    """A forecast on the card must wear the model's colour, like the site."""
    state = {"week": None, "phase": "Preseason", "pot": 60.0, "entrants": 6}
    rows = [{"rank": 1, "name": "P0", "banked": 0.0}]
    forecast = {"finish_rows": [{"name": "P0", "probs": [0.41]}]}
    c = card_content(state, rows, forecast)
    assert c["modelled"] is True
    assert c["lines"][0] == ("1", "P0", "41%")


def test_preseason_without_a_model_falls_back_to_the_field():
    state = {"week": None, "phase": "Preseason", "pot": 60.0, "entrants": 2}
    rows = [{"rank": 1, "name": "A", "banked": 0.0}, {"rank": 2, "name": "B", "banked": 0.0}]
    c = card_content(state, rows, None)
    assert c["modelled"] is False
    assert [line[1] for line in c["lines"]] == ["A", "B"]


def test_site_url_is_derived_from_the_actions_environment():
    from football_pool.cli import default_site_url

    assert default_site_url({"GITHUB_REPOSITORY": "zacohara/football-pool"}) == (
        "https://zacohara.github.io"
    )
    assert default_site_url({}) == ""
    assert default_site_url({"GITHUB_REPOSITORY": "nonsense"}) == ""
