import pytest
from pydantic import ValidationError

from schemas.game import GameState, GameContext


def test_game_state_valid():
    gs = GameState(
        ydstogo=3, yardline_100=35, score_differential=-7,
        half_seconds_remaining=600, game_seconds_remaining=2400,
        quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
    )
    assert gs.ydstogo == 3
    assert gs.wp == 0.35


def test_game_state_validates_ranges():
    with pytest.raises(ValidationError):
        GameState(
            ydstogo=0, yardline_100=35, score_differential=-7,
            half_seconds_remaining=600, game_seconds_remaining=2400,
            quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
        )


def test_game_state_validates_wp():
    with pytest.raises(ValidationError):
        GameState(
            ydstogo=3, yardline_100=35, score_differential=-7,
            half_seconds_remaining=600, game_seconds_remaining=2400,
            quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=1.5,
        )


def test_game_state_to_dict():
    gs = GameState(
        ydstogo=3, yardline_100=35, score_differential=-7,
        half_seconds_remaining=600, game_seconds_remaining=2400,
        quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
    )
    d = gs.model_dump()
    assert d["ydstogo"] == 3
    assert len(d) == 9


def test_game_context_extends_game_state():
    ctx = GameContext(
        ydstogo=3, yardline_100=35, score_differential=-7,
        half_seconds_remaining=600, game_seconds_remaining=2400,
        quarter_seconds_remaining=600, qtr=3, goal_to_go=0, wp=0.35,
        game_id="2023_01_KC_DET", season=2023, week=1,
        posteam="KC", defteam="DET",
    )
    assert ctx.game_id == "2023_01_KC_DET"
    assert ctx.posteam == "KC"
