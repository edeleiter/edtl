"""NFL game state models."""

from pydantic import Field

from schemas._base import (
    StrictModel,
    YardLine,
    YardsToGo,
    Quarter,
    GameSeconds,
    HalfSeconds,
    QuarterSeconds,
    Probability,
)


class GameState(StrictModel):
    """The 9 features that define a 4th-down game situation."""

    ydstogo: YardsToGo = Field(description="Yards needed for first down")
    yardline_100: YardLine = Field(description="Yards from opponent end zone")
    score_differential: int = Field(description="Positive = leading, negative = trailing")
    half_seconds_remaining: HalfSeconds
    game_seconds_remaining: GameSeconds
    quarter_seconds_remaining: QuarterSeconds
    qtr: Quarter
    goal_to_go: int = Field(ge=0, le=1, description="1 if goal-to-go, 0 otherwise")
    wp: Probability = Field(description="Pre-play win probability")


class GameContext(GameState):
    """GameState extended with metadata for logging and analysis."""

    game_id: str = Field(description="Unique game identifier (e.g., 2023_01_KC_DET)")
    season: int = Field(ge=2000, le=2100, description="NFL season year")
    week: int = Field(ge=1, le=22, description="Season week (1-18 regular, 19-22 playoffs)")
    posteam: str = Field(min_length=2, max_length=3, description="Possessing team abbreviation")
    defteam: str = Field(min_length=2, max_length=3, description="Defending team abbreviation")
