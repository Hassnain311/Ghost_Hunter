import pygame
import pytest

from ghost_hunter import competitive, mode_screens
from ghost_hunter.gui import PygameVisualizer
from ghost_hunter.leaderboard import Leaderboard
from ghost_hunter.race_scores import HunterScore, RaceScorecard


@pytest.fixture
def race(monkeypatch):
    cells = {(r, c) for r in range(10) for c in range(10)}
    monkeypatch.setattr(competitive, "generate_maze", lambda *args: (cells, set()))
    return competitive.ChaseRound("Beginner", 41, against_ai=True)


def test_shorter_route_can_beat_an_earlier_capture(race):
    race.positions = [(1, 1), (1, 3)]
    race.env.true_ghost_pos = (1, 2)
    for turn, move in enumerate(((-1, 0), (1, 0), (-1, 0), (1, 0), (0, -1)), 1):
        race.advance((None, move), False, elapsed=turn * .2)
    assert race.capture_ms == [None, 1000]
    assert not race.finished and not race.winners
    actions = race.scheduled_moves(1.2, ((0, 1), None))
    assert actions == ((0, 1), None)
    race.advance(actions, True, elapsed=1.2)
    assert race.finished and race.captors == (0,)
    card = race.scorecard()
    assert card.hunters == (HunterScore(1, 1200, True), HunterScore(5, 1000, True))
    assert [row.points for row in card.hunters] == [987, 985]
    assert race.winners == (0,) and card.verdict == "You win"


def test_idle_hunter_reaches_limit_after_ai_finishes(race):
    race.positions = [(0, 0), (1, 2)]
    race.env.true_ghost_pos = (1, 1)
    race.env.turn_limit = 3
    race.advance(((-1, 0), (0, -1)), False, elapsed=.2)
    assert race.wall_hits == (True, False)
    for now in (.4, .6):
        actions = race.scheduled_moves(now, (None, None))
        race.advance(actions, False, elapsed=now)
    assert race.finished and race.env.turn == 3
    assert race.scorecard().hunters == (HunterScore(0, 600, False), HunterScore(1, 200, True))
    assert race.scorecard().verdict == "AI wins"


def test_last_turn_simultaneous_capture_is_a_scored_draw(race):
    race.env.true_ghost_pos = (1, 1)
    race.env.turn_limit = 2
    race.positions = [(1, 0), (1, 2)]
    race.advance(((0, 0), None), False, elapsed=.2)
    race.advance(((0, 1), (0, -1)), True, elapsed=.4)
    assert race.env.captured and race.scorecard().winners == (0, 1)
    assert race.scorecard().verdict == "Draw"
    assert [row.points for row in race.scorecard().hunters] == [995, 995]


def test_ghost_movement_can_award_a_capture_without_a_hunter_move(race, monkeypatch):
    race.positions = [(1, 0), (8, 8)]
    race.env.true_ghost_pos = (1, 1)
    monkeypatch.setattr(race.env, "valid_moves", lambda pos: [(0, -1)])
    race.advance((None, None), True, elapsed=.7)
    assert race.capture_ms == [700, None] and race.env.turn == 0
    race.advance(((0, 1), (0, 0)), False, elapsed=1)
    assert race.positions[0] == (1, 0) and race.moves == [0, 0]
    assert race.player_radar() == []


def test_point_rounding_ties_and_timeouts():
    assert HunterScore(5, 2049, True).points == 975
    assert HunterScore(5, 2050, True).points == 974
    assert HunterScore(1, 999999, True).points == 1
    tie = RaceScorecard((HunterScore(8, 2000, True), HunterScore(6, 2200, True)))
    assert tie.winners == (0, 1) and tie.verdict == "Draw"
    escape = RaceScorecard((HunterScore(0, 2000, False), HunterScore(10, 2000, False)))
    assert all(hunter.points == 0 for hunter in escape.hunters)
    assert escape.winners == () and escape.verdict == "Draw"


def test_live_race_excludes_pause_and_saves_both_finish_times(race, monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    race.positions = [(1, 0), (1, 4)]
    race.env.true_ghost_pos = (1, 1)
    monkeypatch.setattr(race, "rival_action", lambda: (0, -1))
    monkeypatch.setattr(competitive, "ChaseRound", lambda *args, **kwargs: race)
    gui = PygameVisualizer(race.env)
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
    times = iter((0, .2, .4, 9.4, 9.6))
    batches = iter([[key(pygame.K_RIGHT)], [key(pygame.K_p)], [key(pygame.K_p)], [],
                    [key(pygame.K_SPACE)], [key(pygame.K_ESCAPE)]])
    monkeypatch.setattr(competitive.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    seen = []
    def result(*args, **kwargs):
        seen.append(kwargs["scorecard"])
        return mode_screens.show_capture_result(*args, **kwargs)
    monkeypatch.setattr(competitive, "show_capture_result", result)
    store = Leaderboard(tmp_path / "race.sqlite3")
    def save(captures):
        store.record_times("AI vs Human", "Beginner", captures)
    try:
        assert competitive.run_race(gui, "Beginner", "AI vs Human", ("Player", "AI"), save) == "menu"
        assert seen == [RaceScorecard((HunterScore(1, 200, True), HunterScore(3, 600, True)))]
        assert [(n, ms) for n, ms, _ in store.top("AI vs Human", "Beginner")] == [("Player", 200), ("AI", 600)]
    finally:
        gui.close()


def test_scorecard_replay_starts_a_fresh_round_and_can_end_without_a_capture(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    cells = {(r, c) for r in range(10) for c in range(10)}
    monkeypatch.setattr(competitive, "generate_maze", lambda *args: (cells, set()))
    make_race = competitive.ChaseRound
    rounds = []
    def new_round(*args, **kwargs):
        current = make_race(*args, **kwargs)
        current.positions = [(1, 0), (1, 2)]
        current.env.true_ghost_pos = (1, 1) if not rounds else (9, 9)
        current.env.turn_limit = 1
        monkeypatch.setattr(current, "rival_action", lambda: (0, -1) if len(rounds) == 1 else (0, 0))
        rounds.append(current)
        return current

    monkeypatch.setattr(competitive, "ChaseRound", new_round)
    ticks = iter((0, .2, 10, 10.5))
    monkeypatch.setattr(competitive.time, "monotonic", lambda: next(ticks))
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
    batches = iter([[key(pygame.K_RIGHT)], [key(pygame.K_SPACE)], [key(pygame.K_RETURN)], [],
                    [pygame.event.Event(pygame.WINDOWCLOSE)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    gui = PygameVisualizer(competitive.GridEnvironment(walls=set()))
    saved = []
    try:
        assert competitive.run_race(gui, "Beginner", "AI vs Human", ("A", "AI"), saved.append) == "exit"
        assert saved == [[("A", .2), ("AI", .2)]]
        assert rounds[0].scorecard().winners == (0, 1)
        assert rounds[1].scorecard().hunters == (HunterScore(0, 500, False), HunterScore(0, 500, False))
        assert rounds[1].finished and rounds[1].scorecard().verdict == "Draw"
    finally:
        gui.close()
