import numpy as np
import pygame
import pytest

from config import DIFFICULTY_CONFIGS, HUMAN_MOVE_INTERVAL_SEC
from ghost_hunter import competitive, gui as gui_module
from ghost_hunter.gui import PygameVisualizer
from ghost_hunter.hmm import HMMTracker
from ghost_hunter.race_rival import RaceRival


@pytest.mark.parametrize("difficulty", DIFFICULTY_CONFIGS)
def test_both_hunters_get_exactly_the_same_steps_even_after_a_slow_frame(difficulty):
    race = competitive.ChaseRound(difficulty, 24, against_ai=True)
    step = HUMAN_MOVE_INTERVAL_SEC
    command = (0, 1)
    opportunities = []
    for now in (0, .01, step - .001, step, step + .01, 2*step - .001, 2*step, 2.2, 2.2):
        player, rival = race.scheduled_moves(now, (command, None))
        assert (player is None) == (rival is None)
        if player is not None:
            assert player == command
            assert abs(rival[0]) + abs(rival[1]) <= 1
            opportunities.append(now)
    assert opportunities == pytest.approx([0, step, 2*step, 2.2])
    # Leaving a player slot empty cannot bank an extra move for either competitor.
    player, rival = race.scheduled_moves(3, (None, None))
    assert player is None and rival is not None
    assert race.scheduled_moves(3.01, (command, None)) == (None, None)


@pytest.mark.parametrize("difficulty", ["Intermediate", "Hard", "Extreme"])
def test_private_astar_takes_shortest_detour_around_wall(difficulty):
    walls = {(1, 1), (2, 1)}
    cells = {(r, c) for r in range(3) for c in range(5)} - walls
    tracker = HMMTracker(cells, walls, (3, 5))
    start, goal = (2, 0), (2, 3)
    tracker.belief_vec.fill(0)
    tracker.belief_vec[tracker.cell_to_idx[goal]] = 1
    rival = RaceRival(difficulty, tracker, ghost_interval=100, seed=5)
    route = [start]
    while route[-1] != goal and len(route) <= 7:
        r, c = route[-1]
        dr, dc = rival.choose((r, c), (0, 4), reading=3)
        assert abs(dr) + abs(dc) == 1
        assert (r+dr, c+dc) in cells
        route.append((r+dr, c+dc))
    # Three columns east plus two rows up and back down. The first step goes away from the goal.
    assert route[1] == (1, 0)
    assert route[-1] == goal and len(route) - 1 == 7


def test_rival_forecast_does_not_advance_either_live_tracker():
    race = competitive.ChaseRound("Extreme", 63, against_ai=True)
    public = race.tracker.belief_vec.copy()
    private = race.rival_tracker.belief_vec.copy()
    forecast = race.rival_tracker.forecast(3, *race.positions)
    race.rival_action()
    assert forecast.sum() == pytest.approx(1)
    np.testing.assert_array_equal(public, race.tracker.belief_vec)
    np.testing.assert_array_equal(private, race.rival_tracker.belief_vec)


def test_exact_position_enters_both_trackers_only_during_a_public_glimpse():
    race = competitive.ChaseRound("Hard", 29, against_ai=True)
    target = race.env.true_ghost_pos
    old = race.rival_tracker.belief_vec.copy()
    assert race.update_glimpse(7.99) == 0
    np.testing.assert_array_equal(old, race.rival_tracker.belief_vec)
    assert race.update_glimpse(8) > 0
    for tracker in (race.tracker, race.rival_tracker):
        assert tracker.get_map_estimate() == target
        assert tracker.belief_vec.max() == 1
    assert race.update_glimpse(9) == 0
    race.env.true_ghost_pos = next(c for c in race.env.free_cells if c != target and c not in race.positions)
    race.update_glimpse(9.1)
    assert race.rival_tracker.get_map_estimate() == target


def test_private_routes_never_appear_on_the_rendered_board_or_player_radar(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    cells = {(r, c) for r in range(10) for c in range(10)}
    monkeypatch.setattr(competitive, "generate_maze", lambda *args: (cells, set()))
    monkeypatch.setitem(DIFFICULTY_CONFIGS, "Hard", DIFFICULTY_CONFIGS["Beginner"])
    race = competitive.ChaseRound("Hard", 84, against_ai=True)
    race.env.true_ghost_pos = (0, 4)
    gui = PygameVisualizer(race.env)
    try:
        pygame.display.set_mode((960, 660))
        gui._sync_display()
        gui.show_probability = True
        now = gui.start_time + 1
        monkeypatch.setattr(gui_module.time, "time", lambda: now)
        drawn = []
        draw = gui_module.draw_radar_path

        def record(screen, bounds, cell, path, pulse):
            drawn.append(tuple(path))
            draw(screen, bounds, cell, path, pulse)

        monkeypatch.setattr(gui_module, "draw_radar_path", record)
        def frame_with_private_target(target):
            tracker = race.rival_tracker
            tracker.belief_vec.fill(0)
            tracker.belief_vec[tracker.cell_to_idx[target]] = 1
            move = race.rival_action()
            gui.render(race.tracker.get_belief_grid(), "Hard / AI vs Human",
                       race=(race, ("Player", "AI"), 1, None, True))
            return move, pygame.image.tobytes(gui.screen, "RGB")

        def compare_routes():
            up, first = frame_with_private_target((5, 9))
            left, second = frame_with_private_target((9, 5))
            assert up == (-1, 0) and left == (0, -1)
            assert first == second
            return first

        without_radar = compare_routes()
        assert drawn == []
        key = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x)
        assert gui.handle_events([key], competitive=True)[4]
        assert gui.trigger_astar_hint(race.player_radar())
        with_radar = compare_routes()
        expected = tuple((0, c) for c in range(5))
        assert drawn == [expected, expected]
        assert without_radar != with_radar
    finally:
        gui.close()


def test_live_race_refreshes_only_the_human_radar_when_the_hunters_move(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    cells = {(r, c) for r in range(10) for c in range(10)}
    monkeypatch.setattr(competitive, "generate_maze", lambda *args: (cells, set()))
    race = competitive.ChaseRound("Beginner", 12, against_ai=True)
    race.env.true_ghost_pos = (0, 4)
    monkeypatch.setattr(competitive, "ChaseRound", lambda *args, **kwargs: race)
    ticks = iter([0, .2, .4, .6, .8])
    monkeypatch.setattr(competitive.time, "monotonic", lambda: next(ticks))
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
    batches = iter([[], [key(pygame.K_x)], [key(pygame.K_RIGHT), key(pygame.K_h)],
                    [pygame.event.Event(pygame.WINDOWCLOSE)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    gui = PygameVisualizer(race.env)
    try:
        rendered = []
        draw = gui.render

        def frame(belief, *args, **kwargs):
            np.testing.assert_array_equal(belief, race.tracker.get_belief_grid())
            rendered.append(tuple(gui.astar_path))
            return draw(belief, *args, **kwargs)

        monkeypatch.setattr(gui, "render", frame)
        assert competitive.run_race(gui, "Beginner", "AI vs Human", ("Player", "AI"), None) == "exit"
        assert rendered == [(), tuple((0, c) for c in range(5)), tuple((0, c) for c in range(1, 5))]
        assert gui.show_probability
    finally:
        gui.close()
