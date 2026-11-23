import numpy as np
import pygame
import pytest

import main
from ghost_hunter import competitive, mode_screens
from ghost_hunter.agents import NaiveSensorAgent
from ghost_hunter.environment import GridEnvironment, get_ghost_transition_probabilities
from ghost_hunter.gui import PygameVisualizer
from ghost_hunter.hmm import HMMTracker
from ghost_hunter.leaderboard import Leaderboard, MODES
from ghost_hunter.sensor_model import likelihood_by_distance


def test_solo_reports_blocked_steps_and_supports_radar_and_reveal(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    cells = {(r, c) for r in range(10) for c in range(10)}
    monkeypatch.setattr(main, "generate_maze", lambda **kw: (cells, set()))
    gui = PygameVisualizer(GridEnvironment(walls=set()))
    frames = []
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
    batches = iter([[key(pygame.K_g), key(pygame.K_x)], [], [key(pygame.K_LEFT)],
                    [key(pygame.K_PERIOD)], [pygame.event.Event(pygame.WINDOWCLOSE)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    times = iter(range(1000, 3000, 3))
    monkeypatch.setattr(main.time, "time", lambda: next(times))
    monkeypatch.setattr(GridEnvironment, "move_ghost", lambda self: None)
    monkeypatch.setattr(gui, "render", lambda **kw: frames.append(gui.env.turn))
    try:
        assert main.run_game_loop(gui, mode="play", player_name="A") == "exit"
        assert frames == [0, 0, 1, 2]
        assert gui.blocked_hunter == "A"
        assert gui.reveal_ghost and gui.astar_use_count == 1
    finally:
        gui.close()


def test_instructions_shortcut_and_three_mode_menu(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    gui = PygameVisualizer(GridEnvironment(walls=set()))
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
    buttons = {}
    pages = []
    draw_button, draw_page = gui._menu_button, gui._menu_page

    def button(rect, label, *args, **kwargs):
        buttons[label] = (rect.copy(), kwargs.get("hint"))
        return draw_button(rect, label, *args, **kwargs)

    def page(title, *args, **kwargs):
        pages.append(title)
        return draw_page(title, *args, **kwargs)

    monkeypatch.setattr(gui, "_menu_button", button)
    monkeypatch.setattr(gui, "_menu_page", page)
    try:
        batches = iter([[key(pygame.K_g), key(pygame.K_l)], [key(pygame.K_i)], [key(pygame.K_ESCAPE)]])
        monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
        assert gui.show_start_screen() == "leaderboard"
        assert gui.show_start_screen() == "instructions"
        assert buttons["Instructions"][1] == "I"
        assert gui.show_instructions_screen() is None and pages[-1] == "Instructions"
        assert {"Watch AI Hunt", "Watch Entropy Hunter"} <= buttons.keys()
        monkeypatch.setattr(pygame.event, "get", lambda: [
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=buttons["Instructions"][0].center)])
        assert gui.show_start_screen() == "instructions"
        buttons.clear()
        monkeypatch.setattr(pygame.event, "get", lambda: [key(pygame.K_4), key(pygame.K_2)])
        assert mode_screens.choose_mode(gui) == "AI vs Human"
        assert MODES == ("Solo", "AI vs Human", "Human vs Human")
        assert set(buttons) == {"Return", *MODES}
    finally:
        gui.close()


def test_one_ping_baseline_uses_reading_without_belief_history():
    cells = {(0, c) for c in range(5)}
    agent = NaiveSensorAgent(cells, set())
    assert agent.select_action({"hunter_pos": (0, 0), "reading": 4}) == (0, 1)
    assert agent.last_estimate == (0, 4)
    assert agent.select_action({"hunter_pos": (0, 0), "reading": 0}) == (0, 0)
    assert agent.last_estimate == (0, 0)


def test_filter_matches_the_simulated_motion_and_clipped_sensor():
    cells = {(r, c) for r in range(3) for c in range(3)}
    tracker = HMMTracker(cells, set(), (3, 3))
    tracker.belief_vec.fill(0)
    tracker.belief_vec[tracker.cell_to_idx[1, 1]] = 1
    hunter = (0, 1)
    tracker.time_update(hunter)
    law = get_ghost_transition_probabilities((1, 1), cells, hunter)
    for cell, expected in law.items():
        assert np.isclose(tracker.belief_vec[tracker.cell_to_idx[cell]], expected)
    distribution = sum(likelihood_by_distance(r, 1.2, 4) for r in range(30))
    np.testing.assert_allclose(distribution, np.ones(5))


def test_rival_decides_from_sensor_belief_and_legal_neighbors(monkeypatch):
    first = competitive.ChaseRound("Hard", 81, against_ai=True)
    second = competitive.ChaseRound("Hard", 81, against_ai=True)
    monkeypatch.setattr(competitive, "astar", lambda *args: pytest.fail("The race rival requested a radar path"))
    second.env.true_ghost_pos = (0, 0)
    # The physical position alone cannot affect the decision until a sensor ping arrives.
    dr, dc = first.rival_action()
    assert (dr, dc) == second.rival_action()
    r, c = first.positions[1]
    assert (r + dr, c + dc) in first.env.free_cells


def test_last_turn_tie_records_both_competitors_in_the_correct_mode(monkeypatch, tmp_path):
    cells = {(r, c) for r in range(10) for c in range(10)}
    monkeypatch.setattr(competitive, "generate_maze", lambda *args: (cells, set()))
    race = competitive.ChaseRound("Beginner", 42)
    race.positions = [(1, 0), (1, 2)]
    race.env.true_ghost_pos = (1, 1)
    race.env.turn = race.env.turn_limit - 1
    race.advance([(0, 1), (0, -1)], move_ghost=True, elapsed=8.25)
    assert race.finished and race.env.captured and race.winners == (0, 1)
    store = Leaderboard(tmp_path / "scores.sqlite3")
    names = ("A", "B")
    store.record("Human vs Human", "Beginner", [names[i] for i in race.winners], 8.25)
    store.record("AI vs Human", "Beginner", ["C"], 5)
    assert [(n, ms) for n, ms, _ in store.top("Human vs Human", "Beginner")] == [("A", 8250), ("B", 8250)]
    assert store.top("Solo", "Beginner") == []
