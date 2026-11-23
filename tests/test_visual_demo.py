"""Regressions for presenting the HMM posterior and starting the autonomous demo."""

import sys

import numpy as np
import pygame
import pytest

import main
from ghost_hunter.crypt_board import CryptBoard
from ghost_hunter.environment import GridEnvironment
from ghost_hunter.gui import PygameVisualizer


def test_heatmap_distinguishes_posteriors_with_the_same_map(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    env = GridEnvironment(width=5, height=5, walls=set())
    env.reset(hunter_start=(0, 0), ghost_start=(4, 4))
    gui = PygameVisualizer(env)
    try:
        pygame.display.set_mode((960, 660))
        gui._sync_display()
        monkeypatch.setattr("ghost_hunter.gui.time.time", lambda: gui.start_time + 1)
        first = np.zeros((5, 5))
        first[1, 1], first[2, 2], first[3, 3] = .65, .05, .30
        second = first.copy()
        second[2, 2], second[3, 3] = .30, .05
        cell, left, top, *_ = gui._compute_layout()
        sample = (left + 2*cell + cell//2, top + 2*cell + cell//2)

        def center_pixel(belief):
            gui.render(belief)
            return gui.screen.get_at(sample)

        assert center_pixel(first) == center_pixel(second)  # Same proximity ring.
        gui.handle_events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h)])
        assert center_pixel(first) != center_pixel(second)
    finally:
        gui.close()


@pytest.mark.parametrize(("mode", "name"), [("demo", "HMM Hunter"), ("entropy", "Entropy Hunter")])
def test_demo_command_enters_the_hunt_with_its_arguments(monkeypatch, mode, name):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr(sys, "argv", ["main.py", mode, "--difficulty", "Hard",
                                      "--seed", "73", "--sigma", "2.0"])
    calls = []

    def hunt(gui, difficulty, mode, seed, sigma, *, player_name):
        calls.append((difficulty, mode, seed, sigma, player_name))
        return "exit"

    monkeypatch.setattr(main, "run_game_loop", hunt)
    monkeypatch.setattr(PygameVisualizer, "show_start_screen", lambda self: "exit")
    main.main()
    assert calls == [("Hard", mode, 73, 2.0, name)]


def test_crypt_walls_resist_heatmap_tints_and_refresh_on_restart(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    try:
        screen = pygame.display.set_mode((120, 120))
        board = CryptBoard()
        cell = 24
        sample = pygame.Rect(2 * cell, 2 * cell, cell, cell)

        def tile(env, target=None, levels=None):
            board.draw(screen, screen.get_rect(), env, cell, target, levels, elapsed=0)
            return pygame.image.tobytes(screen.subsurface(sample), "RGB")

        first = GridEnvironment(width=5, height=5, walls={(2, 2)})
        carved_stone = tile(first)
        assert tile(first, target=(2, 3)) == carved_stone
        assert tile(first, levels=np.ones((5, 5))) == carved_stone

        next_maze = GridEnvironment(width=5, height=5, walls={(1, 1)})
        paving = tile(next_maze)
        assert paving != carved_stone
        assert tile(next_maze, target=(2, 3)) != paving
    finally:
        pygame.display.quit()
