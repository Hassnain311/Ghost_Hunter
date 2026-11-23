from unittest.mock import Mock

import pygame
import pytest

import main
from ghost_hunter import mode_screens
from ghost_hunter.audio import GameAudio
from ghost_hunter.environment import GridEnvironment
from ghost_hunter.gui import PygameVisualizer
from ghost_hunter.instance import interactive_instance


@pytest.mark.parametrize("caught", [True, False], ids=["last-step-capture", "steps-exhausted"])
def test_round_plays_one_outcome_across_board_and_result(monkeypatch, caught):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    gui = PygameVisualizer(GridEnvironment(walls=set()))
    try:
        gui.is_fullscreen = False
        cells = {(r, c) for r in range(10) for c in range(10)}
        monkeypatch.setattr(main, "generate_maze", lambda **kw: (cells, set()))
        real_reset = GridEnvironment.reset

        def reset(env):
            env.turn_limit = 1
            return real_reset(env, (0, 0), (0, 1) if caught else (9, 9))

        monkeypatch.setattr(GridEnvironment, "reset", reset)
        finish = Mock(wraps=gui.audio.finish)
        monkeypatch.setattr(gui.audio, "finish", finish)
        key = lambda code: pygame.event.Event(pygame.KEYDOWN, key=code)
        monkeypatch.setattr(pygame.event, "get", lambda: [key(pygame.K_RIGHT)])

        def score(names, elapsed):
            # Local storage can block. Start the cue before attempting the write.
            assert gui.audio.outcome.get_sound() is gui.audio._sounds["capture"]

        def result(*args, **kwargs):
            cue = "capture" if caught else "escape"
            assert gui.audio.outcome.get_sound() is gui.audio._sounds[cue]
            # Draw two capture frames and two result frames, then return to the menu.
            events = ([[], [key(pygame.K_SPACE)]] if caught else []) + [[], [key(pygame.K_ESCAPE)]]
            with monkeypatch.context() as m:
                it = iter(events)
                m.setattr(pygame.event, "get", lambda: next(it))
                return mode_screens.show_capture_result(*args, **kwargs)

        monkeypatch.setattr(main, "show_capture_result", result)
        assert main.run_game_loop(gui, player_name="Hunter", save_score=score) == "menu"
        finish.assert_called_once_with(captured=caught)
    finally:
        gui.close()


def test_audio_without_an_output_device(monkeypatch):
    monkeypatch.setattr(pygame.mixer, "get_init", lambda: None)
    audio = GameAudio()
    audio.begin_hunt()
    audio.footstep()
    audio.sound_radar()
    audio.button("exit")
    audio.wall_hit()
    audio.finish(captured=True)
    audio.toggle_mute()
    audio.close()
    assert audio.status == "Unavailable"


@pytest.mark.parametrize("close_on", ["Exit", "Choose Your Mode", "Name Your Hunter"])
def test_launcher_stops_audio_on_button_or_native_close(monkeypatch, tmp_path, close_on):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    buttons, played = {}, []
    draw_button = PygameVisualizer._menu_button
    play_button = GameAudio.button

    def draw(gui, rect, label, *args, **kwargs):
        buttons[label] = rect.copy()
        return draw_button(gui, rect, label, *args, **kwargs)

    def sound(audio, cue="select"):
        play_button(audio, cue)
        assert audio.interface.get_sound() is audio._sounds["menu_" + cue]
        played.append(cue)

    def click(label):
        return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=buttons[label].center)

    def events():
        if close_on == "Exit":
            return [] if played else [click("Exit")]
        if not played:
            return [click("Enter the Crypt")]
        if close_on == "Name Your Hunter":
            if len(played) == 1:
                return [click("Solo")]
            if len(played) == 2:
                return [click("Beginner")]
        # A key activation and native close can arrive in the same SDL event batch.
        return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN),
                pygame.event.Event(pygame.WINDOWCLOSE)]

    monkeypatch.setattr(PygameVisualizer, "_menu_button", draw)
    monkeypatch.setattr(GameAudio, "button", sound)
    monkeypatch.setattr(pygame.event, "get", events)
    main.interactive_launcher()
    expected = {"Exit": ["exit"], "Choose Your Mode": ["start"],
                "Name Your Hunter": ["start", "select", "select"]}
    assert played == expected[close_on]
    assert pygame.mixer.get_init() is None
    assert not pygame.display.get_init()


def test_single_interactive_instance_releases_lock(tmp_path):
    path = tmp_path / "game.lock"
    with interactive_instance(path) as first, interactive_instance(path) as second:
        assert first and not second
    with interactive_instance(path) as reopened:
        assert reopened
