"""Shared-maze competitive rounds. Solo uses its original game loop."""

import random
import time

import pygame
import numpy as np

from config import DEFAULT_SENSOR_SIGMA, DIFFICULTY_CONFIGS, HUMAN_MOVE_INTERVAL_SEC
from ghost_hunter.environment import GridEnvironment, generate_maze
from ghost_hunter.hmm import HMMTracker
from ghost_hunter.pathfinding import astar
from ghost_hunter.race_rival import RaceRival
from ghost_hunter.race_scores import HunterScore, RaceScorecard
from ghost_hunter.mode_screens import show_capture_result
from ghost_hunter.sensor_model import manhattan_distance, sample_sensor_reading

WASD = {pygame.K_w: (-1, 0), pygame.K_s: (1, 0), pygame.K_a: (0, -1), pygame.K_d: (0, 1)}
ARROWS = {pygame.K_UP: (-1, 0), pygame.K_DOWN: (1, 0), pygame.K_LEFT: (0, -1), pygame.K_RIGHT: (0, 1)}


class ChaseRound:
    def __init__(self, difficulty, seed, *, against_ai=False):
        cfg = DIFFICULTY_CONFIGS[difficulty]
        width, height = cfg["width"], cfg["height"]
        self.ghost_interval = cfg["ghost_speed"]
        self._glimpse_interval = cfg["glimpse_interval"]
        self._glimpse_duration = cfg["glimpse_duration"]
        self._next_glimpse = self._glimpse_interval
        self._glimpse_until = 0.0
        self._visible = False
        cells, walls = generate_maze(seed, cfg["density"], width, height)
        self.env = GridEnvironment(width=width, height=height, walls=walls, seed=seed,
                                   sensor_sigma=DEFAULT_SENSOR_SIGMA, turn_limit=cfg["turn_limit"])
        self.positions = [(0, 0), (height - 1, width - 1)]
        spawn = random.Random(seed).choice(sorted(cells.difference(self.positions)))
        self.env.reset(hunter_start=self.positions[0], ghost_start=spawn)
        self.tracker = HMMTracker(cells, walls, (height, width), self.env.sensor_sigma)
        self.tracker.observation_update(self.env.last_reading, self.positions[0])
        self._next_move = [0.0, 0.0]
        self.moves = [0, 0]
        self.capture_ms = [None, None]
        self.elapsed_ms = 0
        self.captors = ()
        self.rival_tracker = None
        if against_ai:
            self.rival_tracker = HMMTracker(cells, walls, (height, width), self.env.sensor_sigma)
            self.rival = RaceRival(difficulty, self.rival_tracker, self.ghost_interval, seed)
            self.rival_sensor_rng = np.random.default_rng([seed, 1])
            self._observe_rival()
        self.wall_hits = (False, False)

    def _observe_rival(self):
        # The first reading is public on the HUD. The rival carries its own noisy sensor.
        self.rival_tracker.observation_update(self.env.last_reading, self.positions[0])
        distance = manhattan_distance(self.positions[1], self.env.true_ghost_pos)
        self.rival_reading = sample_sensor_reading(distance, self.env.sensor_sigma, self.rival_sensor_rng)
        self.rival_tracker.observation_update(self.rival_reading, self.positions[1])
        if not self.env.captured:
            self._exclude_hunters()

    def _exclude_hunters(self):
        # A hunter who already finished can still share a cell with the ghost.
        belief = self.rival_tracker.belief_vec
        for i, position in enumerate(self.positions):
            if self.capture_ms[i] is None:
                belief[self.rival_tracker.cell_to_idx[position]] = 0
        belief /= belief.sum()

    def _record_public_sighting(self):
        for tracker in (self.tracker, self.rival_tracker):
            if tracker is not None:
                tracker.belief_vec.fill(0)
                tracker.belief_vec[tracker.cell_to_idx[self.env.true_ghost_pos]] = 1

    def update_glimpse(self, elapsed):
        if elapsed >= self._next_glimpse:
            self._glimpse_until = elapsed + self._glimpse_duration
            self._next_glimpse = elapsed + self._glimpse_interval
        remaining = max(0.0, self._glimpse_until - elapsed)
        was_visible = self._visible
        self._visible = remaining > 0
        if self._visible and not was_visible and self.rival_tracker is not None:
            self._record_public_sighting()
        return remaining

    def rival_action(self):
        return self.rival.choose(self.positions[1], self.positions[0], self.rival_reading)

    def player_ready(self, elapsed):
        return elapsed >= self._next_move[0]

    def scheduled_moves(self, elapsed, requested):
        if self.rival_tracker is not None:
            # Both competitors receive the same movement opportunity, even if one waits.
            if not self.player_ready(elapsed):
                return (None, None)
            self._next_move[0] = elapsed + HUMAN_MOVE_INTERVAL_SEC
            player = requested[0] if self.capture_ms[0] is None else None
            if self.capture_ms[1] is None:
                return (player, self.rival_action())
            # With the AI finished, an idle human still uses the shared turn budget.
            return (player or (0, 0), None)

        actions = [None, None]
        for i, move in enumerate(requested):
            if move is not None and elapsed >= self._next_move[i]:
                actions[i] = move
                self._next_move[i] = elapsed + HUMAN_MOVE_INTERVAL_SEC
        return actions

    def player_radar(self):
        # The board may display routes only for human-controlled pieces.
        if self.rival_tracker is not None and self.capture_ms[0] is not None:
            return []
        hunters = self.positions[:1] if self.rival_tracker is not None else self.positions
        path = []
        for position in hunters:
            path.extend(astar(position, self.env.true_ghost_pos, self.env.walls,
                              (self.env.height, self.env.width)))
        return path

    @property
    def finished(self):
        return self.env.captured or self.env.turn >= self.env.turn_limit

    def scorecard(self):
        return RaceScorecard(tuple(
            HunterScore(moves, self.elapsed_ms if finish is None else finish, finish is not None)
            for moves, finish in zip(self.moves, self.capture_ms)
        ))

    @property
    def winners(self):
        if self.rival_tracker is None:
            return self.captors
        return self.scorecard().winners if self.finished else ()

    def _check_capture(self):
        newly_caught = tuple(i for i, pos in enumerate(self.positions)
                             if self.capture_ms[i] is None and pos == self.env.true_ghost_pos)
        if newly_caught:
            self.captors = newly_caught
            for i in newly_caught:
                self.capture_ms[i] = self.elapsed_ms
            self.env.captured = self.rival_tracker is None or None not in self.capture_ms

    def advance(self, actions, move_ghost, *, elapsed):
        if self.finished:
            return
        self.elapsed_ms = round(1000 * elapsed)
        self.captors = ()
        hits = [False, False]
        for index, action in enumerate(actions):
            if action is None or self.capture_ms[index] is not None:
                continue
            row, col = self.positions[index]
            destination = row + action[0], col + action[1]
            if destination in self.env.free_cells:
                if destination != self.positions[index]:
                    self.moves[index] += 1
                self.positions[index] = destination
            else:
                hits[index] = True
        self.wall_hits = tuple(hits)
        self.env.hunter_pos = self.positions[0]
        if any(action is not None for action in actions):
            self.env.turn += 1
        self._check_capture()
        if move_ghost and not self.env.captured:
            # Bias ghost movement away from the nearest hunter, with no fixed tie preference.
            ghost = self.env.true_ghost_pos
            distance = min(manhattan_distance(ghost, pos) for pos in self.positions)
            destinations = [(ghost[0] + dr, ghost[1] + dc) for dr, dc in self.env.valid_moves(ghost)]
            distances = [min(manhattan_distance(cell, pos) for pos in self.positions) for cell in destinations]
            weights = [2.0 ** (value - distance) for value in distances]
            selected = self.env.rng.choice(len(destinations), p=[w / sum(weights) for w in weights])
            self.env.true_ghost_pos = destinations[selected]
            self.tracker.time_update(*self.positions)
            if self.rival_tracker is not None:
                self._exclude_hunters()
                self.rival_tracker.time_update(*self.positions)
            self._check_capture()
        self.env.last_reading = sample_sensor_reading(
            manhattan_distance(self.positions[0], self.env.true_ghost_pos), self.env.sensor_sigma, self.env.rng)
        self.tracker.observation_update(self.env.last_reading, self.positions[0])
        if self.rival_tracker is not None:
            self._observe_rival()
            if self._visible:
                self._record_public_sighting()


def run_race(gui, difficulty, mode, names, save_times):
    against_ai = mode == "AI vs Human"
    bindings = (WASD | ARROWS | {pygame.K_PERIOD: (0, 0)}, {}) if against_ai else (WASD, ARROWS)
    cfg = DIFFICULTY_CONFIGS[difficulty]
    while True:
        seed = random.randrange(1, 2**31)
        race = ChaseRound(difficulty, seed, against_ai=against_ai)
        env = race.env
        gui.resize_for_difficulty(env, cfg["cell_size"])
        gui.blocked_until = 0
        gui.audio.begin_hunt()
        gui.show_probability = False
        gui.paused = gui.reveal_ghost = False
        gui.astar_use_count = 0
        gui.astar_active_until = gui.astar_cooldown_until = 0
        gui.astar_path = []
        gui.ghost_glimpse_until = 0
        elapsed = 0.0
        last_frame = time.monotonic()
        next_ghost = race.ghost_interval
        queued = [None, None]
        save_error = None

        while True:
            now = time.monotonic()
            if not gui.paused and not race.finished:
                elapsed += now - last_frame
            last_frame = now
            events = pygame.event.get()
            running, _, restart, menu, radar, pause_action = gui.handle_events(events, competitive=True)
            if not running:
                return "exit"
            if menu or pause_action == "quit_to_menu":
                return "menu"
            if pause_action == "change_difficulty":
                return "difficulty"
            if restart:
                break

            if gui.paused:
                queued = [None, None]
            elif not race.finished:
                if radar and (not against_ai or race.capture_ms[0] is None):
                    gui.trigger_astar_hint(race.player_radar())
                glimpse = race.update_glimpse(elapsed)
                gui.ghost_glimpse_until = time.time() + glimpse if glimpse else 0
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        for index, keys in enumerate(bindings):
                            if event.key in keys:
                                queued[index] = keys[event.key]
                held = pygame.key.get_pressed()
                requested = [queued[i] or next((move for key, move in keys.items() if held[key]), None)
                             for i, keys in enumerate(bindings)]
                actions = race.scheduled_moves(elapsed, requested)
                for i, move in enumerate(actions):
                    if move is not None:
                        queued[i] = None
                ghost_due = elapsed >= next_ghost
                if ghost_due or any(action is not None for action in actions):
                    previous_positions = race.positions.copy()
                    race.advance(actions, ghost_due, elapsed=elapsed)
                    if race.finished:
                        gui.audio.finish(captured=any(t is not None for t in race.capture_ms))
                    elif any(race.wall_hits):
                        gui.wall_collision(names[race.wall_hits.index(True)])
                    elif race.positions != previous_positions:
                        gui.audio.footstep()
                    if ghost_due:
                        next_ghost = elapsed + race.ghost_interval
                    if race.finished:
                        captures = [(name, ms / 1000) for name, ms in zip(names, race.capture_ms) if ms is not None]
                        if captures and save_times is not None:
                            save_error = save_times(captures)

                        def draw_capture():
                            gui.render(
                                race.tracker.get_belief_grid(), f"{difficulty} / {mode}",
                                race=(race, names, elapsed, save_error, against_ai),
                            )

                        result = show_capture_result(
                            gui, tuple(names[i] for i in race.winners), elapsed, save_error,
                            draw_capture=draw_capture if env.captured else None,
                            step_limit=None if env.captured else env.turn_limit,
                            scorecard=race.scorecard() if against_ai else None, player_name=names[0],
                            difficulty=difficulty,
                        )
                        if result == "restart":
                            break
                        return result
                if gui.is_astar_active() and not race.finished:
                    gui.astar_path = race.player_radar()

            gui.render(race.tracker.get_belief_grid(), f"{difficulty} / {mode}",
                       race=(race, names, elapsed, save_error, against_ai))
            gui.clock.tick(30)
