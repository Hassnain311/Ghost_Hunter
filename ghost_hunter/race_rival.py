"""Private A* pursuit policy for the live race. Only a one-cell action leaves the planner."""

from collections import deque

import numpy as np

from config import ACTION_STAY, GHOST_ACTIONS, HUMAN_MOVE_INTERVAL_SEC
from ghost_hunter.pathfinding import astar
from ghost_hunter.sensor_model import likelihood_by_distance


class RaceRival:
    def __init__(self, difficulty, tracker, ghost_interval, seed):
        self.difficulty = difficulty
        self.tracker = tracker
        self.ghost_interval = ghost_interval
        self.rng = np.random.default_rng([seed, 2])
        self._goal = None
        self._moves_left = 0
        self._last_action = None
        self._distances = {}

    def _path(self, start, goal):
        return astar(start, goal, self.tracker.walls, self.tracker.grid_dims, self._last_action)

    def _single_ping_goal(self, position, other, reading):
        tracker = self.tracker
        distances = tracker.distances_from(position)
        likelihood = likelihood_by_distance(min(reading, int(distances.max())),
                                            tracker.sigma, sum(tracker.grid_dims) - 2)
        weights = likelihood[distances]
        weights[[tracker.cell_to_idx[position], tracker.cell_to_idx[other]]] = 0
        if weights.sum() == 0:
            possible = [i for i, cell in enumerate(tracker.cell_list) if cell not in (position, other)]
            return tracker.cell_list[min(possible, key=lambda i: abs(distances[i] - reading))]
        index = self.rng.choice(tracker.num_states, p=weights / weights.sum())
        return tracker.cell_list[index]

    def _maze_distances(self, source):
        if source not in self._distances:
            lengths = {source: 0}
            queue = deque([source])
            while queue:
                cell = queue.popleft()
                for dr, dc in GHOST_ACTIONS[:4]:
                    neighbor = cell[0] + dr, cell[1] + dc
                    if neighbor in self.tracker.free_cells and neighbor not in lengths:
                        lengths[neighbor] = lengths[cell] + 1
                        queue.append(neighbor)
            unreachable = self.tracker.num_states + 1
            self._distances[source] = np.array([lengths.get(cell, unreachable) for cell in self.tracker.cell_list])
        return self._distances[source]

    def _scouting_value(self, step, belief, contestable):
        # A good first step is close by maze distance to several hypotheses and gives
        # different possible ranges for the next ping when those hypotheses disagree.
        ranges = np.bincount(self.tracker.distances_from(step), weights=belief)
        nonzero = ranges[ranges > 0]
        range_entropy = -nonzero @ np.log2(nonzero)
        return .35 * range_entropy - .6 * (contestable @ self._maze_distances(step))

    def _candidate_paths(self, position, priority, distances):
        tracker = self.tracker
        candidates = np.argsort(-priority, kind="stable")[:min(6, len(priority) - 1)]
        for index in candidates:
            yield self._path(position, tracker.cell_list[index])
        if self.difficulty == "Extreme":
            # Consider alternative first legs, including equally short detours with a better sensor angle.
            for dr, dc in GHOST_ACTIONS[:4]:
                step = position[0] + dr, position[1] + dc
                if step in tracker.free_cells:
                    closer = self._maze_distances(step) < distances
                    index = int(np.argmax(np.where(closer, priority, -np.inf)))
                    yield [position, *self._path(step, tracker.cell_list[index])]

    def _intercept_path(self, position, other):
        tracker = self.tracker
        belief = tracker.belief_vec
        distances = tracker.distances_from(position)
        predicted = belief
        priority = belief / np.sqrt(1 + distances)
        if self.difficulty == "Extreme":
            distances = self._maze_distances(position)
            travel_seconds = float(belief @ distances) * HUMAN_MOVE_INTERVAL_SEC
            ticks = min(4, int(travel_seconds / self.ghost_interval))
            if ticks:
                predicted = tracker.forecast(ticks, position, other)
            # Favor plausible interception cells the rival can reach before the other hunter.
            lead = self._maze_distances(other) - distances
            contest = .15 + .85 / (1 + np.exp(-(lead + 2) / 3))
            contestable = predicted * contest
            contestable /= contestable.sum()
            priority = np.maximum(belief, predicted) * contest / np.sqrt(1 + distances)

        # Shortlist likely cells, then evaluate the actual maze routes through them.
        # Close captures retain their current probability even when the longer chase is uncertain.
        priority[tracker.cell_to_idx[position]] = -1
        best_path, best_score = [position], -float("inf")
        scouting = {}
        for path in self._candidate_paths(position, priority, distances):
            route = [tracker.cell_to_idx[cell] for cell in path[1:]]
            if not route:
                continue
            arrival = np.arange(len(route)) * HUMAN_MOVE_INTERVAL_SEC
            future = np.minimum(1, arrival / self.ghost_interval)
            chances = (1 - future) * belief[route] + future * predicted[route]
            score = np.sum(chances / np.arange(1, len(route) + 1)) / np.sqrt(len(route))
            if self.difficulty == "Extreme":
                first_step = path[1]
                if first_step not in scouting:
                    scouting[first_step] = self._scouting_value(first_step, belief, contestable)
                score = 3 * score + scouting[first_step]
            if score > best_score:
                best_score, best_path = score, path
        return best_path

    def choose(self, position, other, reading):
        if self.difficulty in ("Beginner", "Intermediate"):
            if self._moves_left == 0 or self._goal == position:
                if self.difficulty == "Beginner":
                    self._goal = self._single_ping_goal(position, other, reading)
                    self._moves_left = 5
                else:
                    self._goal = self.tracker.get_map_estimate()
                    self._moves_left = 2
            path = self._path(position, self._goal)
            self._moves_left -= 1
        else:
            path = self._intercept_path(position, other)
        if len(path) < 2:
            return ACTION_STAY
        nr, nc = path[1]
        self._last_action = nr - position[0], nc - position[1]
        return self._last_action
