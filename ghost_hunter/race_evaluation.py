"""Measure the race rival against a player following the available map or radar at equal speed."""

import argparse
import csv
import sys
from collections import Counter

from config import ASTAR_HINT_COOLDOWN_SEC, ASTAR_HINT_DURATION_SEC, DIFFICULTY_CONFIGS, HUMAN_MOVE_INTERVAL_SEC
from ghost_hunter.competitive import ChaseRound
from ghost_hunter.pathfinding import astar


def simulate_race(difficulty, seed, *, player_policy="radar"):
    race = ChaseRound(difficulty, seed, against_ai=True)
    env = race.env
    frame = 0
    next_ghost = race.ghost_interval
    radar_cycle = ASTAR_HINT_DURATION_SEC + ASTAR_HINT_COOLDOWN_SEC
    while not race.finished:
        now = frame / 30
        race.update_glimpse(now)
        player_move = None
        if race.capture_ms[0] is None and race.player_ready(now):
            position = race.positions[0]
            # Only the scripted human gets the true path, while its legal radar is active.
            radar_active = player_policy == "radar" and now % radar_cycle < ASTAR_HINT_DURATION_SEC
            target = env.true_ghost_pos if radar_active else race.tracker.get_map_estimate()
            path = astar(position, target, env.walls, (env.height, env.width))
            step = path[min(1, len(path)-1)]
            player_move = step[0] - position[0], step[1] - position[1]
        actions = race.scheduled_moves(now, (player_move, None))
        ghost_due = now >= next_ghost
        if ghost_due or any(action is not None for action in actions):
            race.advance(actions, ghost_due, elapsed=now)
        if ghost_due:
            next_ghost = now + race.ghost_interval
        frame += 1
    return {(0,): "player", (1,): "ai", (0, 1): "tie", (): "timeout"}[race.winners]


def evaluate(trials, first_seed, stream, *, player_policy="radar"):
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("difficulty", "first_seed", "races", "ai_wins", "player_wins", "ties", "timeouts",
                     "player_policy", "move_interval_seconds"))
    for difficulty in DIFFICULTY_CONFIGS:
        outcomes = Counter(simulate_race(difficulty, seed, player_policy=player_policy)
                           for seed in range(first_seed, first_seed + trials))
        writer.writerow((difficulty, first_seed, trials, *(outcomes[k] for k in ("ai", "player", "tie", "timeout")),
                         player_policy, HUMAN_MOVE_INTERVAL_SEC))
        stream.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument("--player", choices=("radar", "heatmap"), default="radar",
                        help="Follow the player's public A* radar whenever charged, or use the belief map throughout.")
    parser.add_argument("--output", help="Optional CSV path. Defaults to standard output.")
    args = parser.parse_args()
    if args.trials < 1 or args.first_seed < 0:
        parser.error("trials must be positive and first-seed must be nonnegative")
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as stream:
            evaluate(args.trials, args.first_seed, stream, player_policy=args.player)
    else:
        evaluate(args.trials, args.first_seed, sys.stdout, player_policy=args.player)
