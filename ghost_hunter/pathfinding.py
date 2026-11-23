"""
A* Pathfinding Algorithm with Tie-Breaking and Directional Momentum Penalty.
Finds shortest obstacle-avoiding path on the grid while mitigating
oscillation risks (Proposal Risk 2).
"""

from typing import Optional
import heapq
from config import GRID_WIDTH, GRID_HEIGHT, ACTION_STAY


def heuristic(pos: tuple[int, int], goal: tuple[int, int]) -> int:
    """Standard Manhattan distance admissible heuristic for 4-directional grid."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    walls: set[tuple[int, int]],
    grid_dims: tuple[int, int] = (GRID_HEIGHT, GRID_WIDTH),
    last_action: Optional[tuple[int, int]] = None,
    momentum_weight: float = 0.05
) -> list[tuple[int, int]]:
    """
    Computes shortest path from start to goal avoiding walls.
    
    Includes a slight directional momentum penalty:
    If a move continues in the same direction as last_action, it is preferred among equal priorities,
    breaking symmetric ties and preventing the agent from oscillating back-and-forth.

    Returns:
        path: List of coordinates from start to goal: [start, next_cell, ..., goal].
              If start == goal, returns [start].
              If unreachable, returns [start].
    """
    if start == goal:
        return [start]
    
    height, width = grid_dims

    
    if not (0 <= goal[0] < height and 0 <= goal[1] < width) or goal in walls:
        # If goal is in a wall, return in-place
        return [start]

    # Order by total estimated steps, distance remaining, then momentum.
    counter = 0
    open_heap = []
    start_h = heuristic(start, goal)
    heapq.heappush(open_heap, (start_h, start_h, 0.0, counter, 0, start, last_action))
    
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_scores: dict[tuple[int, int], float] = {start: 0.0}
    closed_set: set[tuple[int, int]] = set()

    while open_heap:
        _, _, _, _, g, curr, prev_act = heapq.heappop(open_heap)
        
        if curr == goal:
            # Reconstruct path
            path = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()
            return path
            
        if curr in closed_set:
            continue
        closed_set.add(curr)
        
        r, c = curr
        # 4 cardinal directions
        candidates = [
            ((-1, 0), (r - 1, c)),
            ((1, 0), (r + 1, c)),
            ((0, -1), (r, c - 1)),
            ((0, 1), (r, c + 1))
        ]
        
        for act, neighbor in candidates:
            if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                continue
            if neighbor in walls or neighbor in closed_set:
                continue
            
            tentative_g = g + 1
            if tentative_g < g_scores.get(neighbor, float('inf')):
                came_from[neighbor] = curr
                g_scores[neighbor] = tentative_g
                h = heuristic(neighbor, goal)
                # Momentum only breaks equal-priority ties. Every movement costs one turn.
                momentum = -momentum_weight if act == prev_act else 0.0
                counter += 1
                heapq.heappush(open_heap, (tentative_g + h, h, momentum, counter,
                                           tentative_g, neighbor, act))

    # If no path found (e.g. enclosed), stay at start
    return [start]
