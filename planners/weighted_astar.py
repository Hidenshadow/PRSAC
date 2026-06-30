"""Weighted-cost A* planner for 2D grids."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np


GridCell = tuple[int, int]


def _in_bounds(cell: GridCell, shape: tuple[int, int]) -> bool:
    row, col = cell
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _is_valid_cell(cell: GridCell, cost_map: np.ndarray, obstacle_mask: np.ndarray) -> bool:
    if not _in_bounds(cell, cost_map.shape):
        return False
    row, col = cell
    return bool(np.isfinite(cost_map[row, col])) and not bool(obstacle_mask[row, col])


def _neighbors(allow_diagonal: bool) -> Iterable[tuple[int, int, float]]:
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if allow_diagonal:
        diag = math.sqrt(2.0)
        moves.extend([(-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)])
    return moves


def _heuristic(cell: GridCell, goal: GridCell, min_step_cost: float) -> float:
    # Euclidean distance scaled by the minimum positive traversal cost keeps the
    # heuristic conservative when normalized layers contain near-zero values.
    return math.hypot(cell[0] - goal[0], cell[1] - goal[1]) * min_step_cost


def _reconstruct_path(came_from: dict[GridCell, GridCell], goal: GridCell) -> list[GridCell]:
    path = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def weighted_astar(
    cost_map: np.ndarray,
    obstacle_mask: np.ndarray,
    start: GridCell,
    goal: GridCell,
    allow_diagonal: bool = True,
) -> list[GridCell] | None:
    """Plan a path over a weighted 2D cost map.

    Args:
        cost_map: Non-negative traversal cost for each grid cell.
        obstacle_mask: Boolean mask where True cells are blocked.
        start: Start cell as (row, col).
        goal: Goal cell as (row, col).
        allow_diagonal: Use 8-connected motion when True, otherwise 4-connected.

    Returns:
        A list of (row, col) cells from start to goal, or None if no path exists.
    """

    cost_map = np.asarray(cost_map, dtype=np.float32)
    obstacle_mask = np.asarray(obstacle_mask, dtype=bool)

    if cost_map.ndim != 2 or obstacle_mask.shape != cost_map.shape:
        return None

    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    if not _is_valid_cell(start, cost_map, obstacle_mask):
        return None
    if not _is_valid_cell(goal, cost_map, obstacle_mask):
        return None
    if start == goal:
        return [start]

    free_costs = cost_map[~obstacle_mask]
    finite_positive = free_costs[np.isfinite(free_costs) & (free_costs > 0.0)]
    min_step_cost = float(finite_positive.min()) if finite_positive.size else 1e-6
    min_step_cost = max(min_step_cost, 1e-6)

    open_heap: list[tuple[float, float, GridCell]] = []
    heapq.heappush(open_heap, (_heuristic(start, goal, min_step_cost), 0.0, start))

    came_from: dict[GridCell, GridCell] = {}
    g_score: dict[GridCell, float] = {start: 0.0}
    closed: set[GridCell] = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct_path(came_from, goal)

        closed.add(current)
        for delta_row, delta_col, move_distance in _neighbors(allow_diagonal):
            neighbor = (current[0] + delta_row, current[1] + delta_col)
            if not _is_valid_cell(neighbor, cost_map, obstacle_mask):
                continue
            if neighbor in closed:
                continue

            neighbor_cost = max(float(cost_map[neighbor]), 0.0)
            tentative_g = current_g + move_distance * neighbor_cost
            if tentative_g >= g_score.get(neighbor, math.inf):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            f_score = tentative_g + _heuristic(neighbor, goal, min_step_cost)
            heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return None
