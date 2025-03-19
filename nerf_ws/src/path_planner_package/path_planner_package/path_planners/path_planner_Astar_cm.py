import numpy as np
import heapq
from math import sqrt
from typing import List, Tuple, Dict

def create_node(position: Tuple[int, int], g: float = float('inf'), 
                h: float = 0.0, parent: Dict = None) -> Dict:
    """
    Create a node for the A* algorithm.
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two points.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get valid neighboring cells that are not obstacles.
    In this costmap, a cell with value 100 is considered an obstacle.
    Position is (x,y) but grid is accessed as [y,x].
    """
    x, y = position
    rows, cols = grid.shape
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                # Skip if neighbor cell is an obstacle
                if grid[ny, nx] == 100:
                    continue
                # For diagonal moves, check that adjacent cardinal cells are not obstacles.
                if dx != 0 and dy != 0:
                    if grid[y + dy, x] == 100 or grid[y, x + dx] == 100:
                        continue
                neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = []
    current = goal_node
    while current is not None:
        path.append(current['position'])
        current = current['parent']
    return path[::-1]  # Reverse to get path from start to goal

def a_star_cm(grid: np.ndarray, start: Tuple[int, int], 
              goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find the optimal path using the A* algorithm.
    grid: 2D numpy array of cost values (0 = free, gradient values for near obstacles, 100 = obstacle)
    start and goal: grid indices (x, y)
    Returns a list of grid positions representing the path.
    """
    start_node = create_node(position=start, g=0, h=calculate_heuristic(start, goal))
    open_list = [(start_node['f'], start)]  # Priority queue of (f, position)
    open_dict = {start: start_node}
    closed_set = set()
    
    while open_list:
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]
        
        if current_pos == goal:
            return reconstruct_path(current_node)
            
        closed_set.add(current_pos)
        
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            if neighbor_pos in closed_set:
                continue
        
            # Incorporate both the movement cost (Euclidean distance) and the costmap penalty.
            move_cost = calculate_heuristic(current_pos, neighbor_pos)
            cell_penalty = float(grid[neighbor_pos[1], neighbor_pos[0]])
            tentative_g = current_node['g'] + move_cost + cell_penalty
            
            if neighbor_pos not in open_dict:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node
                
    return []  # No path found