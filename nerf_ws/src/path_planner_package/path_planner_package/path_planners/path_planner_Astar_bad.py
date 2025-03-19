import numpy as np
import heapq

class Node:
    def __init__(self, x, y, cost, parent):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.cost < other.cost
    
def huristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star_bad(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, Node(start[0], start[1], 0, None))

    visited = set()
    while open_set:
        current = heapq.heappop(open_set)

        if (current.x, current.y) == goal:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]
        
        if (current.x, current.y) in visited:
            continue
        
        visited.add((current.x, current.y))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = current.x + dx, current.y + dy
            if 0 <= x and x < rows and 0 <= y and y < cols and grid[x, y] == 0:
                new_cost = current.cost + 1 + huristic((x, y), goal)
                heapq.heappush(open_set, Node(x, y, new_cost, current))

    return None