import numpy as np
import heapq
import random
import math

class Node:
    def __init__(self, x, y, cost=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

# Get distance between two nodes
def distance(node1, node2):
    return np.linalg.norm(np.array([node1.x, node1.y]) - np.array([node2.x, node2.y]))

# Get nearest node to (new) random node
def get_nearest_node(nodes, random_node):
    nearest = min(nodes, key=lambda node: distance(node, random_node))
    return nearest

# Create new node based on nearest node and random node
def get_new_node(nearest_node, random_node, step_size):
    theta = math.atan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)
    new_x = nearest_node.x + step_size * math.cos(theta)
    new_y = nearest_node.y + step_size * math.sin(theta)
    return Node(new_x, new_y)

# Check if new node is collision free
def is_collision_free(node, grid):
    x, y = int(node.x), int(node.y)
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 0

# Get all near nodes within a certain radius
def get_near_nodes(nodes, new_node, radius):
    near_nodes = [node for node in nodes if distance(node, new_node) < radius]
    return near_nodes

# Rewire the tree for more optimal path
def rewire(nodes, new_node, near_nodes): #Rewire never directly uses nodes?
    for near_node in near_nodes:
        new_cost = new_node.cost + distance(new_node, near_node)
        if new_cost < near_node.cost:
            near_node.parent = new_node
            near_node.cost = new_cost


# RRT* Path Planner
# max_itet --> Max number of iterations
# step_size --> Distance to move towards random node
# goal_sample_rate --> Probability of sampling goal node, used to bias path towards node. -- Avoids wandering
# radius --> Radius of near nodes to rewire

def rrt_star(grid, start, goal, max_iter=10000, step_size=1.0, goal_sample_rate=0.3, radius=3.0):
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    nodes = [start_node]

    for _ in range(max_iter):
        # If rand # < sample rate () then use goal node
        if random.random() < goal_sample_rate:
            random_node = goal_node
        else:
            # Otherwise, sample random node within grid
            random_node = Node(random.uniform(0, grid.shape[0]), random.uniform(0, grid.shape[1]))

        # Get nearest node and create new node
        nearest_node = get_nearest_node(nodes, random_node)
        new_node = get_new_node(nearest_node, random_node, step_size)

        # If new node is collision free, add to tree
        if is_collision_free(new_node, grid):
            near_nodes = get_near_nodes(nodes, new_node, radius)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + distance(nearest_node, new_node)
            nodes.append(new_node)
            rewire(nodes, new_node, near_nodes)
            
            # If dist to goal smaller then step size, add goal node to tree
            if distance(new_node, goal_node) < step_size:
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + distance(new_node, goal_node)
                nodes.append(goal_node)
                break

    # Build path backwards from goal node 
    path = []
    node = goal_node

    while node:
        path.append([node.x, node.y])
        node = node.parent
    
    # Reverse list to get path from start to goal
    return path[::-1] if path else None