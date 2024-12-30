
import math
from queue import PriorityQueue

class Node:
    def __init__(self, state, cost, parent=None):
        self.state = state  # Tuple (x, y)
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

def ufs(map, start, goal, include_diagonal_movement):
    diagonal_cost = math.sqrt(2)
    open_set = PriorityQueue()
    open_set.put((0, Node(start, 0, None)))
    visited = set()
    path = []
    updated_map = [row[:] for row in map]  # Copy the map to update it

    while not open_set.empty():
        current_cost, current_node = open_set.get()

        if current_node.state in visited:
            continue
        visited.add(current_node.state)

        x, y = current_node.state

        if map[x][y] != 1 and map[x][y] != 2:  # Mark visited nodes
            updated_map[x][y] = 3

        if current_node.state == goal:  # Goal reached
            node_path = current_node
            while node_path:
                path.insert(0, node_path)
                x, y = node_path.state
                if map[x][y] != 1 and map[x][y] != 2:
                    updated_map[x][y] = 4
                node_path = node_path.parent
            return path

        # Define movement directions
        directions = [
            (-1, 0),  # up
            (0, 1),   # right
            (1, 0),   # down
            (0, -1)   # left
        ]

        if include_diagonal_movement:
            directions += [
                (-1, 1),   # up-right
                (1, 1),    # down-right
                (1, -1),   # down-left
                (-1, -1)   # up-left
            ]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(map) and 0 <= ny < len(map[0]) and (nx, ny) not in visited:
                if map[nx][ny] == 0 or map[nx][ny] == 2:  # Check if walkable
                    visited.add((nx, ny))
                    move_cost = diagonal_cost if abs(dx) + abs(dy) == 2 else 1
                    new_cost = current_node.cost + move_cost
                    next_node = Node((nx, ny), new_cost, current_node)
                    open_set.put((new_cost, next_node))
                    updated_map[nx][ny] = 5  # Mark as in open set

    return None