import heapq
import math


def ucs_tree_search(map, start, goal, include_diagonal_movement):
    diagonal_cost = math.sqrt(2)
    # Priority queue: (cost, (x, y), path)
    open_set = []
    heapq.heappush(open_set, (0, start, []))  # (cost, state, path)
    updated_map = [row[:] for row in map]  # Copy the map to update it

    while open_set:
        # Get the node with the lowest cost
        cost, current, path = heapq.heappop(open_set)
        x, y = current

        # Mark visited on visualization map
        if map[x][y] != 1 and map[x][y] != 2:
            updated_map[x][y] = 3  # Mark as visited

        # Goal check
        if current == goal:
            path.append(current)  # Add goal to path
            for px, py in path:
                if map[px][py] != 1 and map[px][py] != 2:
                    updated_map[px][py] = 4  # Mark final path
            return path, updated_map

        # Define movement directions
        directions = [
            (-1, 0),  # up
            (0, 1),  # right
            (1, 0),  # down
            (0, -1)  # left
        ]

        if include_diagonal_movement:
            directions += [
                (-1, 1),  # up-right
                (1, 1),  # down-right
                (1, -1),  # down-left
                (-1, -1)  # up-left
            ]

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(map) and 0 <= ny < len(map[0]):  # Stay within bounds
                if map[nx][ny] == 0 or map[nx][ny] == 2:  # Walkable or goal
                    move_cost = diagonal_cost if abs(dx) + abs(dy) == 2 else 1
                    heapq.heappush(open_set, (cost + move_cost, (nx, ny), path + [current]))
                    updated_map[nx][ny] = 5  # Mark as in open set

    return None, updated_map  # No path found
