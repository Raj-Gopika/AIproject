import time
import psutil
import math
import queue

def ucs_graph_search(program, include_diagonal_movement=False):
    # Note down method start time
    start_time = time.time()

    # Note down memory before the process
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

    diagonal_cost = math.sqrt(2)
    map = program.map
    start, goal = None, None

    # Locate the start and goal points
    for i in range(program.MapSize):
        for j in range(program.MapSize):
            if map[i][j] == 1:
                start = (i, j)
            elif map[i][j] == 2:
                goal = (i, j)

    if not start or not goal:
        print("Start or goal not found!")
        return [], set()

    open_set = queue.PriorityQueue()  # Create a PriorityQueue
    open_set.put((0, start, []))  # (cost, state, path)
    visited = set()
    unvisited = set()
    total_nodes_visited = 0
    move_cost = 0

    # Define movement directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    if include_diagonal_movement:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonals

    while not open_set.empty():
        cost, current, path = open_set.get()

        if current in visited:
            continue

        visited.add(current)
        total_nodes_visited += 1

        # Check if the goal is reached
        if current == goal:
            for _, position, _ in list(open_set.queue):  # Iterate over the priority queue's elements
                unvisited.add(position)
            end_time = time.time()  # Measure end time to calculate execution time
            print("Execution Time: {:.6f} seconds".format(end_time - start_time))
            # Memory usage after the function
            memory_after = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
            print(f"Memory Used: {memory_after - memory_before:.4f} MB")
            print("Total nodes visited: ", total_nodes_visited)
            print("Total cost of execution: ", move_cost)
            return path, visited, unvisited

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if (
                0 <= neighbor[0] < program.MapSize
                and 0 <= neighbor[1] < program.MapSize
                and map[neighbor[0]][neighbor[1]] != 99
                and neighbor not in visited
            ):
                move_cost += diagonal_cost if abs(dr) + abs(dc) == 2 else 1  # Adjust cost for diagonal
                open_set.put((cost + move_cost, neighbor, path + [current]))

    print("No path found!")
    return [], visited
