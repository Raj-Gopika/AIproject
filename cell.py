def ucs_tree_search(program, include_diagonal_movement=False):
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

    # Priority queue: (cost, current_position, path)
    open_set = PriorityQueue()
    open_set.put((0, start, [start]))
    visited = set()

    # Define movement directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    if include_diagonal_movement:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonals

    while not open_set.empty():
        cost, current, path = open_set.get()

        if current in visited:
            continue

        visited.add(current)

        # Check if the goal is reached
        if current == goal:
            return path, visited

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if (
                0 <= neighbor[0] < program.MapSize
                and 0 <= neighbor[1] < program.MapSize
                and map[neighbor[0]][neighbor[1]] != 99
                and neighbor not in visited
            ):
                new_cost = cost + (1 if abs(dr) + abs(dc) == 1 else math.sqrt(2))  # Adjust cost for diagonal
                open_set.put((new_cost, neighbor, path + [neighbor]))

    print("No path found!")
    return [], visited
