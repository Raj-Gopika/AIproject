#need to get start and end position of the grid
#also need to get any position on the grid to return the number its assossiated with
#define an agent that can move through the grid
# use bfs to calculate path
# show the bfs path in terminal as well as in a different color in GUI
#if time use animate.

'''
[0,1]
[1,0]
[0,-1]
[-1,0]
'''
# writing bfs class insoed program class or separate?
#can add separate py file if needed.


#BFS Graph Search
"""
def bfs_graph(program):
    """

    """
    BFS graph search to find the path from start to goal in the grid.
    :param program: Instance of Program class with the map, start, and goal points.
    :return: List of tuples representing the path from start to goal.
    """

    """
    grid = program.map
    rows, cols = program.MapSize, program.MapSize
    start, goal = None, None

    # Locate start and goal points
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                start = (r, c)
            elif grid[r][c] == 2:
                goal = (r, c)

    if not start or not goal:
        print("Start or goal not found!")
        return []

    # BFS initialization
    queue = [[start]]
    visited = set()

    while queue:
        path = queue.pop(0)
        current = path[-1]

        if current in visited:
            continue
        visited.add(current)

        # Check if goal is reached
        if current == goal:
            return path

        # Explore neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dr, current[1] + dc)

            # Check bounds and obstacle
            if (
                0 <= neighbor[0] < rows
                and 0 <= neighbor[1] < cols
                and grid[neighbor[0]][neighbor[1]] != 99
                and neighbor not in visited
            ):
                queue.append(path + [neighbor])

    return []  # Return empty if no path is found
    """

#BFS Tree Search
"""
def bfs_tree(program):
"""

    """
    BFS tree search to find the path from start to goal in the grid.
    :param program: Instance of Program class with the map, start, and goal points.
    :return: List of tuples representing the path from start to goal.
    """

    """
    grid = program.map
    rows, cols = program.MapSize, program.MapSize
    start, goal = None, None

    # Locate start and goal points
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                start = (r, c)
            elif grid[r][c] == 2:
                goal = (r, c)

    if not start or not goal:
        print("Start or goal not found!")
        return []

    # BFS initialization with a tree structure
    queue = [start]
    parents = {start: None}  # Track parents to reconstruct the path

    while queue:
        current = queue.pop(0)

        # Check if goal is reached
        if current == goal:
            # Reconstruct path from goal to start
            path = []
            while current:
                path.append(current)
                current = parents[current]
            return path[::-1]  # Reverse to get path from start to goal

        # Explore neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dr, current[1] + dc)

            # Check bounds and obstacle
            if (
                0 <= neighbor[0] < rows
                and 0 <= neighbor[1] < cols
                and grid[neighbor[0]][neighbor[1]] != 99
                and neighbor not in parents
            ):
                queue.append(neighbor)
                parents[neighbor] = current

    return []  # Return empty if no path is found


"""

#Function to run on main
"""
if __name__ == "__main__":
    prg = Program()
    prg.create_map_with_path()

    print("Running BFS Graph Search...")
    path_graph = bfs_graph(prg)
    print("Path (Graph Search):", path_graph)

    print("Running BFS Tree Search...")
    path_tree = bfs_tree(prg)
    print("Path (Tree Search):", path_tree)

    if path_graph:
        print("Displaying path from Graph Search...")
        for r, c in path_graph:
            if prg.map[r][c] == 0:  # Mark path in the grid
                prg.map[r][c] = 5
        prg.display_map_gui()

    if path_tree and path_graph != path_tree:
        print("Displaying path from Tree Search...")
        for r, c in path_tree:
            if prg.map[r][c] == 0:  # Mark path in the grid
                prg.map[r][c] = 5
        prg.display_map_gui()

"""