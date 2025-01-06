import time
import psutil
import random
import heapq
import tkinter as tk
import math


#Generate Environment class
class Program:
    def __init__(self):
        self.map = []
        self.MapSize = 0

    def create_map_with_path(self):
        self.MapSize = random.randint(12, 16)
        start = (0, 0)
        goal = (0, 0)

        #self.map = [[0] * self.MapSize for _ in range(self.MapSize)]
        self.map = [[0 for _ in range(self.MapSize)] for _ in range(self.MapSize)]

        self.set_walls()

        # Define start and goal, ensuring they are not against the wall
        while True:
            start = (random.randint(0, self.MapSize // 3), random.randint(0, self.MapSize // 3))
            if self.map[start[0]][start[1]] != 99:
                break

        while True:
            row = random.randint((self.MapSize // 3) * 2, self.MapSize - 1)
            col = random.randint((self.MapSize // 3) * 2, self.MapSize - 1)
            goal = (row, col)
            if self.map[goal[0]][goal[1]] != 99:
                break

        self.map[start[0]][start[1]] = 1
        self.map[goal[0]][goal[1]] = 2

        trail = self.create_trail(start, goal)

        num_obstacles = random.randint(1,self.MapSize // 2)
        for _ in range(num_obstacles):
            self.add_obstacles(trail)

    def add_obstacles(self,trail):
        trail_set = set(trail)
        random_row = random.randint(1, self.MapSize - 2)
        random_col = random.randint(1, self.MapSize - 2)

        choice = random.choice(["Square", "Line", "T_Line"])
        if choice == "Square":
            self.add_square(random_row, random_col, trail_set)
        elif choice == "Line":
            self.add_line(random_row, random_col, trail_set)
        elif choice == "T_Line":
            self.add_t_line(random_row, random_col, trail_set)


    def add_square(self, random_row, random_col, trail_set):
        for i in range(9):
            r = i // 3
            c = i % 3
            if self.check_bounds_and_start_end(random_row + r, random_col + c) and (random_row + r, random_col + c) not in trail_set:
                self.map[random_row + r][random_col + c] = 99

    def add_line(self, random_row, random_col, trail_set):
        direction = random.randint(1, 4)
        length = self.MapSize // 3

        for i in range(length):
            if direction == 1:  # Vertical up
                r = random_row - i
                if self.check_bounds_and_start_end(r, random_col) and (r, random_col) not in trail_set:
                    self.map[r][random_col] = 99
            elif direction == 2:  # Horizontal left
                c = random_col - i
                if self.check_bounds_and_start_end(random_row, c) and (random_row, c) not in trail_set:
                    self.map[random_row][c] = 99
            elif direction == 3:  # Vertical down
                r = random_row + i
                if self.check_bounds_and_start_end(r, random_col) and (r, random_col) not in trail_set:
                    self.map[r][random_col] = 99
            elif direction == 4:  # Horizontal right
                c = random_col + i
                if self.check_bounds_and_start_end(random_row, c) and (random_row, c) not in trail_set:
                    self.map[random_row][c] = 99

    def add_t_line(self, random_row, random_col, trail_set):
        hor_v = random.randint(1, 2)
        for i in range(10):
            if hor_v == 1:  # Horizontal base, vertical stem
                if i < 5:  # Vertical stem
                    if self.check_bounds_and_start_end(random_row + i, random_col) and (random_row + i, random_col) not in trail_set:
                        self.map[random_row + i][random_col] = 99
                else:  # Horizontal base
                    if self.check_bounds_and_start_end(random_row + 2, random_col + i - 5) and (random_row + 2, random_col + i - 5) not in trail_set:
                        self.map[random_row + 2][random_col + i - 5] = 99
            else:  # Vertical base, horizontal stem
                if i < 5:  # Horizontal stem
                    if self.check_bounds_and_start_end(random_row, random_col + i) and (random_row + i, random_col) not in trail_set:
                        self.map[random_row][random_col + i] = 99
                else:  # Vertical base
                    if self.check_bounds_and_start_end(random_row + i - 5, random_col + 2) and (random_row + 2, random_col + i - 5) not in trail_set:
                        self.map[random_row + i - 5][random_col + 2] = 99

    def check_bounds_and_start_end(self, x, y):
        if x < 0 or y < 0 or x >= self.MapSize or y >= self.MapSize:
            return False
        if self.map[x][y] in [1, 2]:
            return False
        return True
    '''
    def create_trail(self, start_point, goal):
        trail = []
        current = start_point

        # Define movement directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while current != goal:
            r, c = current
            best_move = None
            min_distance = float('inf')

            # Check all possible moves
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc

                if self.check_bounds_and_start_end(new_r, new_c):
                    # Avoid adjacent obstacles
                    has_adjacent_obstacle = any(
                        0 <= new_r + adj_dr < self.MapSize and
                        0 <= new_c + adj_dc < self.MapSize and
                        self.map[new_r + adj_dr][new_c + adj_dc] == 99
                        for adj_dr, adj_dc in directions
                    )
                    if has_adjacent_obstacle:
                        continue

                    # Calculate distance to the goal
                    distance = abs(new_r - goal[0]) + abs(new_c - goal[1])

                    if distance < min_distance:
                        min_distance = distance
                        best_move = (new_r, new_c)

            if best_move:
                self.map[best_move[0]][best_move[1]] = 6  # Mark as part of the trail
                trail.append(best_move)
                current = best_move
            else:
                # If no valid moves are possible, backtrack
                print("Trail creation failed, backtracking...")
                break

        return trail
    
    '''
    def create_trail(self, start_point, goal):
        c = goal[1] - 1
        r = goal[0]
        trail = []

        # Move horizontally towards the start column
        while c != start_point[1]:
            self.map[goal[0]][c] = 6
            trail.append((goal[0], c))
            c += 1 if start_point[1] > c else -1

        # Move vertically towards the start row
        while r != start_point[0]:
            self.map[r][c] = 6
            trail.append((r, c))
            r += 1 if start_point[0] > r else -1

        return trail

    def set_walls(self):
        for i in range(self.MapSize):
            for j in range(self.MapSize):
                if i == 0 or i == self.MapSize - 1 or j == 0 or j == self.MapSize - 1:
                    self.map[i][j] = 99

    def display_map(self):
        for row in self.map:
            print(" ".join(str(cell).rjust(3) for cell in row))

#Create GUI for displaying Map
class MapVisualizer:
    def __init__(self, program):
        self.program = program
        self.root = tk.Tk()
        self.root.title("Map Visualization")
        self.canvas = None

        # Dropdown menu options
        self.search_method = tk.StringVar(self.root)
        self.search_method.set("None")  # Default value
        options = ["None", "BFS Graph Simple", "BFS Graph Diagonal", "BFS Tree Simple",
                   "BFS Tree Diagonal", "A* Graph", "A* Tree", "UCS Graph Simple",  "UCS Graph Diagonal",
                   "UCS Tree Simple", "UCS Tree Diagonal"]

        dropdown = tk.OptionMenu(self.root, self.search_method, *options, command=self.run_search)
        dropdown.pack(pady=10)

        reset_button = tk.Button(self.root, text="Reset Map", command=self.reset_map)
        reset_button.pack(pady=10)

    def reset_map(self):
        self.program.create_map_with_path()
        self.display_map()

    def run_search(self, choice):
        if choice == "None":
            self.display_map()  # Display the map without any path

        elif choice == "BFS Graph Simple":
            print("\nRunning BFS Graph Search...(Simple)")
            bfs_path_graph,visited_nodes, unvisited_nodes = bfs_graph(self.program, include_diagonal_movement=False)
            print("Path (BFS Graph Search):", bfs_path_graph)

            if bfs_path_graph:
                print("Displaying path from Graph Search...")
                for r, c in bfs_path_graph:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=bfs_path_graph, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

        elif choice == "BFS Graph Diagonal":
            print("\nRunning BFS Graph Search...(Diagonal enabled)")
            bfs_path_graph,visited_nodes, unvisited_nodes = bfs_graph(self.program, include_diagonal_movement=True)
            print("Path (BFS Graph Search):", bfs_path_graph)

            if bfs_path_graph:
                print("Displaying path from Graph Search...")
                for r, c in bfs_path_graph:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=bfs_path_graph, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

        elif choice == "BFS Tree Simple":
            print("\nRunning BFS Tree Search...(Simple)")
            bfs_path_tree, visited_nodes,unvisited_nodes = bfs_tree(self.program, include_diagonal_movement=False)
            print("Path (BFS Tree Search):", bfs_path_tree)

            if bfs_path_tree:
                print("Displaying path from Tree Search...")
                for r, c in bfs_path_tree:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=bfs_path_tree, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

        elif choice == "BFS Tree Diagonal":
            print("\nRunning BFS Tree Search...(Diagonal enabled)")
            bfs_path_tree, visited_nodes,unvisited_nodes = bfs_tree(self.program, include_diagonal_movement=True)
            print("Path (BFS Tree Search):", bfs_path_tree)

            if bfs_path_tree:
                print("Displaying path from Tree Search...")
                for r, c in bfs_path_tree:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=bfs_path_tree, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

        elif choice == "A* Graph":
            print("\nRunning A* Graph Search...")
            astar_path_graph, visited_nodes, unvisited_nodes = a_star_graph(self.program)
            print("Path (A* Graph Search):", astar_path_graph)

            if astar_path_graph:
                for r, c in astar_path_graph:
                    if self.program.map[r][c] == 0:
                        self.program.map[r][c] = 5
                self.display_map(path=astar_path_graph, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

        elif choice == "A* Tree":
            print("\nRunning A* Tree Search...")
            astar_path_tree, visited_nodes, unvisited_nodes = a_star_tree(self.program)
            print("Path (A* Tree Search):", astar_path_tree)

            if astar_path_tree:
                for r, c in astar_path_tree:
                    if self.program.map[r][c] == 0:
                        self.program.map[r][c] = 5
                self.display_map(path=astar_path_tree, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

        elif choice == "UCS Graph Simple":
            print("\nRunning Uniform Cost Graph Search (UCS - Simple)...")
            ucs_path, visited_nodes,unvisited_nodes = ucs_graph_search(self.program, include_diagonal_movement=False)
            print("Path (UCS):", ucs_path)

            if ucs_path:
                for r, c in ucs_path:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=ucs_path, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

        elif choice == "UCS Graph Diagonal":
            print("\nRunning Uniform Cost Graph Search (UCS - Diagonal enabled)...")
            ucs_path, visited_nodes,unvisited_nodes = ucs_graph_search(self.program, include_diagonal_movement=True)
            print("Path (UCS):", ucs_path)

            if ucs_path:
                for r, c in ucs_path:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=ucs_path, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

        elif choice == "UCS Tree Simple":
            print("\nRunning Uniform Cost Tree Search (UCS - Simple)...")
            ucs_path, visited_nodes, unvisited_nodes = ucs_tree_search(self.program, include_diagonal_movement=False)
            print("Path (UCS):", ucs_path)

            if ucs_path:
                for r, c in ucs_path:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=ucs_path, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

        elif choice == "UCS Tree Diagonal":
            print("\nRunning Uniform Cost Tree Search (UCS - Diagonal enabled)...")
            ucs_path, visited_nodes, unvisited_nodes = ucs_tree_search(self.program, include_diagonal_movement=True)
            print("Path (UCS):", ucs_path)

            if ucs_path:
                for r, c in ucs_path:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=ucs_path, visited_nodes=visited_nodes, unvisited_nodes=unvisited_nodes)

    def display_map(self, path=None, visited_nodes=None, unvisited_nodes=None):
        # Destroy the previous canvas
        if self.canvas:
            self.canvas.destroy()

        # Create window
        cell_size = 25  # Size of each cell in pixels
        self.canvas = tk.Canvas(self.root, width=self.program.MapSize * cell_size, height=self.program.MapSize * cell_size)
        self.canvas.pack()

        # Add colors to cells
        colors = {
            0: "white",  # Empty cell
            1: "green",  # Start
            2: "red",  # Goal
            99: "black"  # Wall or obstacle
        }

        path_color = "purple"
        visited_color = 'yellow'
        unvisited_color = "orange"

        # Draw the map
        for i in range(self.program.MapSize):
            for j in range(self.program.MapSize):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                if self.program.map[i][j] == 1:
                    color = "green"
                elif self.program.map[i][j] == 2:
                    color  = "red"
                elif path and (i, j) in path:
                    color = path_color
                elif visited_nodes and (i, j) in visited_nodes:
                    color = visited_color
                elif unvisited_nodes and (i, j) in unvisited_nodes:
                    color = unvisited_color
                else:
                    color = colors.get(self.program.map[i][j], "white")

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

#Search Algorithms
def bfs_graph(program, include_diagonal_movement=False):
    #Note down method start time
    start_time = time.time()

    # Note down memory before the process
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

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
    unvisited = set()
    total_nodes_visited = 0

    # Define movement directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    if include_diagonal_movement:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonals

    while queue:
        path = queue.pop(0)
        current = path[-1]

        if current in visited:
            continue
        visited.add(current)
        total_nodes_visited +=1

        # Check if goal is reached
        if current == goal:
            for remaining_path in queue:
                unvisited.update(remaining_path)

            end_time = time.time()  # `Measure end time to calculate execution time
            print("Execution Time: {:.6f} seconds".format(end_time - start_time))
            # Memory usage after the function
            memory_after = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
            print(f"Memory Used: {memory_after - memory_before:.4f} MB")
            print("Total nodes visited: ", total_nodes_visited)
            return path, visited, unvisited

        # Explore neighbors
        for dr, dc in directions:
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

def bfs_tree(program, include_diagonal_movement=False):
    # Note down method start time
    start_time = time.time()

    # Note down memory before the process
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

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
    unvisited = set()
    total_nodes_visited = 0

    # Define movement directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    if include_diagonal_movement:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonals

    while queue:
        path = queue.pop(0)
        current = path[-1]

        visited.add(current)
        total_nodes_visited += 1

        # Check if goal is reached
        if current == goal:
            for remaining_path in queue:
                unvisited.update(remaining_path)

            end_time = time.time()  # `Measure end time to calculate execution time
            print("Execution Time: {:.6f} seconds".format(end_time - start_time))
            # Memory usage after the function
            memory_after = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
            print(f"Memory Used: {memory_after - memory_before:.4f} MB")
            print("Total nodes visited: ", total_nodes_visited)
            return path, visited, unvisited

        # Explore neighbors
        for dr, dc in directions:
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


def a_star_graph(program):
    # Note down method start time
    start_time = time.time()

    # Note down memory before the process
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

    start = None
    goal = None

    # Locate the start and goal points
    for i in range(program.MapSize):
        for j in range(program.MapSize):
            if program.map[i][j] == 1:
                start = (i, j)
            elif program.map[i][j] == 2:
                goal = (i, j)

    if not start or not goal:
        return [], []

    # Initialize priority queue, cost dictionary, parent tracking, and visited set
    open_set = [(0, start)]  # (priority, node)
    g_cost = {start: 0}  # Cost from start to a node
    came_from = {}  # To reconstruct the path
    visited = set()  # Keeps track of visited nodes
    unvisited = set()
    total_nodes_visited = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        # If the goal is reached, reconstruct the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            for remaining_path in open_set:
                unvisited.update(remaining_path)

            end_time = time.time()  # `Measure end time to calculate execution time
            print("Execution Time: {:.6f} seconds".format(end_time - start_time))
            # Memory usage after the function
            memory_after = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
            print(f"Memory Used: {memory_after - memory_before:.4f} MB")
            print("Total nodes visited: ", total_nodes_visited)
            return path, visited, unvisited

        # Mark the current node as visited
        visited.add(current)
        total_nodes_visited += 1

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check if the neighbor is within bounds, not an obstacle, and not visited
            if (
                0 <= neighbor[0] < program.MapSize
                and 0 <= neighbor[1] < program.MapSize
                and program.map[neighbor[0]][neighbor[1]] != 99
                and neighbor not in visited
            ):
                tentative_g = g_cost[current] + 1

                # Add neighbor to open set if it hasn't been visited with a cheaper cost
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    f_cost = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_cost, neighbor))
                    came_from[neighbor] = current

    return [], visited  # Return empty path and visited nodes if no path is found

def a_star_tree(program):
    # Note down method start time
    start_time = time.time()

    #Note down memory before the process
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

    start = None
    goal = None

    # Locate the start and goal points
    for i in range(program.MapSize):
        for j in range(program.MapSize):
            if program.map[i][j] == 1:
                start = (i, j)
            elif program.map[i][j] == 2:
                goal = (i, j)

    if not start or not goal:
        return [], []

    # Initialize priority queue, cost dictionary, and parent tracking
    open_set = [(0, start)]  # (priority, node)
    g_cost = {start: 0}  # Cost from start to a node
    visited = set()
    unvisited = set()
    came_from = {}  # To reconstruct the path
    total_nodes_visited = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        visited.add(current)
        total_nodes_visited += 1

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            for remaining_path in open_set:
                unvisited.update(remaining_path)

            end_time = time.time()  # `Measure end time to calculate execution time
            print("Execution Time: {:.6f} seconds".format(end_time - start_time))
            # Memory usage after the function
            memory_after = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
            print(f"Memory Used: {memory_after - memory_before:.4f} MB")
            print("Total nodes visited: ", total_nodes_visited)
            return path, visited, unvisited

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if (
                0 <= neighbor[0] < program.MapSize
                and 0 <= neighbor[1] < program.MapSize
                and program.map[neighbor[0]][neighbor[1]] != 99
            ):
                tentative_g = g_cost[current] + 1

                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    f_cost = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_cost, neighbor))
                    came_from[neighbor] = current

    return [], visited

def heuristic(node, goal):
    """Heuristic function for A*. Uses Manhattan distance."""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def ucs_tree_search(program, include_diagonal_movement=False):
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

    open_set = []
    heapq.heappush(open_set, (0, start, []))  # (cost, state, path)
    visited = set()
    unvisited = set()
    total_nodes_visited = 0
    move_cost = 0

    # Define movement directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    if include_diagonal_movement:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonals

    while open_set:
        cost, current, path = heapq.heappop(open_set)

        #if current in visited:
            #continue

        visited.add(current)
        total_nodes_visited += 1

        # Check if the goal is reached
        if current == goal:
            for _, position, _ in open_set:
                unvisited.add(position)
            end_time = time.time()  # `Measure end time to calculate execution time
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
                heapq.heappush(open_set, (cost + move_cost, neighbor, path + [current]))

    print("No path found!")
    return [], visited

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

    open_set = []
    heapq.heappush(open_set, (0, start, []))  # (cost, state, path)
    visited = set()
    unvisited = set()
    total_nodes_visited = 0
    move_cost = 0

    # Define movement directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    if include_diagonal_movement:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonals

    while open_set:
        cost, current, path = heapq.heappop(open_set)

        if current in visited:
            continue

        visited.add(current)
        total_nodes_visited += 1

        # Check if the goal is reached
        if current == goal:
            for _, position, _ in open_set:
                unvisited.add(position)
            end_time = time.time()  # `Measure end time to calculate execution time
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
                heapq.heappush(open_set, (cost + move_cost, neighbor, path + [current]))

    print("No path found!")
    return [], visited



# Run the program
if __name__ == "__main__":
    prg = Program()
    prg.create_map_with_path()

    visualizer = MapVisualizer(prg)
    visualizer.root.mainloop()
