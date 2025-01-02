import random
import heapq
import tkinter as tk
import math
from queue import PriorityQueue

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
        options = ["None", "BFS Graph", "BFS Tree", "A* Graph", "A* Tree", "UFS"]

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

        elif choice == "BFS Graph":
            print("Running BFS Graph Search...")
            bfs_path_graph,visited_nodes = bfs_graph(self.program)
            print("Path (Graph Search):", bfs_path_graph)

            if bfs_path_graph:
                print("Displaying path from Graph Search...")
                for r, c in bfs_path_graph:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=bfs_path_graph, visited_nodes=visited_nodes)

        elif choice == "BFS Tree":
            print("Running BFS Tree Search...")
            bfs_path_tree, visited_nodes = bfs_tree(self.program)
            print("Path (Tree Search):", bfs_path_tree)

            if bfs_path_tree:
                print("Displaying path from Tree Search...")
                for r, c in bfs_path_tree:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=bfs_path_tree, visited_nodes=visited_nodes)

        elif choice == "A* Graph":
            print("Running A* Graph Search...")
            astar_path_graph, visited_nodes = a_star_graph(self.program)
            print("Path (A* Graph Search):", astar_path_graph)

            if astar_path_graph:
                for r, c in astar_path_graph:
                    if self.program.map[r][c] == 0:
                        self.program.map[r][c] = 5
                self.display_map(path=astar_path_graph, visited_nodes=visited_nodes)

        elif choice == "A* Tree":
            print("Running A* Tree Search...")
            astar_path_tree, visited_nodes = a_star_tree(self.program)
            print("Path (A* Tree Search):", astar_path_tree)

            if astar_path_tree:
                for r, c in astar_path_tree:
                    if self.program.map[r][c] == 0:
                        self.program.map[r][c] = 5
                self.display_map(path=astar_path_tree, visited_nodes=visited_nodes)


        elif choice == "UFS":
            print("Running Uniform Cost Search (UFS)...")
            UFS_path, visited_nodes = UFS(self.program, include_diagonal_movement=False)
            print("Path (UFS):", UFS_path)

            if UFS_path:
                print("Displaying path from UFS...")
                for r, c in UFS_path:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(path=UFS_path, visited_nodes=visited_nodes)

    def display_map(self, path=None, visited_nodes=None):
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
            #5: "blue",  # Path
            99: "black"  # Wall or obstacle
        }

        path_color = "purple"
        visited_color = 'yellow'

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
                else:
                    color = colors.get(self.program.map[i][j], "white")

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

#Search Algorithms
def bfs_graph(program):
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
            return path, visited

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

def bfs_tree(program):
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
            return path, visited

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


def a_star_graph(program):
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

    # Initialize priority queue, visited set, and cost dictionary
    open_set = [(0, start)]  # (priority, node)
    came_from = {}  # To reconstruct the path
    g_cost = {start: 0}  # Cost from start to a node
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, visited

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if (
                0 <= neighbor[0] < program.MapSize
                and 0 <= neighbor[1] < program.MapSize
                and neighbor not in visited
                and program.map[neighbor[0]][neighbor[1]] != 99
            ):
                tentative_g = g_cost[current] + 1

                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    f_cost = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_cost, neighbor))
                    came_from[neighbor] = current

    return [], visited

def a_star_tree(program):
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
    came_from = {}  # To reconstruct the path

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, visited

        visited.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if (
                0 <= neighbor[0] < program.MapSize
                and 0 <= neighbor[1] < program.MapSize
                and neighbor not in visited
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

class Node:
    def __init__(self, state, cost, parent):
        self.state = state
        self.cost = cost
        self.parent = parent

def UFS(program, include_diagonal_movement=False):
    """
    Uniform Cost Search (UFS) algorithm to find the shortest path.
    """
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
        return [], []

    diagonal_cost = math.sqrt(2)
    open_set = PriorityQueue()
    open_set.put((0, Node(start, 0, None)))  # (priority, Node)
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
                path.insert(0, node_path.state)
                x, y = node_path.state
                if map[x][y] != 1 and map[x][y] != 2:
                    updated_map[x][y] = 4
                node_path = node_path.parent
            return path, visited

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
                if map[nx][ny] != 99:  # Ensure the cell is not an obstacle
                    movement_cost = diagonal_cost if abs(dx) == 1 and abs(dy) == 1 else 1
                    new_cost = current_node.cost + movement_cost
                    neighbor_node = Node((nx, ny), new_cost, current_node)
                    open_set.put((new_cost, neighbor_node))

            node_path = []
            while current_node:
                node_path.append(current_node.state)
                current_node = current_node.parent
            node_path.reverse()
            return node_path, visited  # Return the path and visited nodes

        # Define movements (cardinal and optional diagonal)
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_diagonal_movement:
            movements += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    return path, visited  # Return empty path if no solution



# Run the program
if __name__ == "__main__":
    prg = Program()
    prg.create_map_with_path()

    visualizer = MapVisualizer(prg)
    visualizer.root.mainloop()
