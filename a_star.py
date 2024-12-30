import heapq

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

# Integrate A* search methods into MapVisualizer
class MapVisualizer:
    def __init__(self, program):
        self.program = program
        self.root = tk.Tk()
        self.root.title("Map Visualization")
        self.canvas = None

        # Dropdown menu options
        self.search_method = tk.StringVar(self.root)
        self.search_method.set("None")  # Default value
        options = ["None", "BFS Graph", "BFS Tree", "A* Graph", "A* Tree"]

        dropdown = tk.OptionMenu(self.root, self.search_method, *options, command=self.run_search)
        dropdown.pack(pady=10)

        reset_button = tk.Button(self.root, text="Reset Map", command=self.reset_map)
        reset_button.pack(pady=10)

    def reset_map(self):
        self.program.create_map_with_path()
        self.display_map()

    def run_search(self, choice):
        if choice == "None":
            self.display_map()

        elif choice == "BFS Graph":
            print("Running BFS Graph Search...")
            path_graph, visited_nodes = bfs_graph(self.program)
            print("Path (Graph Search):", path_graph)

            if path_graph:
                for r, c in path_graph:
                    if self.program.map[r][c] == 0:
                        self.program.map[r][c] = 5
                self.display_map(bfs_path=path_graph, visited_nodes=visited_nodes)

        elif choice == "BFS Tree":
            print("Running BFS Tree Search...")
            path_tree, visited_nodes = bfs_tree(self.program)
            print("Path (Tree Search):", path_tree)

            if path_tree:
                for r, c in path_tree:
                    if self.program.map[r][c] == 0:
                        self.program.map[r][c] = 5
                self.display_map(bfs_path=path_tree, visited_nodes=visited_nodes)

        elif choice == "A* Graph":
            print("Running A* Graph Search...")
            path_astar_graph, visited_nodes = a_star_graph(self.program)
            print("Path (A* Graph Search):", path_astar_graph)

            if path_astar_graph:
                for r, c in path_astar_graph:
                    if self.program.map[r][c] == 0:
                        self.program.map[r][c] = 5
                self.display_map(bfs_path=path_astar_graph, visited_nodes=visited_nodes)

        elif choice == "A* Tree":
            print("Running A* Tree Search...")
            path_astar_tree, visited_nodes = a_star_tree(self.program)
            print("Path (A* Tree Search):", path_astar_tree)

            if path_astar_tree:
                for r, c in path_astar_tree:
                    if self.program.map[r][c] == 0:
                        self.program.map[r][c] = 5
                self.display_map(bfs_path=path_astar_tree, visited_nodes=visited_nodes)

    def display_map(self, bfs_path=None, visited_nodes=None):
        if self.canvas:
            self.canvas.destroy()

        cell_size = 25
        self.canvas = tk.Canvas(self.root, width=self.program.MapSize * cell_size, height=self.program.MapSize * cell_size)
        self.canvas.pack()

        colors = {
            0: "white",
            1: "green",
            2: "red",
            5: "blue",
            99: "black"
        }

        bfs_color = "purple"
        visited_color = "yellow"

        for i in range(self.program.MapSize):
            for j in range(self.program.MapSize):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                if self.program.map[i][j] == 1:
                    color = "green"
                elif self.program.map[i][j] == 2:
                    color = "red"
                elif bfs_path and (i, j) in bfs_path:
                    color = bfs_color
                elif visited_nodes and (i, j) in visited_nodes:
                    color = visited_color
                else:
                    color = colors.get(self.program.map[i][j], "white")

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
