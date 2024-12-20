import random
import tkinter as tk


class Program:
    def __init__(self):
        self.map = []
        self.MapSize = 0
        self.ObstacleMultiplier = 2  # this will be multiplied by the map size

    def create_map_with_path(self):
        self.MapSize = random.randint(12, 16)
        start = (0, 0)
        goal = (0, 0)

        self.map = [[0] * self.MapSize for _ in range(self.MapSize)]

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

        num_obstacles = self.MapSize * self.ObstacleMultiplier
        self.place_obstacles(num_obstacles, trail)

    def place_obstacles(self, num_obstacles, trail):
        trail_path = set(trail)
        count = 0

        while count < num_obstacles:
            row = random.randint(0, self.MapSize - 1)
            col = random.randint(0, self.MapSize - 1)

            if (row, col) not in trail_path and self.map[row][col] == 0:
                self.map[row][col] = 99
                count += 1


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


class MapVisualizer:
    def __init__(self, program):
        self.program = program
        self.root = tk.Tk()
        self.root.title("Map Visualization")
        self.canvas = None

        # Dropdown menu options
        self.search_method = tk.StringVar(self.root)
        self.search_method.set("None")  # Default value
        options = ["None", "BFS Graph", "BFS Tree"]

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
            path_graph,visited_nodes = bfs_graph(self.program)
            print("Path (Graph Search):", path_graph)

            if path_graph:
                print("Displaying path from Graph Search...")
                for r, c in path_graph:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(bfs_path=path_graph, visited_nodes=visited_nodes)

        elif choice == "BFS Tree":
            print("Running BFS Tree Search...")
            path_tree, visited_nodes = bfs_tree(self.program)
            print("Path (Tree Search):", path_tree)

            if path_tree:
                print("Displaying path from Tree Search...")
                for r, c in path_tree:
                    if self.program.map[r][c] == 0:  # Mark path in the grid
                        self.program.map[r][c] = 5
                self.display_map(bfs_path=path_tree, visited_nodes=visited_nodes)

    def display_map(self, bfs_path=None, visited_nodes=None):
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
            5: "blue",  # Path
            99: "black"  # Wall or obstacle
        }

        bfs_color = "purple"
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
                elif bfs_path and (i, j) in bfs_path:
                    color = bfs_color
                elif visited_nodes and (i, j) in visited_nodes:
                    color = visited_color
                else:
                    color = colors.get(self.program.map[i][j], "white")

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")


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

# Run the program
if __name__ == "__main__":
    prg = Program()
    prg.create_map_with_path()

    visualizer = MapVisualizer(prg)
    visualizer.root.mainloop()