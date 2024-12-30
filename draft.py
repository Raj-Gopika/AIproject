import random
import heapq
import tkinter as tk

class Program:
    def __init__(self):
        self.map = []
        self.MapSize = 0

    def create_map_with_path(self):
        self.MapSize = random.randint(12, 16)
        start = (0, 0)
        goal = (0, 0)

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

        # Labels to show visited nodes and path length
        self.visited_label = tk.Label(self.root, text="Visited Nodes: 0")
        self.visited_label.pack(pady=5)

        self.path_label = tk.Label(self.root, text="Path Length: 0")
        self.path_label.pack(pady=5)

    def reset_map(self):
        self.program.create_map_with_path()
        self.display_map()

    def run_search(self, choice):
        if choice == "None":
            self.display_map()  # Display the map without any path

        elif choice == "BFS Graph":
            print("Running BFS Graph Search...")
            path_graph, visited_nodes = bfs_graph(self.program)
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

        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack()

        node_size = 600 // self.program.MapSize
        for r in range(self.program.MapSize):
            for c in range(self.program.MapSize):
                x1 = c * node_size
                y1 = r * node_size
                x2 = x1 + node_size
                y2 = y1 + node_size
                if self.program.map[r][c] == 99:
                    color = "black"  # Wall
                elif self.program.map[r][c] == 1:
                    color = "green"  # Start
                elif self.program.map[r][c] == 2:
                    color = "red"  # Goal
                elif self.program.map[r][c] == 5:
                    color = "blue"  # Path
                elif self.program.map[r][c] == 6:
                    color = "yellow"  # Trail
                else:
                    color = "white"  # Empty space

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

        # Update the labels
        if visited_nodes is not None:
            self.visited_label.config(text=f"Visited Nodes: {visited_nodes}")

        if bfs_path is not None:
            self.path_label.config(text=f"Path Length: {len(bfs_path)}")

        self.root.mainloop()


def bfs_graph(program):
    # Dummy BFS graph search (implement your own)
    visited_nodes = 0
    path = [(0, 0), (0, 1), (1, 1), (1, 2)]
    return path, visited_nodes

def bfs_tree(program):
    # Dummy BFS tree search (implement your own)
    visited_nodes = 0
    path = [(0, 0), (1, 0), (1, 1)]
    return path, visited_nodes

def a_star_graph(program):
    # Dummy A* graph search (implement your own)
    visited_nodes = 0
    path = [(0, 0), (1, 0), (1, 1)]
    return path, visited_nodes

def a_star_tree(program):
    # Dummy A* tree search (implement your own)
    visited_nodes = 0
    path = [(0, 0), (0, 1), (1, 1)]
    return path, visited_nodes


if __name__ == "__main__":
    program = Program()
    program.create_map_with_path()
    visualizer = MapVisualizer(program)
    visualizer.display_map()
