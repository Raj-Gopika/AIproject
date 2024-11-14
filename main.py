import random

import tkinter as tk

from debugpy.common.timestamp import reset


class Program:
    def __init__(self):
        self.map = []
        self.MapSize = 0
        self.ObstacleMultiplier = 2  # this will be multiplied by the map size

        #initilaizing window and canvas
        self.root = tk.Tk()
        self.root.title("Map Visualization")
        self.canvas = None

        reset_button = tk.Button(self.root, text="Reset Map", command=self.create_map_with_path)
        reset_button.pack(pady=10)

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

        self.display_map()
        self.display_map_gui()

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
            self.map[goal[0]][c] = 5
            trail.append((goal[0], c))
            c += 1 if start_point[1] > c else -1

        # Move vertically towards the start row
        while r != start_point[0]:
            self.map[r][c] = 5
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

    # GUI for the map
    def display_map_gui(self):
        #Destroy the previous canvas
        if self.canvas:
            self.canvas.destroy()

        #Create Window
        cell_size = 25  # Size of each cell in pixels
        self.canvas = tk.Canvas(self.root, width=self.MapSize * cell_size, height=self.MapSize * cell_size)
        self.canvas.pack()

        # Add colors to cells
        colors = {
            0: "white",  # Empty cell
            1: "green",  # Start
            2: "red",  # Goal
            5: "blue",  # Path
            99: "black"  # Wall or obstacle
        }

        # Draw the map
        for i in range(self.MapSize):
            for j in range(self.MapSize):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                color = colors.get(self.map[i][j], "white")
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

        self.root.mainloop()


# Run the program
if __name__ == "__main__":
    prg = Program()
    prg.create_map_with_path()
