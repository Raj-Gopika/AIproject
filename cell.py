import random

class MapMaker:
    def __init__(self, size, start, end, obstacles, robot_size):
        self.size_of_map = size
        self.start_position = start
        self.end_position = end
        self.obstacles = obstacles
        self.robot_size = robot_size
        self.map_matrix = [[0 for _ in range(size)] for _ in range(size)]

    def create_matrix(self):
        for i in range(self.size_of_map):
            for col in range(self.size_of_map):
                self.map_matrix[i][col] = 0
            self.map_matrix[i][0] = 9
            self.map_matrix[i][self.size_of_map - 1] = 9

        for i in range(self.size_of_map):
            self.map_matrix[0][i] = 9
            self.map_matrix[self.size_of_map - 1][i] = 9

        self.map_matrix[self.start_position[0]][self.start_position[1]] = 1
        self.map_matrix[self.end_position[0]][self.end_position[1]] = 2

        for _ in range(self.obstacles[0]):
            self.add_obstacles()

        return self.map_matrix

    def add_obstacles(self):
        random_row = random.randint(1, self.size_of_map - 2)
        random_col = random.randint(1, self.size_of_map - 2)

        if self.obstacles[1] == "Square":
            self.add_square(random_row, random_col)
        elif self.obstacles[1] == "Line":
            self.add_line(random_row, random_col)
        elif self.obstacles[1] == "T_Line":
            self.add_t_line(random_row, random_col)
        else:
            choice = random.choice(["Square", "Line", "T_Line"])
            if choice == "Square":
                self.add_square(random_row, random_col)
            elif choice == "Line":
                self.add_line(random_row, random_col)
            elif choice == "T_Line":
                self.add_t_line(random_row, random_col)

    def add_square(self, random_row, random_col):
        for i in range(9):
            r = i // 3
            c = i % 3
            if self.check_bounds_and_start_end(random_row + r, random_col + c):
                self.map_matrix[random_row + r][random_col + c] = 9

    def add_line(self, random_row, random_col):
        direction = random.randint(1, 4)
        length = self.size_of_map // 3

        for i in range(length):
            if direction == 1:  # Vertical up
                r = random_row - i
                if self.check_bounds_and_start_end(r, random_col):
                    self.map_matrix[r][random_col] = 9
            elif direction == 2:  # Horizontal left
                c = random_col - i
                if self.check_bounds_and_start_end(random_row, c):
                    self.map_matrix[random_row][c] = 9
            elif direction == 3:  # Vertical down
                r = random_row + i
                if self.check_bounds_and_start_end(r, random_col):
                    self.map_matrix[r][random_col] = 9
            elif direction == 4:  # Horizontal right
                c = random_col + i
                if self.check_bounds_and_start_end(random_row, c):
                    self.map_matrix[random_row][c] = 9

    def add_t_line(self, random_row, random_col):
        hor_v = random.randint(1, 2)
        for i in range(10):
            if hor_v == 1:  # Horizontal base, vertical stem
                if i < 5:  # Vertical stem
                    if self.check_bounds_and_start_end(random_row + i, random_col):
                        self.map_matrix[random_row + i][random_col] = 9
                else:  # Horizontal base
                    if self.check_bounds_and_start_end(random_row + 2, random_col + i - 5):
                        self.map_matrix[random_row + 2][random_col + i - 5] = 9
            else:  # Vertical base, horizontal stem
                if i < 5:  # Horizontal stem
                    if self.check_bounds_and_start_end(random_row, random_col + i):
                        self.map_matrix[random_row][random_col + i] = 9
                else:  # Vertical base
                    if self.check_bounds_and_start_end(random_row + i - 5, random_col + 2):
                        self.map_matrix[random_row + i - 5][random_col + 2] = 9

    def check_bounds_and_start_end(self, x, y):
        if x < 0 or y < 0 or x >= self.size_of_map or y >= self.size_of_map:
            return False
        if self.map_matrix[x][y] in [1, 2]:
            return False
        return True


size = 10
start = (1, 1)
end = (8, 8)
obstacles = (5, "Square")  # 5 obstacles of type "Square"
robot_size = 1

map_maker = MapMaker(size, start, end, obstacles, robot_size)
generated_map = map_maker.create_matrix()

for row in generated_map:
    print(row)