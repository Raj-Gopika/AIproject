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