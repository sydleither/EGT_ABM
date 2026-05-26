import argparse
import json
import os

import numpy as np


class ABM:
    def __init__(
        self, save_loc, seed, grid_length, radius, init_freqs, payoff_matrix, write_freq, steps
    ):
        # Model parameters
        self.save_loc = save_loc
        self.rng = np.random.default_rng(seed)
        self.grid_length = grid_length
        self.local_radius = radius
        self.num_strategies = len(init_freqs)
        self.payoff_matrix = np.array(payoff_matrix).reshape(
            self.num_strategies, self.num_strategies
        )
        self.write_freq = write_freq
        self.steps = steps
        self.neighborhoods = self.get_neighborhoods(grid_length)
        # Internal state tracking
        self.timestep = 0
        self.grid = self.create_grid(init_freqs, grid_length)
        self.grid_history = []
        self.frequency_history = []

    def create_grid(self, init_freqs, grid_length):
        grid = []
        for i, freq in enumerate(init_freqs):
            num_type = int(np.round(freq * grid_length**2))
            grid.extend([i] * num_type)
        if len(grid) < grid_length**2:
            grid.append(self.num_strategies - 1)
        if len(grid) > grid_length**2:
            del grid[-1]
        grid = np.array(grid, dtype=np.uint8)
        self.rng.shuffle(grid)
        return grid.reshape(grid_length, grid_length)

    def get_neighborhoods(self, grid_length):
        neighborhoods = []
        for row in range(grid_length):
            neighborhoods.append([])
            for col in range(grid_length):
                neighbors = []
                for dr in range(-self.local_radius, self.local_radius + 1):
                    for dc in range(-self.local_radius, self.local_radius + 1):
                        if abs(dr) + abs(dc) <= self.local_radius and (dr != 0 or dc != 0):
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < grid_length and 0 <= nc < grid_length:
                                neighbors.append((nr, nc))
                neighborhoods[row].append(neighbors)
        return neighborhoods

    def calculate_payoff(self, focal_strategy, neighbor_strategies):
        total = sum(self.payoff_matrix[focal_strategy, s] for s in neighbor_strategies)
        return total / len(neighbor_strategies)

    def step(self):
        # Fill in fitness of each grid square
        fitness_grid = np.zeros(shape=(self.grid_length, self.grid_length))
        for row in range(self.grid_length):
            for col in range(self.grid_length):
                focal_strategy = self.grid[row, col]
                neighbors = self.neighborhoods[row][col]
                neighbor_strategies = [self.grid[nr, nc] for nr, nc in neighbors]
                fitness_grid[row, col] = self.calculate_payoff(focal_strategy, neighbor_strategies)
        # Update grid with biased selection
        new_grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=np.uint8)
        for row in range(self.grid_length):
            for col in range(self.grid_length):
                candidates = self.neighborhoods[row][col] + [(row, col)]
                strategies = [self.grid[nr, nc] for nr, nc in candidates]
                fitnesses = [fitness_grid[nr, nc] for nr, nc in candidates]
                sum_fitnesses = np.sum(fitnesses)
                fitness_probs = fitnesses / sum_fitnesses if sum_fitnesses > 0 else None
                new_grid[row, col] = self.rng.choice(strategies, p=fitness_probs)
        # Update tracking parameters
        self.grid = new_grid
        self.timestep += 1

    def run(self):
        self.grid_history.append(self.grid.copy())
        self.frequency_history.append(
            np.bincount(self.grid.flatten(), minlength=self.num_strategies)
        )
        for i in range(1, self.steps + 1):
            self.step()
            if i % self.write_freq == 0:
                self.grid_history.append(self.grid.copy())
                self.frequency_history.append(
                    np.bincount(self.grid.flatten(), minlength=self.num_strategies)
                )

    def save(self):
        with open(f"{self.save_loc}/summary.csv", "w") as f:
            f.write("time,strategy,frequency\n")
            for i, t in enumerate(range(0, self.steps + self.write_freq, self.write_freq)):
                for j in range(self.num_strategies):
                    freq = self.frequency_history[i][j]
                    f.write(f"{t},{j},{freq}\n")
        with open(f"{self.save_loc}/coords.csv", "w") as f:
            f.write("time,x,y,strategy\n")
            for i, t in enumerate(range(0, self.steps + self.write_freq, self.write_freq)):
                coords = list(np.ndindex(self.grid_history[i].shape))
                for r, c in coords:
                    f.write(f"{t},{c},{r},{self.grid_history[i][r, c]}\n")


def main():
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-loc", "--save_loc", type=str)
    parser.add_argument("-seed", "--seed", type=int, default=0)
    parser.add_argument("-l", "--grid", type=int, default=100)
    parser.add_argument("-r", "--radius", type=int, default=2)
    parser.add_argument("-f", "--init_freq", type=float, nargs="+")
    parser.add_argument("-p", "--payoff", type=float, nargs="+")
    parser.add_argument("-write", "--write_freq", type=int, default=4)
    parser.add_argument("-steps", "--steps", type=int, default=80)
    args = parser.parse_args()

    # Input validation
    if len(args.init_freq) ** 2 != len(args.payoff):
        raise ValueError("Match count of payoff matrix strategies to count of initial frequencies.")

    # Save run parameters
    if not os.path.exists(args.save_loc):
        os.makedirs(args.save_loc)
    with open(f"{args.save_loc}/config.json", "w") as f:
        json.dump(vars(args), f)

    # Initialize, run, and save ABM
    abm = ABM(
        args.save_loc,
        args.seed,
        args.grid,
        args.radius,
        args.init_freq,
        args.payoff,
        args.write_freq,
        args.steps,
    )
    abm.run()
    abm.save()


if __name__ == "__main__":
    main()
