import argparse
import json
import os

import numpy as np


class ABM:
    def __init__(
        self, save_loc, seed, grid_length, radius, init_freqs, payoff_matrix, grid_save_freq, steps
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
        self.grid_save_freq = grid_save_freq
        self.steps = steps
        # Internal state tracking
        self.timestep = 0
        self.grid = self.rng.choice(
            range(self.num_strategies), size=(grid_length, grid_length), p=init_freqs
        ).astype(np.uint8)
        self.frequency_history = []
        self.reproduction_history = []
        # Initialize coords.csv
        with open(f"{self.save_loc}/coords.csv", "w") as f:
            f.write("time,x,y,strategy\n")

    def get_neighbors(self, row, col):
        neighbors: list[tuple[int, int]] = []
        for dr in range(-self.local_radius, self.local_radius + 1):
            for dc in range(-self.local_radius, self.local_radius + 1):
                if abs(dr) + abs(dc) <= self.local_radius and (dr != 0 or dc != 0):
                    neighbors.append(((row + dr) % self.grid_length, (col + dc) % self.grid_length))
        return neighbors

    def calculate_payoff(self, focal_strategy, neighbor_strategies):
        total = sum(self.payoff_matrix[focal_strategy, s] for s in neighbor_strategies)
        return total / len(neighbor_strategies)

    def step(self):
        reproductions = [0] * self.num_strategies
        for i in self.rng.permutation(self.grid_length * self.grid_length):
            row, col = divmod(i, self.grid_length)
            focal_strategy = self.grid[row, col]
            neighbors = self.get_neighbors(row, col)
            neighbor_strategies = [self.grid[nr, nc] for nr, nc in neighbors]
            payoff = self.calculate_payoff(focal_strategy, neighbor_strategies)
            if payoff > self.rng.random():
                nr, nc = neighbors[self.rng.integers(len(neighbors))]
                self.grid[nr, nc] = focal_strategy
                reproductions[focal_strategy] += 1
        self.frequency_history.append(
            np.bincount(self.grid.flatten(), minlength=self.num_strategies)
        )
        self.reproduction_history.append(reproductions)
        self.timestep += 1

    def write_grid(self, timestep):
        with open(f"{self.save_loc}/coords.csv", "a") as f:
            coords = list(np.ndindex(self.grid.shape))
            for r, c in coords:
                f.write(f"{timestep},{c},{r},{self.grid[r, c]}\n")

    def run(self):
        self.write_grid(0)
        for i in range(1, self.steps + 1):
            self.step()
            if i % self.grid_save_freq == 0:
                self.write_grid(i)

    def save(self):
        with open(f"{self.save_loc}/summary.csv", "w") as f:
            f.write("time,strategy,frequency,reproductions\n")
            for i in range(self.steps):
                for j in range(self.num_strategies):
                    freq = self.frequency_history[i][j]
                    repro = self.reproduction_history[i][j]
                    f.write(f"{i},{j},{freq},{repro}\n")


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
