#!/usr/bin/env python3
"""
Single‐table Round‐Robin Scheduler with OR‐Tools CP‐SAT
Minimizes the maximum idle gap between matches.
"""

from ortools.sat.python import cp_model
import pandas as pd


def schedule_round_robin(n, time_limit_s=None, num_workers=8):
    # Total rounds = total matches
    M = n * (n - 1) // 2
    # All unordered pairs (i < j)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Try each candidate T from 0 upward
    for T in range(M):
        model = cp_model.CpModel()

        # Decision variables
        x = {}  # x[i,j,r] = 1 if i vs j in round r
        y = {}  # y[i,r] = 1 if player i plays in round r
        for i, j in pairs:
            for r in range(M):
                x[(i, j, r)] = model.NewBoolVar(f"x_{i}_{j}_{r}")
        for i in range(n):
            for r in range(M):
                y[(i, r)] = model.NewBoolVar(f"y_{i}_{r}")

        # 1) Each match exactly once
        for i, j in pairs:
            model.Add(sum(x[(i, j, r)] for r in range(M)) == 1)

        # 2) Only one match per round
        for r in range(M):
            model.Add(sum(x[(i, j, r)] for (i, j) in pairs) == 1)

        # 3) Link x and y: y[i,r] == sum of matches involving i in round r
        for i in range(n):
            for r in range(M):
                model.Add(
                    sum(x[(a, b, r)] for (a, b) in pairs if a == i or b == i)
                    == y[(i, r)]
                )

        # 4) Idle‐time constraint: in any window of T+1 rounds, each player plays at least once
        for i in range(n):
            for start in range(M - T):
                model.Add(
                    sum(y[(i, start + s)] for s in range(T + 1)) >= 1
                )

        # Solve as a pure feasibility problem
        solver = cp_model.CpSolver()
        if time_limit_s is not None:
            solver.parameters.max_time_in_seconds = time_limit_s
        solver.parameters.num_search_workers = num_workers

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Extract schedule
            schedule = []
            for (i, j, r), var in x.items():
                if solver.Value(var):
                    schedule.append((r + 1, i + 1, j + 1))  # 1‐based indexing
            return T, sorted(schedule)

    # Should never get here
    return None


if __name__ == "__main__":
    import sys

    # Read number of players from command line, default to 6
    n_players = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    if result := schedule_round_robin(n_players, time_limit_s=30, num_workers=8):
        T_opt, schedule = result
        print(f"Optimal maximum idle gap: {T_opt}\n")

        data = []
        for round_, p1, p2 in schedule:
            row = {"Round": round_}
            for player in range(1, n_players + 1):
                row[f'Player {player}'] = 1 if player in (p1, p2) else 0
            data.append(row)
            print(f"Round {round_:2d}: Player {p1} vs Player {p2}")
        df = pd.DataFrame(data)
        df.set_index('Round', inplace=True)

    else:
        print("No feasible schedule found within the time limit.")
