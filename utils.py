import numpy as np
from scipy import stats
from scipy.linalg import eigvals
from scipy.optimize import approx_fprime


def latin_hypercube_sample(num_samples, param_names, lower_bounds, upper_bounds, ints, seed):
    sampler = stats.qmc.LatinHypercube(d=len(lower_bounds), seed=seed)
    unscaled_sample = sampler.random(n=num_samples)
    sample = stats.qmc.scale(unscaled_sample, lower_bounds, upper_bounds).tolist()
    sampled_params = [
        {param_names[i]: round(s[i]) if ints[i] else s[i] for i in range(len(s))} for s in sample
    ]
    return sampled_params


def create_run_cmd(
    save_loc,
    run_cmd,
    seed,
    sample,
    strategies,
    init_freq,
    grid,
    radius,
    write_freq,
    steps,
    abm_path="."
):
    abm_args = f"-seed {seed} -l {grid} -r {radius} -write {write_freq} -steps {steps}"
    if strategies == 2:
        payoff = " ".join(
            [str(x) for x in [sample["P_00"], sample["P_01"], sample["P_10"], sample["P_11"]]]
        )
    else:
        payoff = " ".join(
            [
                f"{x:5.3f}"
                for x in [
                    sample["P_00"],
                    sample["P_01"],
                    sample["P_02"],
                    sample["P_10"],
                    sample["P_11"],
                    sample["P_12"],
                    sample["P_20"],
                    sample["P_21"],
                    sample["P_22"],
                ]
            ]
        )
    init_freq = " ".join([f"{x:7.5f}" for x in init_freq])
    sample_args = f"-loc {save_loc} -f {init_freq} -p {payoff}"
    return f"{run_cmd} {abm_path}/abm.py {abm_args} {sample_args}\n"


def classify_three_strategy_replicator(P):
    def check_edge(P, i, j):
        mix_denom = P[i][i] - P[j][i] - P[i][j] + P[j][j]
        if mix_denom != 0:
            mix = (-P[i][j] + P[j][j]) / mix_denom
            mix_stable = mix_denom < 0
        else:
            mix = np.nan
            mix_stable = np.nan
        if mix <= 0 or mix >= 1:
            mix = np.nan
            mix_stable = np.nan
        return mix, mix_stable

    def replicator(t, x, P):
        return x * (P @ x - x.T @ P @ x)

    def f(x):
        return replicator(None, np.array([x[0], x[1], 1 - x[0] - x[1]]), P)[:2]

    def interior_stability(mix_all, P):
        x_eq = np.array(mix_all[:2])
        J = np.array([approx_fprime(x_eq, lambda x: f(x)[i], 1e-6) for i in range(2)])
        eigs = eigvals(J)
        return bool(np.all(np.real(eigs) < 0))

    # Corners
    all_0_stable = P[0][0] > P[1][0] and P[0][0] > P[2][0]
    all_1_stable = P[1][1] > P[0][1] and P[1][1] > P[2][1]
    all_2_stable = P[2][2] > P[0][2] and P[2][2] > P[1][2]
    # Edges
    mix_01, mix_01_stable = check_edge(P, 0, 1)
    mix_02, mix_02_stable = check_edge(P, 0, 2)
    mix_12, mix_12_stable = check_edge(P, 1, 2)
    # Interior
    eqs_lhs = np.array([P[0] - P[1], P[0] - P[2], [1, 1, 1]])
    eqs_rhs = np.array([0, 0, 1])
    try:
        mix_all = np.linalg.solve(eqs_lhs, eqs_rhs)
        mix_all = (
            mix_all if np.all(mix_all >= 0) and np.all(mix_all <= 1) else [np.nan, np.nan, np.nan]
        )
    except np.linalg.LinAlgError:
        mix_all = [np.nan, np.nan, np.nan]
    mix_all_stable = np.nan
    if not np.any(np.isnan(mix_all)):
        mix_all_stable = interior_stability(mix_all, P)
    # Return
    return (
        [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (mix_01, 1 - mix_01, 0),
            (mix_02, 0, 1 - mix_02),
            (0, mix_12, 1 - mix_12),
            mix_all,
        ],
        [
            all_0_stable,
            all_1_stable,
            all_2_stable,
            mix_01_stable,
            mix_02_stable,
            mix_12_stable,
            mix_all_stable,
        ],
    )


def classify_two_strategy_replicator(P):
    all_0_stable = P[0][0] > P[1][0]
    all_1_stable = P[1][1] > P[0][1]
    mix_denom = P[0][0] - P[1][0] - P[0][1] + P[1][1]
    if mix_denom != 0:
        mix = (-P[0][1] + P[1][1]) / mix_denom
        mix_stable = mix_denom < 0
    else:
        mix = np.nan
        mix_stable = np.nan
    if mix < 0 or mix > 1:
        mix = np.nan
        mix_stable = np.nan
    return [1, 0, mix], [all_0_stable, all_1_stable, mix_stable]


def classify_three_strategy_dynamic(stable):
    stable_points = sum(1 for x in stable if x and not np.isnan(x))
    players = ["0", "1", "2", "0 and 1", "0 and 2", "1 and 2", "all"]
    if stable_points == 0:
        return "Neutrality"
    elif stable_points == 1:
        for i in range(len(stable)):
            if stable[i] and not np.isnan(stable[i]):
                if 0 <= i < 3:
                    return f"{players[i]} Wins"
                if 3 <= i <= 6:
                    return f"Coexistence: {players[i]}"
    else:
        outcome = ", ".join(
            [players[p] for p in range(len(stable)) if stable[p] and not np.isnan(stable[p])]
        )
        return f"Bistability: {outcome}"


def classify_two_strategy_dynamic(stable):
    if np.isnan(stable[2]):
        if stable[0]:
            return "0 Wins"
        if stable[1]:
            return "1 Wins"
    else:
        if stable[2]:
            return "Coexistence"
        if not stable[2]:
            return "Bistability"
    return "Neutrality"
