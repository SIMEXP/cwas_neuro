import simulation_tools_v2 as sim
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Define the arguments here or pass them as function arguments
    path_conn = "/home/neuromod/ad_sz/data/abide/abide_controls_concat.csv"
    N = 200  # Sample size
    pi = 0.20  # Percentage of connections to modify
    d = 0.5  # Effect size
    q = 0.05  # FDR threshold
    num_sample = 100  # Number of simulation samples

    # Run simulation
    (
        group2_conn,
        group2_modified,
        connections_to_modify,
        result,
    ) = sim.run_simulation_experiment(path_conn, N, pi, d, q, num_sample)
    print(result)
