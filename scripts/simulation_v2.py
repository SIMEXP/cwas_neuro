import simulation_tools_v2 as sim
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

if __name__ == "__main__":
    # Define the arguments here or pass them as function arguments
    path_conn = "/home/neuromod/ad_sz/data/abide/connectomes/abide_controls_concat.csv"
    N = 100  # Sample size
    pi = 0.20  # Percentage of connections to modify
    d = 0.5  # Effect size
    q = 0.05  # FDR threshold
    num_sample = 20  # Number of simulation samples

    # Run simulation
    (
        group2_conn,
        group2_modified,
        connections_to_modify,
        result,
    ) = sim.run_simulation_experiment(path_conn, N, pi, d, q, num_sample)
    print(result)

    # Create histogram of values for one connection
    conn_i = connections_to_modify.columns[0]
    group2_data = group2_conn[conn_i]
    group2_modified_data = group2_modified[conn_i]

    plt.hist(
        group2_data,
        color="orange",
        alpha=0.5,
        label="Original values",
        edgecolor="black",
    )
    plt.hist(
        group2_modified_data,
        color="blue",
        alpha=0.5,
        label="Values post modification",
        edgecolor="black",
    )
    plt.title(f"Modified Connection: {conn_i}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    out_p = Path("/home/neuromod/ad_sz/")
    file_name = f"histogram_{conn_i}.png"

    plt.savefig(out_p / file_name)
    plt.show()
