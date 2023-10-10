import simulation_tools_v2 as sim
import numpy as np


def run_simulation_experiment(path_conn, N, pi, d, q, num_sample):
    sensitivity_list = []
    specificity_list = []
    correct_rejected_count = 0

    for sample in range(num_sample):
        # Load connectomes and perform steps 1-4 of simulation
        group1_conn, connections_to_modify, pval_list = sim.run_simulation(
            path_conn, N, pi, d
        )

        # Step 5: Apply FDR correction
        corrected_pval_list, sensitivity, specificity = sim.apply_fdr(
            group1_conn, connections_to_modify, pval_list, q
        )

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

        # If null hypothesis rejected, plus 1
        if np.any(corrected_pval_list < q):
            correct_rejected_count += 1

    result = sim.summary(
        correct_rejected_count, sensitivity_list, specificity_list, d, N, num_sample
    )

    return result


if __name__ == "__main__":
    # Define the arguments here or pass them as function arguments
    path_conn = "/home/neuromod/ad_sz/data/abide/connectomes/abide_controls_concat.csv"
    N = 100  # Sample size
    pi = 0.20  # Percentage of connections to modify
    d = 0.5  # Effect size
    q = 0.05  # FDR threshold
    num_sample = 100  # Number of simulation samples

    result = run_simulation_experiment(path_conn, N, pi, d, q, num_sample)
    print(result)
