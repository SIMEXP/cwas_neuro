import statsmodels.api as sm
import pandas as pd
import numpy as np
import random

from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path


def _load_connectomes(data_dir):
    connectome_files = data_dir.glob("*.npy")
    connectomes = []
    for connectome_path in connectome_files:
        connectome = np.load(connectome_path)

        # Keep upper triangle and diagonal, and flatten
        upper_indices = np.triu_indices_from(connectome, k=0)
        upper_connectome = connectome[upper_indices]
        upper_connectome_flat = upper_connectome.flatten()
        connectomes.append(upper_connectome_flat)

    # Convert the list of connectomes to a NumPy array
    connectomes = np.array(connectomes)

    return connectomes


def _pick_random_connections(pi):
    # Calculate the number of connections to select based on pi
    num_connections = len(connectomes[0])
    num_selected_connections = int(pi * num_connections)

    # Randomly select connections
    random_connections = random.sample(range(num_connections), num_selected_connections)
    return random_connections


def _split_subjects(connectomes):
    num_subjects = connectomes.shape[0]

    # Randomly shuffle the indices to assign subjects to groups
    indices = list(range(num_subjects))
    random.shuffle(indices)

    # Split into two group
    num_group1 = num_subjects // 2
    group1_indices = indices[:num_group1]
    group2_indices = indices[num_group1:]

    group1_connectomes = connectomes[group1_indices]
    group2_connectomes = connectomes[group2_indices]

    return group1_connectomes, group2_connectomes


def _run_cwas(n_connections, group1, group2):
    pvals = []
    for connection_i in range(n_connections):
        # Extract the connectivity data for this connection
        connectivity_i_group1 = group1[:, connection_i]
        connectivity_i_group2 = group2[:, connection_i]

        # Stack the connectivity data
        connectivity_data = np.hstack((connectivity_i_group1, connectivity_i_group2))

        # Create a design matrix with the group
        design_matrix = np.array(
            [0] * len(connectivity_i_group1) + [1] * len(connectivity_i_group2)
        )

        # Perform linear regression
        model = sm.OLS(connectivity_data, design_matrix)
        results = model.fit()

        pval = results.pvalues

        pvals.append(pval)

    return np.array(pvals).flatten()


def _simulate_power(N, pi, d, q, target_power, num_iterations, connectomes):
    sensitivities = []
    specificities = []
    achieved_power = 0
    for _ in range(num_iterations):
        # Randomly select N subjects
        total_subjects = connectomes.shape[0]
        selected_subjects = random.sample(range(total_subjects), N)

        # Step 1: Pick pi% of connections at random
        random_connections = _pick_random_connections(pi)

        # Step 2: Randomly split N selected subjects into 2 groups.
        group1, group2 = _split_subjects(connectomes[selected_subjects])

        # Step 3: Iterate through randomly picked connections for subjects in group2 and modify
        for i in random_connections:
            mean_i = np.mean(group2[:, i])
            std_i = np.std(group2[:, i])
            for subject_idx in range(len(group2)):
                group2[subject_idx][i] = group2[subject_idx][i] + d * std_i

        # Step 4: Run CWAS between group1 and group2
        n_connections = group1.shape[1]
        pvals = _run_cwas(n_connections, group1, group2)

        # Step 5: Apply FDR Correction
        rejected, _ = fdrcorrection(pvals, alpha=q)

        # Calculate the number of true positives (connections correctly detected)
        true_positives = np.sum(rejected[:n_connections])

        if true_positives > 0:
            sensitivity = true_positives / n_connections
        else:
            sensitivity = np.nan

        # Calculate specificity (proportion of true negatives among condition negatives)
        false_positives = np.sum(rejected[n_connections:])
        true_negatives = len(pvals) - n_connections - false_positives
        if true_negatives > 0:
            specificity = true_negatives / (len(pvals) - n_connections)
        else:
            specificity = np.nan

        sensitivities.append(sensitivity)
        specificities.append(specificity)

        # Calculate the power as the proportion of true positives
        power = true_positives / n_connections

        if power >= target_power:
            achieved_power += 1

    # Calculate averages
    mean_sensitivity = np.mean(sensitivities)
    mean_specificity = np.mean(specificities)

    estimated_power = achieved_power / num_iterations

    return mean_sensitivity, mean_specificity, estimated_power


if __name__ == "__main__":
    # Load control connectomes from ABIDE site UM
    abide_dir = Path("")
    connectomes = _load_connectomes(abide_dir)

    # Set values
    N = 74  # sample size - must be an even number to split into 2 groups
    pi = 0.80  # percentage of randomly selected connections to modify
    d = 0.80  # effect size
    q = 0.05  # fdr threshold
    target_power = 0.95
    num_iterations = 100

    # Generate synthetic connectomes

    mean_sensitivity, mean_specificity, estimated_power = _simulate_power(
        N, pi, d, q, target_power, num_iterations, connectomes
    )

    print(
        f"Estimated power to detect d={d} with N={N}: {estimated_power} "
        f"with a mean sensitivity of {mean_sensitivity} and mean specificity {mean_specificity}"
    )
