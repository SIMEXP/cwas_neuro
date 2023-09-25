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


def _select_subjects(connectomes, N):
    total_subjects = connectomes.shape[0]
    selected_subjects_indices = random.sample(range(total_subjects), N)
    selected_connectomes = connectomes[selected_subjects_indices]

    return selected_connectomes


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


def _pick_random_connections(pi, connectomes):
    # Calculate the number of connections to select based on pi
    num_connections = len(connectomes[0])
    num_selected_connections = int(pi * num_connections)

    # Randomly select connections
    random_connections = random.sample(range(num_connections), num_selected_connections)
    return random_connections


def _modify_connections(
    random_connections, selected_subjects_connectomes, group2_connectomes
):
    for i in random_connections:
        std_i = np.std(selected_subjects_connectomes[:, i])
        for subject_idx in range(len(group2_connectomes)):
            group2_connectomes[subject_idx][i] = (
                group2_connectomes[subject_idx][i] + d * std_i
            )

    return group2_connectomes


def _run_cwas(group1_connectomes, group2_modified):
    n_connections = group1_connectomes.shape[1]

    pvals = []
    for connection_i in range(n_connections):
        # Extract the connectivity data for this connection
        connectivity_i_group1 = group1_connectomes[:, connection_i]
        connectivity_i_group2 = group2_modified[:, connection_i]

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


def _apply_fdr(pvals, q):
    n_connections = group1_connectomes.shape[1]
    rejected, corrected_pvals = fdrcorrection(pvals, alpha=q)

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

    return corrected_pvals, sensitivity, specificity


# Load control connectomes from ABIDE site UM
abide_dir = Path("")
connectomes = _load_connectomes(abide_dir)

# Set values
N = 50  # sample size - must be an even number to split into 2 groups
pi = 0.10  # percentage of randomly selected connections to modify
d = 0.30  # effect size
q = 0.05  # fdr threshold
target_power = 0.95
num_iterations = 100

sensitivities = []
specificities = []
correct_rejections = 0
for iteration in range(num_iterations):
    # Randomly select N subjects
    selected_subjects_connectomes = _select_subjects(connectomes, N)

    # Step 1: Randomly split N selected subjects into 2 groups.
    group1_connectomes, group2_connectomes = _split_subjects(
        selected_subjects_connectomes
    )

    # Step 2: Pick pi% of connections at random
    random_connections = _pick_random_connections(pi, group2_connectomes)

    # Step 3: Iterate through randomly picked connections for subjects in group2 and modify
    group2_modified = _modify_connections(
        random_connections, selected_subjects_connectomes, group2_connectomes
    )

    # Step 4: Run CWAS between group1 and group2
    pvals = _run_cwas(group1_connectomes, group2_modified)

    # Step 5: Apply FDR Correction
    corrected_pvals, sensitivity, specificity = _apply_fdr(pvals, q)

    sensitivities.append(sensitivity)
    specificities.append(specificity)

    # If null hypothesis rejected, plus 1
    if np.any(corrected_pvals < q):
        correct_rejections += 1


# Calculate the estimated statistical power
power = correct_rejections / num_iterations

print(
    f"Estimated power to detect d={d} with N={N}: {power},"
    f" with a mean sensitivity of {np.mean(sensitivities)} and mean specificity {np.mean(specificities)}"
)

if power >= target_power:
    print(f"Estimated power is equal to or greater than {target_power}!")

else:
    print(f"Target power of {target_power} not reached.")
