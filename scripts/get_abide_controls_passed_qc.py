import re
import h5py
import numpy as np
import pandas as pd

from pathlib import Path


def create_qc_df(qc_dir):
    qc_dfs = []
    for site in qc_dir.iterdir():
        qc_file_p = qc_dir / site / "task-rest_report.tsv"
        qc_df = pd.read_csv(qc_file_p, sep="\t")
        qc_dfs.append(qc_df)
    qc_df = pd.concat(qc_dfs)

    return qc_df


def get_subject_conn(hdf5_dir, subject_id, run):
    data = None
    with h5py.File(hdf5_dir, "r") as file:
        dataset_p = f"sub-00{subject_id}/sub-00{subject_id}_task-rest_run-{run}_atlas-MIST_desc-64_connectome"

        dataset = file.get(dataset_p)

        if dataset is not None:
            data = dataset[...]  # Reads the entire dataset into a NumPy array
        # else:
        # print(f"Dataset at path '{dataset_p}' not found in the HDF5 file.")

    return data


root_p = Path("/home/neuromod/ad_sz/data/abide")
pheno_p = root_p / "Phenotypic_V1_0b.csv"
qc_dir = root_p / "abide1_giga-auto-qc-0.3.1"
conn_p = root_p / "abide1_connectomes-0.4.1_MIST"
output_dir = conn_p / "control_conn_passed_qc"

output_dir.mkdir(parents=True, exist_ok=True)

# Load qc info and concatenate into one df
qc_df = create_qc_df(qc_dir)

# Load phenotypic info
pheno_df = pd.read_csv(pheno_p)

# Filter rows in 'pheno_df' where DX_GROUP is 2
control_pheno_df = pheno_df[pheno_df["DX_GROUP"] == 2]

# Filter rows in 'qc_df' where pass_all_qc is True
passed_qc_df = qc_df[qc_df["pass_all_qc"] == True]

# Merge dataframes to get only controls whose scans passed qc
merged_df = pd.merge(
    passed_qc_df,
    control_pheno_df,
    left_on="participant_id",
    right_on="SUB_ID",
    how="inner",
)

subject_dict = {}
# Loop through the filtered subjects (controls only)
for index, row in merged_df.iterrows():
    participant_id = f"{row['participant_id']}"
    run = row["run"]

    # Add the subject_number as a key in the dictionary
    if participant_id not in subject_dict:
        subject_dict[participant_id] = {"site_name": set(), "arrays": []}

    # Loop through the sites in the qc directory and load the HDF5 files from the conn directory
    # since conn directory contains non-site directories
    for site in qc_dir.iterdir():
        site_name = site.parts[-1]
        hdf5_dir = conn_p / site_name / "atlas-MIST_desc-scrubbing.5+gsr.h5"
        conn_npy = get_subject_conn(hdf5_dir, participant_id, run)

        if conn_npy is not None:
            connectome_npy_fisher = np.arctanh(conn_npy)

            subject_dict[participant_id]["site_name"].add(site_name)
            subject_dict[participant_id]["arrays"].append(connectome_npy_fisher)

# Iterate through the subject dictionary, and average arrays if >1. Otherwise, save array
for participant_id, data in subject_dict.items():
    # site_name = data["site_name"]
    site_name = re.sub(r"[^a-zA-Z0-9]", "", str(data["site_name"]))
    arrays = data["arrays"]

    if len(arrays) > 1:
        average_array = np.mean(arrays, axis=0)

        output_filename = (
            output_dir
            / f"sub-{participant_id}_{site_name}_average_atlas-MIST_desc-64_connectome.npy"
        )
        np.save(output_filename, average_array)
        print(f"Averaged array saved for subject {participant_id} to {output_filename}")

    else:
        output_filename = (
            output_dir
            / f"sub-{participant_id}_{site_name}_atlas-MIST_desc-64_connectome.npy"
        )
        np.save(output_filename, arrays)

# Some print statements to ensure the correct number of connectomes are saved (also ran without averaging to check N)
num_rows_pass_qc = (merged_df["pass_all_qc"] == True).sum()
print(f"Number of rows where pass_all_qc is True: {num_rows_pass_qc}")

num_subjects_with_qc = merged_df.groupby("participant_id")["pass_all_qc"].any().sum()
print(
    f"Number of subjects with at least one pass_all_qc == True: {num_subjects_with_qc}"
)
