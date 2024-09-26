#import os dont think I need this?
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def load_connectome(file, participant_id, session, identifier, no_session):
    # Construct the dataset path
    if no_session:
        dataset_path = f"sub-{participant_id}/{identifier}_atlas-MIST_desc-64_connectome"
    else:
        dataset_path = f"sub-{participant_id}/ses-{session}/{identifier}_atlas-MIST_desc-64_connectome"

    dataset = file.get(dataset_path)

    connectome_npy = None  # Initialise to None in case missing
    if dataset is not None:
        connectome_npy = dataset[...]  # Load the dataset into a NumPy array
    return connectome_npy

def save_connectome(connectome_npy, output_p, identifier):
    if connectome_npy is not None:
        output_file_p = output_p / f"{identifier}.npy"
        np.save(output_file_p, connectome_npy)
        print(f"Saved data for {identifier}")
    else:
        print(f"Connectome for {identifier} not found or could not be loaded")

def process_connectome_participant(row, base_dir, output_p):
    identifier = row["identifier"]
    participant_id = row["participant_id"]
    session = row["ses"]
    # Construct path to subject's HDF5 file
    hdf5_file_p = base_dir / f"{participant_id}" / f"sub-{participant_id}_atlas-MIST_desc-scrubbing.5+gsr.h5"

    if hdf5_file_p.exists():
        with h5py.File(hdf5_file_p, "r") as file:
            connectome_npy = load_connectome(file, participant_id, session, identifier)
            save_connectome(connectome_npy, output_p, identifier)

def process_connectome_group(row, file, output_p, no_session):
    identifier = row["identifier"]
    participant_id = row["participant_id"]
    session = row["ses"]

    connectome_npy = load_connectome(file, participant_id, session, identifier, no_session)
    save_connectome(connectome_npy, output_p, identifier)

def process_cobre(conn_p, df, output_dir):
    base_dir = conn_p / "cobre_connectome-0.4.1"
    hdf5_file_p = base_dir / "atlas-MIST_desc-scrubbing.5+gsr.h5"
    output_p = output_dir / "cobre"
    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    # Filter df for dataset
    filtered_df = df[df['dataset'] == 'cobre']

    with h5py.File(hdf5_file_p, "r") as file:
        # Loop through the filtered dataframe
        for index, row in filtered_df.iterrows():
            process_connectome_group(row, file, output_p)

def process_ds000030(conn_p, df, output_dir):
    base_dir = conn_p / "ds000030_connectomes-0.4.1"
    hdf5_file_p = base_dir / "atlas-MIST_desc-scrubbing.5+gsr.h5"
    output_p = output_dir / "ds000030"
    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    # Filter df for dataset
    filtered_df = df[df['dataset'] == 'ds000030']

    with h5py.File(hdf5_file_p, "r") as file:
        # Loop through the filtered dataframe
        for index, row in filtered_df.iterrows():
            process_connectome_group(row, file, output_p, no_session=True)

def process_hcpep(conn_p, df, output_dir):
    base_dir = conn_p / "hcp-ep_connectome-0.4.1"
    output_p = output_dir / "hcpep"
    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    # Filter df for dataset
    filtered_df = df[df['dataset'] == 'hcpep']

    # Loop through the filtered dataframe
    for index, row in filtered_df.iterrows():
        process_connectome_participant(row, base_dir, output_p)

def process_srpbs(conn_p, df, output_dir):
    base_dir = conn_p / "srpbs_connectome-0.4.1"
    output_p = output_dir / "srpbs"
    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    # Filter df for dataset
    filtered_df = df[df['dataset'] == 'srpbs']

    # Loop through the filtered dataframe
    for index, row in filtered_df.iterrows():
        process_connectome_participant(row, base_dir, output_p)

def process_adni(conn_p, final_file_p, output_dir):
    base_dir = conn_p / "adni_connectomes-0.4.1"
    final_file = final_file_p / "final_adni.tsv"
    output_p = output_dir / "adni"
    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    # Load the file to get identifiers and participant_ids
    df = pd.read_csv(final_file, sep="\t")

    # Loop through the filtered dataframe
    for index, row in df.iterrows():
        process_connectome_participant(row, base_dir, output_p)

def process_oasis3(conn_p, final_file_p, output_dir):
    base_dir = conn_p / "oasis3_connectomes-0.4.1"
    final_file = final_file_p / "final_oasis3.tsv"
    output_p = output_dir / "oasis3"
    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    # Load the file to get identifiers and participant_ids
    df = pd.read_csv(final_file, sep="\t")

    # Loop through the filtered dataframe
    for index, row in df.iterrows():
        process_connectome_participant(row, base_dir, output_p)

def process_cimaq(conn_p, final_file_p, output_dir):
    base_dir = conn_p / "cimaq_connectomes-0.4.1"
    final_file = final_file_p / "final_cimaq.tsv"
    output_p = output_dir / "cimaq"
    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    # Load the file to get identifiers and participant_ids
    df = pd.read_csv(final_file, sep="\t")

    # Loop through the filtered dataframe
    for index, row in df.iterrows():
        process_connectome_participant(row, base_dir, output_p)


if __name__ == "__main__":
    passed_qc_file_p = Path("/home/nclarke/projects/rrg-pbellec/nclarke/ad_sz/files/passed_qc_master.tsv")
    conn_p = Path("/home/nclarke/scratch")
    output_dir = Path("/home/nclarke/scratch/npy_connectome")

    # These datasets needed some extra steps aftre checking QC, so get identifiers from a different tsv
    final_file_p = Path("/home/nclarke/projects/rrg-pbellec/nclarke/ad_sz/files")

    # Load the file to get identifiers and participant_ids
    df = pd.read_csv(passed_qc_file_p, sep="\t")

    #process_cobre(conn_p, df, output_dir)
    process_ds000030(conn_p, df, output_dir)
    #process_hcpep(conn_p, df, output_dir)
    #process_srpbs(conn_p, df, output_dir)
    #process_adni(conn_p, final_file_p, output_dir)
    #process_cimaq(conn_p, final_file_p, output_dir)
    #process_oasis3(conn_p, final_file_p, output_dir)








