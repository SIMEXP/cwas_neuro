import h5py
import numpy as np
import pandas as pd
from pathlib import Path

dataset_configs = {
    "adni": {
        "connectome_path": "adni_connectome-0.4.1_MIST_afc",
        "no_session": False,
        "subject_folder": True,
    },
}


def load_connectome(file, participant_id, session, identifier, no_session):
    # Path construction within .h5 file
    dataset_path = f"sub-{participant_id}"

    if not no_session:
        dataset_path += f"/ses-{session}"

    dataset_path += f"/{identifier}_atlas-MIST_desc-64_connectome"
    dataset = file.get(dataset_path)

    if dataset is not None:
        return dataset[...]
    else:
        print(f"Connectome for {identifier} not found")
        return None


def save_npy_connectome(connectome_npy, output_p, identifier):
    # Not using this function currently
    if connectome_npy is not None:
        output_file_p = output_p / f"{identifier}.npy"
        np.save(output_file_p, connectome_npy)
        print(f"Saved data for {identifier}")
    else:
        print(f"Connectome for {identifier} not found")


def connectome_to_edge_matrix(connectome):
    """
    Transform a single connectome into an edge matrix.
    Extract the upper triangular part of the connectome, including the diagonal.
    """
    # Get the upper triangular indices including the diagonal
    row_indices, col_indices = np.triu_indices_from(connectome)

    # Extract the values at these indices
    edge_matrix = connectome[row_indices, col_indices]

    return edge_matrix


def process_connectomes(connectomes_by_participant):
    """
    Apply Fisher z transform, compute the global signal and convert to an edge matrix. For multiple connectomes per participant, connectomes are averaged first.
    """
    edges_matrix_dict = {}
    global_signal_dict = {}
    for participant_id, connectomes in connectomes_by_participant.items():
        # Compute the mean connectome
        mean_connectome = np.mean(connectomes, axis=0)
        # Apply Fisher Z transformation
        transformed_connectome = np.arctanh(mean_connectome)
        # Compute global signal (mean of the Z values)
        global_signal = np.mean(transformed_connectome)
        global_signal_dict[participant_id] = global_signal
        # Convert connectome to a marix for ComBat
        edges_matrix = connectome_to_edge_matrix(transformed_connectome)
        edges_matrix_dict[participant_id] = edges_matrix

    return edges_matrix_dict, global_signal_dict


def matrix_dict_to_df(edges_matrix_dict):
    # Create a df from the edge matrices
    # Convert dictionary values to a 2D numpy array
    edge_matrix_combined = np.column_stack(list(edges_matrix_dict.values()))
    participants = list(edges_matrix_dict.keys())
    edges_df = pd.DataFrame(edge_matrix_combined, columns=participants)

    # Transpose so suitable for ComBat
    edges_df = edges_df.T

    return edges_df


def process_datasets(conn_p, df, dataset_name):
    config = dataset_configs[dataset_name]
    base_dir = conn_p / config["connectome_path"]

    # Filter the df for required scans for the dataset
    filtered_df = df[df["dataset"] == dataset_name]
    valid_rows = []  # To store rows where the connectome was found

    connectomes_by_participant = {}
    for index, row in filtered_df.iterrows():
        participant_id = row["participant_id"]
        identifier = row["identifier"]
        session = None if config["no_session"] else row.get("ses", None)

        # Adjust file path based on whether dataset uses a subject folder
        if config["subject_folder"]:
            file_path = (
                base_dir
                / participant_id
                / f"sub-{participant_id}_atlas-MIST_desc-scrubbing.5+gsr.h5"
            )
        else:
            file_path = base_dir / "atlas-MIST_desc-scrubbing.5+gsr.h5"
        try:
            with h5py.File(file_path, "r") as file:
                connectome = load_connectome(
                    file, participant_id, session, identifier, config["no_session"]
                )
                if connectome is not None:
                    # Append the connectome to the participant's list in the dictionary
                    if participant_id not in connectomes_by_participant:
                        connectomes_by_participant[participant_id] = []
                    connectomes_by_participant[participant_id].append(connectome)

                    # Add the row to valid_rows if the connectome is found
                    valid_rows.append(row)

        except FileNotFoundError:
            print(f"File not found: {file_path}")

    edges_df = pd.DataFrame()
    # Process connectomes
    if connectomes_by_participant:
        edges_matrix_dict, global_signal_dict = process_connectomes(
            connectomes_by_participant
        )
        edges_df = matrix_dict_to_df(edges_matrix_dict)

    # Convert the list of valid rows to a df
    valid_df = pd.DataFrame(valid_rows)

    return edges_df, valid_df, global_signal_dict


def create_pheno_df(valid_df, global_signal_dict):
    # Group by 'participant_id' and aggregate, taking the first entry for variables that do not change (for this use case)
    aggregation_functions = {
        "sex": "first",
        "age": "first",
        "diagnosis": "first",
        "site": "first",
        "dataset": "first",
        "mean_fd_scrubbed": "mean",  # Averaging mean framewise displacement across scans
        "group": "first",
    }
    pheno_df = valid_df.groupby("participant_id").agg(aggregation_functions)
    # Add global signal to pheno_df
    pheno_df["mean_conn"] = pheno_df.index.map(global_signal_dict)
    return pheno_df


def create_covariates_df_old(valid_df):
    # Group by participant_id and select the first entry for each group
    first_entries = valid_df.groupby("participant_id").first()

    # Extract the relevant covariates
    covariates_df = first_entries[["site", "sex", "age", "diagnosis"]]

    return covariates_df


def one_hot_encode_column(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    df.drop(column_name, axis=1, inplace=True)
    return df


def one_hot_encode_column_no_prefix(df, column_name):
    dummies = pd.get_dummies(df[column_name])
    df = pd.concat([df, dummies], axis=1)
    df.drop(column_name, axis=1, inplace=True)
    return df


def create_covariates_df(pheno_df):
    # Extract the relevant covariates from pheno_df
    covariates_df = pheno_df[
        [
            "site",
            "sex_male",
            "sex_female",
            "age",
            "diagnosis_MCI",
            "diagnosis_ADD",
            "diagnosis_CON",
            "diagnosis_SCHZ",
        ]
    ]  # split CON into controls from AD and SCHZ datasets?

    return covariates_df


if __name__ == "__main__":
    conn_p = Path("/home/nclarke/scratch")
    output_p = Path("/home/nclarke")
    df = pd.read_csv("/home/nclarke/final_qc_pheno.tsv", sep="\t")

    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    all_edges = []
    all_pheno = []
    for dataset in dataset_configs:
        edges_df, valid_df, global_signal_dict = process_datasets(conn_p, df, dataset)
        pheno_df = create_pheno_df(valid_df, global_signal_dict)

        # Collect data per dataset
        all_edges.append(edges_df)
        all_pheno.append(pheno_df)

    # Concatenate data across datasets
    combined_edges_df = pd.concat(all_edges)
    combined_pheno_df = pd.concat(all_pheno)

    # One-hot encode columns
    combined_pheno_df = one_hot_encode_column(combined_pheno_df, "sex")
    combined_pheno_df = one_hot_encode_column(combined_pheno_df, "diagnosis")
    combined_pheno_df = one_hot_encode_column(combined_pheno_df, "group")

    # Create covariates_df from pheno_df
    covariates_df = create_covariates_df(combined_pheno_df)

    # Capitalize column names, necessary for combat etc
    covariates_df.columns = [x.upper() for x in covariates_df.columns]
    combined_pheno_df.columns = [x.upper() for x in combined_pheno_df.columns]

    print("Shape of combined_edges_df = ", combined_edges_df.shape)
    print("Shape of covariates_df = ", covariates_df.shape)
    print("Shape of combined_pheno_df = ", combined_pheno_df.shape)

    # Ensure order of data is identical and output
    combined_edges_df.sort_index().to_csv(
        output_p / "combat_edges.tsv", sep="\t", index=True
    )
    covariates_df.sort_index().to_csv(
        output_p / "combat_covariates.tsv", sep="\t", index=True
    )
    combined_pheno_df.sort_index().to_csv(output_p / "pheno.tsv", sep="\t", index=True)

    print(f"Saved edges matrix, covariates and pheno to {output_p}")
