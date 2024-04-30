import h5py
import numpy as np
import pandas as pd
from pathlib import Path

dataset_configs = {
    'cobre': {
        'connectome_path': "cobre_connectome-0.4.1_MIST_afc",
        'no_session': False,
        'subject_folder': False
    },
    'cimaq': {
        'connectome_path': "cimaq_connectome-0.4.1_MIST_afc",
        'no_session': False,
        'subject_folder': True
    },
    'hcpep': {
        'connectome_path': "hcp-ep_connectome-0.4.1",
        'no_session': False,
        'subject_folder': True
    },
    'ds000030': {
        'connectome_path': "ds000030_connectomes-0.4.1",
        'no_session': True,
        'subject_folder': False
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
    For multiple connectomes per participant, average them, apply Fisher z transform, and convert to an edge matrix.
    """
    edges_matrix_dict = {}
    for participant_id, connectomes in connectomes_by_participant.items():
        # Compute the mean connectome
        mean_connectome = np.mean(connectomes, axis=0)
        # Apply Fisher Z transformation
        transformed_connectome = np.arctanh(mean_connectome)
        # Convert connectome to a marix for ComBat
        edges_matrix = connectome_to_edge_matrix(transformed_connectome)
        edges_matrix_dict[participant_id] = edges_matrix

    return edges_matrix_dict


def matrix_dict_to_df(edges_matrix_dict):
    # Create a DataFrame from the edge matrices
    # Convert dictionary values to a 2D numpy array
    edge_matrix_combined = np.column_stack(list(edges_matrix_dict.values()))
    participants = list(edges_matrix_dict.keys())
    edges_df = pd.DataFrame(edge_matrix_combined, columns=participants)

    # Transpose so suitable for ComBat
    edges_df = edges_df.T

    return edges_df

def process_datasets(conn_p, df, dataset_name):
    config = dataset_configs[dataset_name]
    base_dir = conn_p / config['connectome_path']

    # Filter the df for required scans for the dataset
    filtered_df = df[df['dataset'] == dataset_name]
    valid_rows = []  # To store rows where the connectome was found

    connectomes_by_participant = {}
    for index, row in filtered_df.iterrows():
        participant_id = row["participant_id"]
        identifier = row["identifier"]
        session = None if config['no_session'] else row.get("ses", None)

        # Adjust file path based on whether dataset uses a subject folder
        if config['subject_folder']:
            file_path = base_dir / participant_id / f"sub-{participant_id}_atlas-MIST_desc-scrubbing.5+gsr.h5"
        else:
            file_path = base_dir / "atlas-MIST_desc-scrubbing.5+gsr.h5"
        try:
            with h5py.File(file_path, "r") as file:
                connectome = load_connectome(file, participant_id, session, identifier, config['no_session'])
                if connectome is not None:
                    # Append the connectome to the participant's list in the dictionary
                    if participant_id not in connectomes_by_participant:
                        connectomes_by_participant[participant_id] = []
                    connectomes_by_participant[participant_id].append(connectome)

                    # Add the row to valid_rows if the connectome is found
                    valid_rows.append(row)

        except FileNotFoundError:
            print(f"File not found: {file_path}")

    # Process connectomes
    if connectomes_by_participant:
        edges_matrix_dict = process_connectomes(connectomes_by_participant)
        edges_df = matrix_dict_to_df(edges_matrix_dict)

    # Convert the list of valid rows to a DataFrame
    valid_df = pd.DataFrame(valid_rows)

    return edges_df, valid_df

def create_pheno_df(valid_df):
    # Group by 'participant_id' and aggregate, taking the first entry for variables that do not change (for this use case)
    aggregation_functions = {
        'sex': 'first',
        'age': 'first',
        'diagnosis': 'first',
        'site': 'first',
        'mean_fd_scrubbed': 'mean',  # Averaging mean framewise displacement across scans
        'mbi_status': 'first'
    }
    pheno_df = valid_df.groupby('participant_id').agg(aggregation_functions)

    return pheno_df

def create_covariates_df(valid_df):
    # Group by participant_id and select the first entry for each group
    first_entries = valid_df.groupby('participant_id').first()

    # Extract the relevant covariates
    covariates_df = first_entries[['site', 'sex', 'age', 'diagnosis']]

    return covariates_df

def one_hot_encode_column(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    df.drop(column_name, axis=1, inplace=True)
    return df

def process_covariates(combined_covariates_df):
    processed_covariates_df = combined_covariates_df.copy()
    # Rename diagnosis variable
    processed_covariates_df['diagnosis'] = processed_covariates_df['diagnosis'].replace('PSYC', 'SCHZ')

    # One-hot encode covariates
    processed_covariates_df = one_hot_encode_column(processed_covariates_df, 'sex')
    processed_covariates_df = one_hot_encode_column(processed_covariates_df, 'diagnosis')

    return processed_covariates_df

if __name__ == "__main__":
    conn_p = Path("/home/neuromod/ad_sz/data")
    output_p = Path("/home/neuromod/ad_sz/data/input_data")
    df = pd.read_csv("/home/neuromod/wrangling-phenotype/outputs/final_master_pheno.tsv",sep="\t")

    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    all_edges = []
    all_pheno = []
    all_covariates = []
    for dataset in dataset_configs:
        edges_df, valid_df = process_datasets(conn_p, df, dataset)
        pheno_df = create_pheno_df(valid_df)
        covariates_df = create_covariates_df(valid_df)

        # Collect data per dataset
        all_edges.append(edges_df)
        all_pheno.append(pheno_df)
        all_covariates.append(covariates_df)

    # Concatenate data across datasets
    combined_edges_df = pd.concat(all_edges)
    combined_pheno_df = pd.concat(all_pheno)
    combined_covariates_df = pd.concat(all_covariates)

    # Process covariates and pheno data suitable for analysis
    processed_covariates_df = process_covariates(combined_covariates_df)

    # Ensure order of data is identical and output
    combined_edges_df.sort_index().to_csv(output_p / 'combat_edges.tsv', sep='\t', index=True)
    processed_covariates_df.sort_index().to_csv(output_p / 'combat_covariates.tsv', sep='\t', index=True)
    combined_pheno_df.sort_index().to_csv(output_p / 'pheno.tsv', sep='\t', index=True)

    print(f"Saved edges matrix, covariates and pheno to {output_p}")
