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

def save_npy_connectome(connectome_npy, output_p, identifier):
    # Not using this function currently
    if connectome_npy is not None:
        output_file_p = output_p / f"{identifier}.npy"
        np.save(output_file_p, connectome_npy)
        print(f"Saved data for {identifier}")
    else:
        print(f"Connectome for {identifier} not found or could not be loaded")

def connectome_to_edges_matrix_with_ids(connectomes_with_ids):
    # Update function to new one
    num_participants = len(connectomes_with_ids)
    if num_participants == 0:
        return np.array([]), []  # Handling the case of no connectomes

    num_rois = connectomes_with_ids[0][1].shape[0]  # Shape from the first connectome
    num_edges = num_rois * (num_rois + 1) // 2

    edges_matrix = np.zeros((num_edges, num_participants))
    identifiers = []

    for i, (identifier, connectome) in enumerate(connectomes_with_ids):
        identifiers.append(identifier)
        edge_index = 0
        for row in range(num_rois):
            for col in range(row, num_rois):  # Include the diagonal
                edges_matrix[edge_index, i] = connectome[row, col]
                edge_index += 1

    return edges_matrix, identifiers


def process_datasets(conn_p, df, dataset_name):
    config = dataset_configs[dataset_name]
    base_dir = conn_p / config['connectome_path']

    filtered_df = df[df['dataset'] == dataset_name]

    connectomes_ids = []
    covariate_data_list = []
    for index, row in filtered_df.iterrows():
        participant_id = row["participant_id"]
        identifier = row["identifier"]
        session = None if config['no_session'] else row.get("ses", None)
        site = row["site"]
        sex = row["sex"]
        age = row["age"]
        #diagnosis = row["diagnosis"]

        # Adjust file path based on whether dataset uses a subject folder
        if config['subject_folder']:
            file_path = base_dir / participant_id / f"sub-{participant_id}_atlas-MIST_desc-scrubbing.5+gsr.h5"
        else:
            file_path = base_dir / "atlas-MIST_desc-scrubbing.5+gsr.h5"
        try:
            with h5py.File(file_path, "r") as file:
                connectome = load_connectome(file, participant_id, session, identifier, config['no_session'])
                if connectome is not None:
                    connectomes_ids.append((identifier, connectome))
                    #covariates = pd.DataFrame({'SITE': site, 'SEX': sex, 'AGE': age, 'DIAGNOSIS': diagnosis}, index=[identifier])
                    covariates = pd.DataFrame({'SITE': site, 'SEX': sex, 'AGE': age}, index=[identifier])
                    covariate_data_list.append(covariates)
        except FileNotFoundError:
            print(f"File not found: {file_path}")

    if connectomes_ids:
        edges_matrix, identifiers = connectome_to_edges_matrix_with_ids(connectomes_ids)
        edges_df = pd.DataFrame(edges_matrix.T, index=identifiers)
        covariate_df = pd.concat(covariate_data_list, ignore_index=False)
        return edges_df, covariate_df
    else:
        return pd.DataFrame(), pd.DataFrame()

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
        print(f"Connectome for {identifier} not found or could not be loaded")
        return None

def one_hot_encode_column(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    df.drop(column_name, axis=1, inplace=True)
    return df


if __name__ == "__main__":
    conn_p = Path("/home/neuromod/ad_sz/data")
    output_p = Path("/home/neuromod/ad_sz/data/input_data")
    df = pd.read_csv("/home/neuromod/wrangling-phenotype/outputs/final_master_pheno.tsv",sep="\t")

    if not output_p.exists():
        output_p.mkdir(parents=True, exist_ok=True)

    all_edges = []
    all_covariates = []
    for dataset in dataset_configs:
        edges_df, covariates_df = process_datasets(conn_p, df, dataset)
        all_edges.append(edges_df)
        all_covariates.append(covariates_df)

    # Concatenate all edges and covariates
    combined_edges_df = pd.concat(all_edges)
    combined_covariates_df = pd.concat(all_covariates)

    # One-hot encode covariates
    combined_covariates_df = one_hot_encode_column(combined_covariates_df, 'SEX')
    #combined_covariates_df = one_hot_encode_column(combined_covariates_df, 'DIAGNOSIS')

    combined_edges_df.sort_index().to_csv(output_p / 'combined_edges.tsv', sep='\t', index=True)
    combined_covariates_df.sort_index().to_csv(output_p / 'combat_covariates.tsv', sep='\t', index=True)

    print(f"Saved edges matrix and coavraites to {output_p}")
