from pathlib import Path
import numpy as np

def load_connectomes(base_directory_p):
    """
    Load connectome data from .npy files across multiple subdirectories.

    Parameters:
    - base_directory_p: Path to the base directory containing subdirectories of connectome files.

    Returns:
    - connectomes: A list of connectome matrices, one per participant across all subdirectories.
    """
    connectomes = []
    # Iterate over each subdirectory in the base directory
    for directory_p in base_directory_p.iterdir():
        if directory_p.is_dir():
            for file_p in directory_p.glob("*.npy"):
                connectome = np.load(file_p)
                connectomes.append(connectome)
    return connectomes

def connectome_to_edges_matrix(connectomes):
    """
    Transform a list of connectomes into a matrix suitable for ComBat,
    where rows are ROI pairs and columns are participants.
    """
    num_participants = len(connectomes)
    num_rois = connectomes[0].shape[0]
    num_edges = num_rois * (num_rois - 1) // 2

    edges_matrix = np.zeros((num_edges, num_participants))
    for i, connectome in enumerate(connectomes):
        edge_index = 0
        for row in range(num_rois):
            for col in range(row + 1, num_rois):
                edges_matrix[edge_index, i] = connectome[row, col]
                edge_index += 1

    return edges_matrix

# Set paths
base_directory_p = Path("/home/neuromod/ad_sz/data/npy_connectome")
output_p = "/home/neuromod/ad_sz/data/edges_matrix.tsv"

# Load connectomes from all subdirectories.
connectomes = load_connectomes(base_directory_p)

print(f"Number of connectomes loaded: {len(connectomes)}")
for connectome in connectomes[:5]:  # Print the shape of the first 5 connectomes as a check
    print(connectome.shape)

# Transform into the ComBat-ready matrix.
edges_matrix = connectome_to_edges_matrix(connectomes)

print(f"Edges matrix shape: {edges_matrix.shape}")

# Save the edges_matrix to a .tsv file
np.savetxt(output_p, edges_matrix, delimiter='\t')

print(f"Saved edges matrix to {output_p}")
