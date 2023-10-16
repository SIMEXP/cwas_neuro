import numpy as np
import pandas as pd
from pathlib import Path

# Directory containing .npy files
root_p = Path("/home/neuromod/ad_sz/data/abide")
conn_p = root_p / "abide1_connectomes-0.4.1_MIST" / "control_conn_passed_qc"
output_p = root_p / "abide_controls_concat.csv"

connectome_file = conn_p.glob("*.npy")
connectome_data = []

# Iterate through each .npy file and load the data
for connectome_path in connectome_file:
    connectome = np.load(connectome_path)

    # Extract subject and site from the file name
    file_name = connectome_path.stem
    parts = file_name.split("_")
    subject = parts[0].split("-")[1]
    site = parts[1]

    # Keep upper triangle and diagonal, and flatten
    upper_indices = np.triu_indices_from(connectome, k=0)
    upper_connectome = connectome[upper_indices]
    upper_connectome_flat = upper_connectome.flatten()

    # Create a row of data as a list, including subject, site, and connectome values
    row_data = [subject, site] + upper_connectome_flat.tolist()

    # Append the row of data to the connectome_data list
    connectome_data.append(row_data)

# Define column names for the DataFrame
columns = ["Subject", "Site"] + [f"{i}" for i in range(len(upper_connectome_flat))]

# Create a DataFrame from the connectome_data list
df = pd.DataFrame(connectome_data, columns=columns)

df.to_csv(output_p, index=False)
