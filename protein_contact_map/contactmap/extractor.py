from Bio.PDB import PDBParser
import numpy as np

def extract_ca_coordinates(pdb_file: str) -> np.ndarray:
    """
    Extracts all alpha carbon (Cα) atom coordinates from a PDB file.

    Parameters:
        pdb_file (str): The path to the .pdb or .ent file downloaded from RCSB.

    Returns:
        np.ndarray: A NumPy array of shape (N, 3), where N is the number of Cα atoms,
                    and each row contains the x, y, z coordinates of one atom.
    """
    # Create a PDBParser instance (quiet=True to suppress warnings)
    parser = PDBParser(QUIET=True)

    # Load the structure from the file
    structure = parser.get_structure("protein", pdb_file)

    # List to store the coordinates of all Cα atoms
    ca_coords = []

    # Loop through all models, chains, and residues in the structure
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if the residue has a Cα atom
                if 'CA' in residue:
                    # Get the coordinate of the Cα atom
                    coord = residue['CA'].get_coord()
                    ca_coords.append(coord)

    # Convert the list of coordinates into a NumPy array and return
    return np.array(ca_coords)


# Example usage — runs only if the file is executed directly
if __name__ == "__main__":
    # The downloaded COVID spike protein file
    pdb_path = "pdb6vsb.ent"

    # Extract all alpha carbon coordinates
    ca_array = extract_ca_coordinates(pdb_path)

    # Output results
    print(f"Extracted {len(ca_array)} alpha carbon atoms.")
    print("First 5 coordinates:")
    print(ca_array[:5])
import os

# Save coordinates to a CSV file
output_file = "ca_coordinates_6vsb.csv"

# Save the array with comma-separated values and a header
np.savetxt(output_file, ca_array, delimiter=",", header="x,y,z", comments='')

# Confirm in terminal
print(f"Coordinates saved to: {os.path.abspath(output_file)}")
