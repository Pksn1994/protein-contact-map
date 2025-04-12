from Bio.PDB import PDBList

def download_pdb_file(pdb_id: str, save_dir: str = ".") -> str:
    """
    Downloads a PDB file from the RCSB Protein Data Bank using Biopython.

    Parameters:
        pdb_id (str): 4-character PDB ID (e.g., '6VSB')
        save_dir (str): Directory to save the file. Default is current folder.

    Returns:
        str: Full path to the downloaded file.
    """
    # Create a downloader object
    pdbl = PDBList()

    # Download the file (format='pdb', saved in save_dir)
    file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=save_dir, file_format="pdb")

    return file_path


# This will only run if the script is executed directly
if __name__ == "__main__":
    # Define the protein PDB ID (COVID spike protein)
    pdb_id = "6vsb"

    # Download the file to current folder
    path = download_pdb_file(pdb_id)

    # Confirm the download
    print(f"PDB file downloaded to: {path}")
