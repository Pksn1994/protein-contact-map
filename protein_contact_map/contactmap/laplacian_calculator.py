"""
Step 4: Calculate the Laplacian matrix from contact matrix
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


def compute_laplacian_matrix(contact_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the graph Laplacian matrix from a contact matrix.
    
    The Laplacian matrix L = D - A, where:
    - D is the degree matrix (diagonal matrix of node degrees)
    - A is the adjacency matrix (contact_matrix)
    
    Parameters:
    -----------
    contact_matrix : np.ndarray
        Binary adjacency matrix representing contacts
        
    Returns:
    --------
    laplacian_matrix : np.ndarray
        The graph Laplacian matrix
    """
    degrees = np.sum(contact_matrix, axis=1)
    # print(f"degrees shape: {degrees.shape}")  # debugging line
    
    # Create degree matrix (diagonal matrix)
    degree_matrix = np.diag(degrees)
    
    laplacian_matrix = degree_matrix - contact_matrix
    
    return laplacian_matrix


def analyze_laplacian_properties(laplacian_matrix: np.ndarray):
    """
    Analyze mathematical properties of the Laplacian matrix.
    
    Parameters:
    -----------
    laplacian_matrix : np.ndarray
        The Laplacian matrix to analyze
    """
    is_symmetric = np.allclose(laplacian_matrix, laplacian_matrix.T)
    
    eigenvalues, _ = eigh(laplacian_matrix)
    is_psd = np.all(eigenvalues >= -1e-10)  # Small tolerance for numerical errors
    
    # Row sums should be zero for Laplacian
    row_sums = np.sum(laplacian_matrix, axis=1)
    zero_row_sums = np.allclose(row_sums, 0)
    
    # print(f"eigenvalues computed: {len(eigenvalues)}")  # check this worked
    
    print(f"Laplacian Matrix Properties:")
    print(f"Shape: {laplacian_matrix.shape}")
    print(f"Is symmetric: {is_symmetric}")
    print(f"Is positive semi-definite: {is_psd}")
    print(f"Row sums are zero: {zero_row_sums}")
    print(f"Number of zero eigenvalues: {np.sum(eigenvalues < 1e-10)}")
    print(f"Smallest eigenvalue: {np.min(eigenvalues):.6f}")
    print(f"Second smallest eigenvalue: {eigenvalues[1]:.6f}")
    
    return {
        'eigenvalues': eigenvalues,
        'is_symmetric': is_symmetric,
        'is_psd': is_psd,
        'zero_row_sums': zero_row_sums
    }


def visualize_eigenvalue_spectrum(eigenvalues: np.ndarray, filename: str = "eigenvalue_spectrum.png"):
    """
    Visualize the eigenvalue spectrum of the Laplacian matrix.
    
    Parameters:
    -----------
    eigenvalues : np.ndarray
        Sorted eigenvalues of the Laplacian matrix
    filename : str
        Output filename for the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot 1: All eigenvalues
    plt.subplot(1, 2, 1)
    plt.plot(eigenvalues, 'b.-', markersize=3)
    plt.title('All Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # plt.plot(eigenvalues[:20], 'r.-', markersize=5)  # original
    n_plot = min(20, len(eigenvalues))
    plt.plot(eigenvalues[:n_plot], 'r.-', markersize=5)
    plt.title('First 20 Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Eigenvalue spectrum saved to: {filename}")


def save_laplacian_matrix(laplacian_matrix: np.ndarray, filename: str = "laplacian_matrix.csv"):
    """
    Save Laplacian matrix to CSV file.
    
    Parameters:
    -----------
    laplacian_matrix : np.ndarray
        The Laplacian matrix to save
    filename : str
        Output filename
    """
    np.savetxt(filename, laplacian_matrix, delimiter=",", fmt='%.6f')
    print(f"Laplacian matrix saved to: {filename}")


def load_contact_matrix(filename: str) -> np.ndarray:
    """
    Load contact matrix from CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to contact matrix CSV file
        
    Returns:
    --------
    contact_matrix : np.ndarray
        The loaded contact matrix
    """
    # return np.loadtxt(filename, delimiter=",", dtype=int)  # original line
    try:
        return np.loadtxt(filename, delimiter=",", dtype=int)
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        return None


if __name__ == "__main__":
    # Load contact matrix (assuming it was created by contact_matrix.py)
    print("Loading contact matrix...")
    contact_matrix = load_contact_matrix("contact_matrix_6vsb.csv")
    
    if contact_matrix is None:
        print("Exiting due to file loading error")
        exit(1)
    
    # Compute Laplacian matrix
    print("Computing Laplacian matrix...")
    laplacian_matrix = compute_laplacian_matrix(contact_matrix)
    
    properties = analyze_laplacian_properties(laplacian_matrix)
    
    save_laplacian_matrix(laplacian_matrix, "laplacian_matrix_6vsb.csv")
    
    visualize_eigenvalue_spectrum(properties['eigenvalues'], "eigenvalue_spectrum_6vsb.png")
    
    print(f"\nThe Laplacian matrix has been computed successfully!")
    print(f"Matrix dimensions: {laplacian_matrix.shape}")
    print(f"This completes Step 4 of the analysis pipeline.")