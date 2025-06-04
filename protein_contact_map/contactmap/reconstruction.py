"""
Steps 5-6: Calculate eigenvalues/eigenvectors and reconstruct 3D coordinates
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def calculate_eigendecomposition(laplacian_matrix: np.ndarray):
    """
    Calculate eigenvalues and eigenvectors of the Laplacian matrix.
    
    Parameters:
    -----------
    laplacian_matrix : np.ndarray
        The Laplacian matrix
        
    Returns:
    --------
    eigenvalues : np.ndarray
        Sorted eigenvalues (ascending order)
    eigenvectors : np.ndarray
        Corresponding eigenvectors as columns
    """
    # DEBUG - print matrix shape and sample
    print(f"DEBUG: Laplacian matrix shape: {laplacian_matrix.shape}")
    print(f"DEBUG: First 3x3 of matrix:\n{laplacian_matrix[:3, :3]}")
    
    # Check if matrix is symmetric (Laplacian should be)
    is_symmetric = np.allclose(laplacian_matrix, laplacian_matrix.T)
    print(f"DEBUG: Matrix is symmetric: {is_symmetric}")
    
    # First attempted using numpy's eigen
    # eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    # print("Using numpy eig - might give complex numbers sometimes...")
    
    # Try another method
    # eigenvalues = np.linalg.eigvals(laplacian_matrix)
    # eigenvectors = np.zeros_like(laplacian_matrix)
    # for i in range(len(eigenvalues)):
    #     # This is too slow!!!
    #     eigenvectors[:,i] = np.linalg.solve(laplacian_matrix - eigenvalues[i]*np.eye(len(laplacian_matrix)), 
    #                                        np.ones(len(laplacian_matrix)))
    
    # Calculate eigenvalues and eigenvectors
    print("Calculating eigendecomposition...")
    eigenvalues, eigenvectors = eigh(laplacian_matrix)
    
    # check eigenvalues
    print(f"DEBUG: First 5 eigenvalues before sorting: {eigenvalues[:5]}")
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # double check sorting worked
    print(f"DEBUG: First 5 eigenvalues after sorting: {eigenvalues[:5]}")
    
    # maybe check 
    # residual = laplacian_matrix @ eigenvectors[:, 0] - eigenvalues[0] * eigenvectors[:, 0]
    # print(f"DEBUG: Eigenvector verification residual: {np.sum(np.abs(residual))}")
    
    print(f"Eigendecomposition completed:")
    print(f"Number of eigenvalues: {len(eigenvalues)}")
    print(f"Number of zero eigenvalues: {np.sum(eigenvalues < 1e-10)}")
    
    # Check smallest eigenvalues to understand graph connectivity
    if np.sum(eigenvalues < 1e-10) > 0:
        print(f"Smallest non-zero eigenvalue: {eigenvalues[eigenvalues > 1e-10][0]:.6f}")
    else:
        print("No zero eigenvalues found - graph may be disconnected???")
    
    return eigenvalues, eigenvectors


def reconstruct_coordinates(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Reconstruct 3D coordinates using the 3 smallest non-null eigenvectors.
    
    For a connected graph, the first eigenvalue should be 0 with eigenvector
    of all ones. The next 3 eigenvectors provide the x, y, z coordinates.
    
    Parameters:
    -----------
    eigenvectors : np.ndarray
        Matrix of eigenvectors (columns are eigenvectors)
        
    Returns:
    --------
    reconstructed_coords : np.ndarray
        Nx3 array of reconstructed coordinates
    """
    # Skip the first eigenvector (should be constant for connected graph)
    # Use eigenvectors 1, 2, 3 as x, y, z coordinates
    reconstructed_coords = eigenvectors[:, 1:4]
    
    print(f"Coordinates reconstructed using eigenvectors 1, 2, 3")
    print(f"Reconstructed coordinates shape: {reconstructed_coords.shape}")
    
    return reconstructed_coords


def visualize_original_vs_reconstructed(original_coords: np.ndarray, 
                                      reconstructed_coords: np.ndarray,
                                      filename: str = "coordinate_comparison.png"):
    """
    Create 3D visualization comparing original and reconstructed coordinates.
    
    Parameters:
    -----------
    original_coords : np.ndarray
        Original coordinates from PDB
    reconstructed_coords : np.ndarray
        Reconstructed coordinates from eigenvectors
    filename : str
        Output filename for the plot
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Plot original coordinates
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2], 
               c='blue', alpha=0.6, s=1)
    ax1.set_title('Original Structure')
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    
    # Plot reconstructed coordinates
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(reconstructed_coords[:, 0], reconstructed_coords[:, 1], reconstructed_coords[:, 2], 
               c='red', alpha=0.6, s=1)
    ax2.set_title('Reconstructed Structure')
    ax2.set_xlabel('Eigenvector 1')
    ax2.set_ylabel('Eigenvector 2')
    ax2.set_zlabel('Eigenvector 3')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Coordinate comparison visualization saved to: {filename}")


def save_reconstructed_coordinates(coords: np.ndarray, filename: str = "reconstructed_coords.csv"):
    """
    Save reconstructed coordinates to CSV file.
    
    Parameters:
    -----------
    coords : np.ndarray
        Reconstructed coordinates
    filename : str
        Output filename
    """
    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df.to_csv(filename, index=False)
    print(f"Reconstructed coordinates saved to: {filename}")


def load_laplacian_matrix(filename: str) -> np.ndarray:
    """
    Load Laplacian matrix from CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to Laplacian matrix CSV file
        
    Returns:
    --------
    laplacian_matrix : np.ndarray
        The loaded Laplacian matrix
    """
    return np.loadtxt(filename, delimiter=",")


def load_original_coordinates(filename: str) -> np.ndarray:
    """
    Load original coordinates from CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to original coordinates CSV file
        
    Returns:
    --------
    coordinates : np.ndarray
        The loaded coordinates
    """
    return pd.read_csv(filename).values


if __name__ == "__main__":
    # Load Laplacian matrix
    print("Loading Laplacian matrix...")
    laplacian_matrix = load_laplacian_matrix("laplacian_matrix_6vsb.csv")
    
    # Load original coordinates for comparison
    print("Loading original coordinates...")
    original_coords = load_original_coordinates("ca_coordinates_6vsb.csv")
    
    # Step 5: Calculate eigendecomposition
    print("\nStep 5: Calculating eigendecomposition...")
    eigenvalues, eigenvectors = calculate_eigendecomposition(laplacian_matrix)
    
    # Step 6: Reconstruct coordinates
    print("\nStep 6: Reconstructing coordinates...")
    reconstructed_coords = reconstruct_coordinates(eigenvectors)
    
    # Save results
    save_reconstructed_coordinates(reconstructed_coords, "reconstructed_coords_6vsb.csv")
    
    # Create visualization
    visualize_original_vs_reconstructed(original_coords, reconstructed_coords, 
                                      "coordinate_comparison_6vsb.png")
    
    # Display some statistics
    print(f"\nReconstruction completed successfully!")
    print(f"Original structure: {len(original_coords)} residues")
    print(f"Reconstructed structure: {len(reconstructed_coords)} residues")
    
    # Save eigenvalues for later analysis
    np.savetxt("eigenvalues_6vsb.csv", eigenvalues, delimiter=",", fmt='%.8f')
    print(f"Eigenvalues saved to: eigenvalues_6vsb.csv")
    
    print(f"\nNext steps: Run alignment.py for Procrustes analysis (Step 7)")