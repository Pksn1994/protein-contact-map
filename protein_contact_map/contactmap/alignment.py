"""
Step 7: Procrustes alignment and Step 8: RMSD calculation
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def center_coordinates(coords: np.ndarray) -> tuple:
    """
    Center coordinates by subtracting the centroid.
    
    Parameters:
    -----------
    coords : np.ndarray
        Nx3 coordinate array
        
    Returns:
    --------
    centered_coords : np.ndarray
        Centered coordinates
    centroid : np.ndarray
        The centroid that was subtracted
    """
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    return centered_coords, centroid


def procrustes_alignment(coords1: np.ndarray, coords2: np.ndarray) -> tuple:
    """
    Align coords2 to coords1 using Procrustes analysis.
    
    This finds the optimal rotation, translation, and scaling to align
    two sets of coordinates while minimizing RMSD.
    
    Parameters:
    -----------
    coords1 : np.ndarray
        Reference coordinates (Nx3)
    coords2 : np.ndarray
        Coordinates to align (Nx3)
        
    Returns:
    --------
    aligned_coords : np.ndarray
        Aligned version of coords2
    transformation : dict
        Dictionary containing rotation matrix, translation, and scaling
    """
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have the same shape")
    
    # Center both coordinate sets
    coords1_centered, centroid1 = center_coordinates(coords1)
    coords2_centered, centroid2 = center_coordinates(coords2)
    
    # Calculate cross-covariance matrix
    H = coords2_centered.T @ coords1_centered
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Calculate rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation to centered coords2
    coords2_rotated = coords2_centered @ R.T
    
    # Calculate optimal scaling factor
    numerator = np.sum(coords1_centered * coords2_rotated)
    denominator = np.sum(coords2_centered * coords2_centered)
    scale = numerator / denominator if denominator > 0 else 1.0
    
    # Apply scaling
    coords2_scaled = coords2_rotated * scale
    
    # Apply translation to match centroid1
    aligned_coords = coords2_scaled + centroid1
    
    transformation = {
        'rotation': R,
        'translation': centroid1 - centroid2,
        'scaling': scale,
        'centroid1': centroid1,
        'centroid2': centroid2
    }
    
    return aligned_coords, transformation


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculate Root Mean Square Deviation between two coordinate sets.
    
    Parameters:
    -----------
    coords1, coords2 : np.ndarray
        Coordinate arrays of shape (N, 3)
        
    Returns:
    --------
    rmsd : float
        RMSD value in the same units as coordinates (typically Angstroms)
    """
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have the same shape")
    
    # Calculate squared differences
    diff = coords1 - coords2
    squared_distances = np.sum(diff**2, axis=1)
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(squared_distances))
    
    return rmsd


def visualize_alignment(original: np.ndarray, reconstructed: np.ndarray, 
                       aligned: np.ndarray, filename: str = "alignment_comparison.png"):
    """
    Visualize original, reconstructed, and aligned structures.
    
    Parameters:
    -----------
    original : np.ndarray
        Original coordinates
    reconstructed : np.ndarray
        Reconstructed coordinates (before alignment)
    aligned : np.ndarray
        Aligned reconstructed coordinates
    filename : str
        Output filename
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Original structure
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], 
               c='blue', alpha=0.7, s=1, label='Original')
    ax1.set_title('Original Structure')
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    
    # Reconstructed (before alignment)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], 
               c='red', alpha=0.7, s=1, label='Reconstructed')
    ax2.set_title('Reconstructed (Before Alignment)')
    ax2.set_xlabel('Eigenvector 1')
    ax2.set_ylabel('Eigenvector 2')
    ax2.set_zlabel('Eigenvector 3')
    
    # Overlay after alignment
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(original[:, 0], original[:, 1], original[:, 2], 
               c='blue', alpha=0.5, s=1, label='Original')
    ax3.scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2], 
               c='red', alpha=0.5, s=1, label='Aligned')
    ax3.set_title('Overlay: Original vs Aligned')
    ax3.set_xlabel('X (Å)')
    ax3.set_ylabel('Y (Å)')
    ax3.set_zlabel('Z (Å)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Alignment visualization saved to: {filename}")


def plot_rmsd_per_residue(original: np.ndarray, aligned: np.ndarray, 
                         filename: str = "rmsd_per_residue.png"):
    """
    Plot RMSD for each residue individually.
    
    Parameters:
    -----------
    original : np.ndarray
        Original coordinates
    aligned : np.ndarray
        Aligned coordinates
    filename : str
        Output filename
    """
    # Calculate per-residue deviations
    diff = original - aligned
    per_residue_rmsd = np.sqrt(np.sum(diff**2, axis=1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(per_residue_rmsd, 'b-', alpha=0.7, linewidth=1)
    plt.xlabel('Residue Index')
    plt.ylabel('RMSD (Å)')
    plt.title('Per-Residue RMSD: Original vs Reconstructed Structure')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_rmsd = np.mean(per_residue_rmsd)
    max_rmsd = np.max(per_residue_rmsd)
    plt.axhline(y=mean_rmsd, color='red', linestyle='--', 
                label=f'Mean RMSD: {mean_rmsd:.3f} Å')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Per-residue RMSD plot saved to: {filename}")
    
    return per_residue_rmsd


def save_alignment_results(aligned_coords: np.ndarray, transformation: dict, 
                          rmsd: float, filename_prefix: str = "alignment_results"):
    """
    Save alignment results to files.
    
    Parameters:
    -----------
    aligned_coords : np.ndarray
        Aligned coordinates
    transformation : dict
        Transformation parameters
    rmsd : float
        Overall RMSD
    filename_prefix : str
        Prefix for output filenames
    """
    # Save aligned coordinates
    coords_file = f"{filename_prefix}_aligned_coords.csv"
    df = pd.DataFrame(aligned_coords, columns=['x', 'y', 'z'])
    df.to_csv(coords_file, index=False)
    
    # Save transformation parameters
    transform_file = f"{filename_prefix}_transformation.txt"
    with open(transform_file, 'w') as f:
        f.write(f"Procrustes Alignment Results\n")
        f.write(f"===========================\n\n")
        f.write(f"Overall RMSD: {rmsd:.6f} Å\n\n")
        f.write(f"Rotation Matrix:\n")
        for row in transformation['rotation']:
            f.write(f"  {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}\n")
        f.write(f"\nTranslation Vector:\n")
        f.write(f"  {transformation['translation'][0]:10.6f} ")
        f.write(f"{transformation['translation'][1]:10.6f} ")
        f.write(f"{transformation['translation'][2]:10.6f}\n")
        f.write(f"\nScaling Factor: {transformation['scaling']:.6f}\n")
        f.write(f"\nCentroid 1 (Original): {transformation['centroid1']}\n")
        f.write(f"Centroid 2 (Reconstructed): {transformation['centroid2']}\n")
    
    print(f"Aligned coordinates saved to: {coords_file}")
    print(f"Transformation parameters saved to: {transform_file}")


def load_coordinates(filename: str) -> np.ndarray:
    """
    Load coordinates from CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to CSV file
        
    Returns:
    --------
    coordinates : np.ndarray
        Loaded coordinates
    """
    return pd.read_csv(filename).values


if __name__ == "__main__":
    # Load original and reconstructed coordinates
    print("Loading coordinates...")
    original_coords = load_coordinates("ca_coordinates_6vsb.csv")
    reconstructed_coords = load_coordinates("reconstructed_coords_6vsb.csv")
    
    print(f"Original coordinates shape: {original_coords.shape}")
    print(f"Reconstructed coordinates shape: {reconstructed_coords.shape}")
    
    # Step 7: Procrustes alignment
    print("\nStep 7: Performing Procrustes alignment...")
    aligned_coords, transformation = procrustes_alignment(original_coords, reconstructed_coords)
    
    # Step 8: Calculate RMSD
    print("\nStep 8: Calculating RMSD...")
    rmsd = calculate_rmsd(original_coords, aligned_coords)
    
    print(f"\nAlignment Results:")
    print(f"Overall RMSD: {rmsd:.6f} Å")
    print(f"Scaling factor: {transformation['scaling']:.6f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_alignment(original_coords, reconstructed_coords, aligned_coords, 
                       "alignment_comparison_6vsb.png")
    
    per_residue_rmsd = plot_rmsd_per_residue(original_coords, aligned_coords, 
                                           "rmsd_per_residue_6vsb.png")
    
    # Save results
    save_alignment_results(aligned_coords, transformation, rmsd, "alignment_6vsb")
    
    # Additional statistics
    print(f"\nDetailed Statistics:")
    print(f"Mean per-residue RMSD: {np.mean(per_residue_rmsd):.6f} Å")
    print(f"Std per-residue RMSD: {np.std(per_residue_rmsd):.6f} Å")
    print(f"Max per-residue RMSD: {np.max(per_residue_rmsd):.6f} Å")
    print(f"Min per-residue RMSD: {np.min(per_residue_rmsd):.6f} Å")
    
    print(f"\nProcrustes alignment completed successfully!")
    print(f"Next step: Run link_analysis.py for Step 9 (link importance analysis)")