import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
from tqdm import tqdm


def get_contact_list(contact_matrix: np.ndarray) -> list:
    """
    Get list of all contacts (links) from contact matrix.
    
    Parameters:
    -----------
    contact_matrix : np.ndarray
        Binary contact matrix
        
    Returns:
    --------
    contact_list : list
        List of tuples (i, j) representing contacts
    """
    # Get upper triangular part to avoid duplicates
    contacts = np.where(np.triu(contact_matrix, k=1))
    contact_list = list(zip(contacts[0], contacts[1]))
    
    return contact_list


def reconstruct_with_modified_contacts(original_coords: np.ndarray, 
                                     contact_matrix: np.ndarray) -> tuple:
    """
    Perform full reconstruction pipeline with given contact matrix.
    
    Parameters:
    -----------
    original_coords : np.ndarray
        Original coordinates for alignment
    contact_matrix : np.ndarray
        Contact matrix (possibly modified)
        
    Returns:
    --------
    rmsd : float
        RMSD after reconstruction and alignment
    aligned_coords : np.ndarray
        Aligned reconstructed coordinates
    """
    # Calculate Laplacian matrix
    degrees = np.sum(contact_matrix, axis=1)
    degree_matrix = np.diag(degrees)
    laplacian_matrix = degree_matrix - contact_matrix
    
    # Eigendecomposition
    eigenvalues, eigenvectors = eigh(laplacian_matrix)
    
    # Check if graph is connected (second eigenvalue should be > 0)
    if eigenvalues[1] < 1e-10:
        return np.inf, None  # Disconnected graph
    
    # Reconstruct coordinates using eigenvectors 1, 2, 3
    reconstructed_coords = eigenvectors[:, 1:4]
    
    # Procrustes alignment
    aligned_coords = procrustes_alignment_simple(original_coords, reconstructed_coords)
    
    # Calculate RMSD
    diff = original_coords - aligned_coords
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd, aligned_coords


def procrustes_alignment_simple(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Simplified Procrustes alignment (translation + rotation only).
    
    Parameters:
    -----------
    coords1 : np.ndarray
        Reference coordinates
    coords2 : np.ndarray
        Coordinates to align
        
    Returns:
    --------
    aligned_coords : np.ndarray
        Aligned coordinates
    """
    # Center coordinates
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    
    # Calculate optimal rotation
    H = coords2_centered.T @ coords1_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply transformation
    aligned_coords = coords2_centered @ R.T + centroid1
    
    return aligned_coords


def analyze_link_importance(original_coords: np.ndarray, 
                          contact_matrix: np.ndarray,
                          max_links: int = None) -> pd.DataFrame:
    """
    Analyze importance of each contact by removal.
    
    Parameters:
    -----------
    original_coords : np.ndarray
        Original coordinates
    contact_matrix : np.ndarray
        Contact matrix
    max_links : int
        Maximum number of links to test (None for all)
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with link importance results
    """
    # Get baseline RMSD
    print("Calculating baseline RMSD...")
    baseline_rmsd, _ = reconstruct_with_modified_contacts(original_coords, contact_matrix)
    print(f"Baseline RMSD: {baseline_rmsd:.6f} Å")
    
    # Get all contacts
    contact_list = get_contact_list(contact_matrix)
    total_contacts = len(contact_list)
    
    if max_links is not None:
        contact_list = contact_list[:max_links]
        print(f"Testing {len(contact_list)} out of {total_contacts} contacts")
    else:
        print(f"Testing all {total_contacts} contacts")
    
    results = []
    
    print("Analyzing link importance...")
    for idx, (i, j) in enumerate(tqdm(contact_list, desc="Testing contacts")):
        # Create modified contact matrix
        modified_contact = contact_matrix.copy()
        modified_contact[i, j] = 0
        modified_contact[j, i] = 0
        
        # Reconstruct with modified matrix
        try:
            rmsd, _ = reconstruct_with_modified_contacts(original_coords, modified_contact)
            rmsd_change = rmsd - baseline_rmsd
            
            results.append({
                'contact': (i, j),
                'residue_i': i,
                'residue_j': j,
                'distance': abs(i - j),  # Sequence separation
                'baseline_rmsd': baseline_rmsd,
                'modified_rmsd': rmsd,
                'rmsd_change': rmsd_change,
                'relative_change': rmsd_change / baseline_rmsd * 100
            })
            
        except Exception as e:
            print(f"Error processing contact ({i}, {j}): {e}")
            results.append({
                'contact': (i, j),
                'residue_i': i,
                'residue_j': j,
                'distance': abs(i - j),
                'baseline_rmsd': baseline_rmsd,
                'modified_rmsd': np.inf,
                'rmsd_change': np.inf,
                'relative_change': np.inf
            })
    
    # Convert to DataFrame and sort by importance
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmsd_change', ascending=False)
    
    return results_df


def visualize_link_importance(results_df: pd.DataFrame, 
                            filename: str = "link_importance.png",
                            top_n: int = 20):
    """
    Visualize the most important links.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from link importance analysis
    filename : str
        Output filename
    top_n : int
        Number of top links to show
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Top N most important links
    top_links = results_df.head(top_n)
    ax1.barh(range(len(top_links)), top_links['rmsd_change'])
    ax1.set_yticks(range(len(top_links)))
    ax1.set_yticklabels([f"{row['residue_i']}-{row['residue_j']}" 
                        for _, row in top_links.iterrows()])
    ax1.set_xlabel('RMSD Change (Å)')
    ax1.set_title(f'Top {top_n} Most Important Contacts')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RMSD change vs sequence separation
    finite_results = results_df[results_df['rmsd_change'] != np.inf]
    ax2.scatter(finite_results['distance'], finite_results['rmsd_change'], 
               alpha=0.6, s=10)
    ax2.set_xlabel('Sequence Separation')
    ax2.set_ylabel('RMSD Change (Å)')
    ax2.set_title('RMSD Change vs Sequence Separation')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of RMSD changes
    ax3.hist(finite_results['rmsd_change'], bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('RMSD Change (Å)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of RMSD Changes')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Contact map with importance
    contact_matrix = np.zeros((results_df['residue_i'].max() + 1, 
                             results_df['residue_j'].max() + 1))
    
    for _, row in finite_results.iterrows():
        i, j = row['residue_i'], row['residue_j']
        importance = row['rmsd_change']
        contact_matrix[i, j] = importance
        contact_matrix[j, i] = importance
    
    im = ax4.imshow(contact_matrix, cmap='Reds', interpolation='nearest')
    ax4.set_title('Contact Importance Map')
    ax4.set_xlabel('Residue Index')
    ax4.set_ylabel('Residue Index')
    plt.colorbar(im, ax=ax4, label='RMSD Change (Å)')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Link importance visualization saved to: {filename}")


def save_results(results_df: pd.DataFrame, filename: str = "link_importance_results.csv"):
    """
    Save link importance results to CSV file.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results DataFrame
    filename : str
        Output filename
    """
    results_df.to_csv(filename, index=False)
    print(f"Link importance results saved to: {filename}")


def generate_summary_report(results_df: pd.DataFrame) -> str:
    """
    Generate a summary report of the link importance analysis.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results DataFrame
        
    Returns:
    --------
    report : str
        Summary report text
    """
    finite_results = results_df[results_df['rmsd_change'] != np.inf]
    
    report = f"""
Link Importance Analysis Report
===============================

Total contacts analyzed: {len(results_df)}
Successful reconstructions: {len(finite_results)}
Failed reconstructions: {len(results_df) - len(finite_results)}

RMSD Change Statistics:
- Mean: {finite_results['rmsd_change'].mean():.6f} Å
- Std:  {finite_results['rmsd_change'].std():.6f} Å
- Min:  {finite_results['rmsd_change'].min():.6f} Å
- Max:  {finite_results['rmsd_change'].max():.6f} Å

Top 10 Most Critical Contacts:
"""
    
    for i, (_, row) in enumerate(finite_results.head(10).iterrows()):
        report += f"{i+1:2d}. Residues {row['residue_i']:4d}-{row['residue_j']:4d} "
        report += f"(sep: {row['distance']:3d}): RMSD change = {row['rmsd_change']:.6f} Å\n"
    
    return report


def load_data_files():
    """Load required data files."""
    original_coords = pd.read_csv("ca_coordinates_6vsb.csv").values
    contact_matrix = np.loadtxt("contact_matrix_6vsb.csv", delimiter=",", dtype=int)
    return original_coords, contact_matrix


if __name__ == "__main__":
    print("Loading data files...")
    original_coords, contact_matrix = load_data_files()
    
    print(f"Loaded {len(original_coords)} residues")
    print(f"Contact matrix shape: {contact_matrix.shape}")
    print(f"Total contacts: {np.sum(contact_matrix) // 2}")
    
    # Perform link importance analysis
    # Note: For large proteins, you might want to set max_links to a smaller number
    # for faster computation (e.g., max_links=1000)
    results_df = analyze_link_importance(original_coords, contact_matrix, max_links=1000)
    
    # Save results
    save_results(results_df, "link_importance_6vsb.csv")
    
    # Create visualizations
    visualize_link_importance(results_df, "link_importance_6vsb.png", top_n=20)
    
    # Generate and save summary report
    report = generate_summary_report(results_df)
    print(report)
    
    with open("link_importance_report_6vsb.txt", "w") as f:
        f.write(report)
    
    print(f"\nStep 9 completed successfully!")
    print(f"Analysis complete! Check the generated files for results.")