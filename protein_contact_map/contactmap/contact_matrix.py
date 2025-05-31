"""
Step 3: Create contact matrix by thresholding distance matrix
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_contact_matrix(distance_matrix: np.ndarray, threshold: float = 7.0) -> np.ndarray:
    """
    Create binary contact matrix from distance matrix using threshold.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Symmetric distance matrix of shape (N, N)
    threshold : float
        Distance threshold in Angstroms (default: 7.0)
        
    Returns:
    --------
    contact_matrix : np.ndarray
        Binary contact matrix where 1 indicates contact (distance <= threshold)
    """
    # Create binary matrix: 1 if distance <= threshold, 0 otherwise
    contact_matrix = (distance_matrix <= threshold).astype(int)
    
    # Remove self-contacts (diagonal should be 0)
    np.fill_diagonal(contact_matrix, 0)
    
    # print(f"Contact matrix created, shape: {contact_matrix.shape}")  # check dimensions
    
    return contact_matrix


def analyze_contact_matrix(contact_matrix: np.ndarray):
    """
    Analyze properties of the contact matrix.
    
    Parameters:
    -----------
    contact_matrix : np.ndarray
        Binary contact matrix
    """
    n_residues = contact_matrix.shape[0]
    n_contacts = np.sum(contact_matrix) // 2  # Divide by 2 since matrix is symmetric
    
    # Calculate degree (number of contacts per residue)
    degrees = np.sum(contact_matrix, axis=1)
    
    print(f"Contact Matrix Analysis:")
    print(f"Number of residues: {n_residues}")
    print(f"Total contacts: {n_contacts}")
    print(f"Contact density: {n_contacts / (n_residues * (n_residues - 1) / 2):.4f}")
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Min degree: {np.min(degrees)}")
    print(f"Max degree: {np.max(degrees)}")
    
    return {
        'n_residues': n_residues,
        'n_contacts': n_contacts,
        'degrees': degrees
    }


def load_distance_matrix(filename: str) -> np.ndarray:
    """
    Load distance matrix from CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to distance matrix CSV file
        
    Returns:
    --------
    distance_matrix : np.ndarray
        The loaded distance matrix
    """
    # return np.loadtxt(filename, delimiter=",")  # original
    try:
        distance_matrix = np.loadtxt(filename, delimiter=",")
        print(f"Distance matrix loaded: {distance_matrix.shape}")
        return distance_matrix
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None


def visualize_contact_matrix(contact_matrix: np.ndarray, filename: str = "contact_matrix.png"):
    """
    Create visualization of the contact matrix.
    
    Parameters:
    -----------
    contact_matrix : np.ndarray
        Binary contact matrix
    filename : str
        Output filename for the plot
    """
    # Check if matrix is too large for visualization
    if contact_matrix.shape[0] > 1000:
        print(f"Warning: Large matrix ({contact_matrix.shape[0]}x{contact_matrix.shape[0]}), visualization might be slow")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(contact_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Contact (1) / No Contact (0)')
    plt.title('Protein Contact Matrix (7Å threshold)')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Contact matrix visualization saved to: {filename}")


def save_contact_matrix(contact_matrix: np.ndarray, filename: str = "contact_matrix.csv"):
    """
    Save contact matrix to CSV file.
    
    Parameters:
    -----------
    contact_matrix : np.ndarray
        Binary contact matrix
    filename : str
        Output filename
    """
    np.savetxt(filename, contact_matrix, delimiter=",", fmt='%d')
    print(f"Contact matrix saved to: {filename}")


if __name__ == "__main__":
    # Load distance matrix (assuming it was created by distance_calculator.py)
    print("Loading distance matrix...")
    distance_matrix = load_distance_matrix("distance_matrix_6vsb.csv")
    
    if distance_matrix is None:
        print("Exiting - could not load distance matrix")
        exit(1)
    
    # Create contact matrix with 7 Angstrom threshold
    print("Creating contact matrix with 7Å threshold...")
    contact_matrix = create_contact_matrix(distance_matrix, threshold=7.0)
    
    # Analyze the contact matrix
    stats = analyze_contact_matrix(contact_matrix)
    
    # Save contact matrix
    save_contact_matrix(contact_matrix, "contact_matrix_6vsb.csv")
    
    # Create visualization
    visualize_contact_matrix(contact_matrix, "contact_matrix_6vsb.png")
    
    # Show some example contacts
    print("\nExample contacts (first 10 residues):")
    for i in range(min(10, contact_matrix.shape[0])):
        contacts = np.where(contact_matrix[i] == 1)[0]
        if len(contacts) > 0:
            print(f"Residue {i}: contacts with residues {contacts[:10]}")  # Show first 10 contacts
        # else:
        #     print(f"Residue {i}: no contacts found")  # debugging isolated residues