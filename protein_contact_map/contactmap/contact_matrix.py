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


if __name__ == "__main__":
    # Load distance matrix (assuming it was created by distance_calculator.py)
    print("Loading distance matrix...")
    distance_matrix = load_distance_matrix("distance_matrix_6vsb.csv")
    
    if distance_matrix is None:
        print("Exiting - could not load distance matrix")
        exit(1)
    
    # Create contact matrix with 7 Angstrom threshold
    print("Creating contact matrix with 7Å threshold...")
    # contact_matrix = create_contact_matrix(distance_matrix, threshold=9.0)
    contact_matrix = create_contact_matrix(distance_matrix, threshold=7.0)
    
    # Analyze the contact matrix
    stats = analyze_contact_matrix(contact_matrix)
    
    print(f"\nContact matrix creation completed!")
    print(f"Matrix dimensions: {contact_matrix.shape}")
    print(f"Ready for visualization and saving in next step...")