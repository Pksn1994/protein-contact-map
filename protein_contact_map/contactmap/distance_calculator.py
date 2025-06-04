"""
Step 2: Calculate pairwise distance matrix between all alpha-Carbons
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd


def calculate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise Euclidean distances between all alpha-carbon atoms.
    
    Parameters:
    -----------
    coordinates : np.ndarray
        Array of shape (N, 3) containing x, y, z coordinates of N alpha-carbons
        
    Returns:
    --------
    distance_matrix : np.ndarray
        Symmetric matrix of shape (N, N) containing pairwise distances
    """
    # Print shape to check input - helps with debugging
    print(f"coords shape: {coordinates.shape}")
    
    # Tried manually first
    # n = len(coordinates)
    # distance_matrix = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(i+1, n):
    #         # Calculate Euclidean distance formula sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    #         dist = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
    #         distance_matrix[i, j] = dist
    #         distance_matrix[j, i] = dist  # symmetric matrix
    
    # this doesn't work - gives weird shape
    # all_dists = []
    # for i, coord1 in enumerate(coordinates):
    #    dists = np.sqrt(np.sum((coordinates - coord1)**2, axis=1))
    #    all_dists.append(dists)
    # distance_matrix = np.array(all_dists)
    
    # Calculate pairwise distances using scipy
    distances = pdist(coordinates, metric='euclidean')
    
    # Convert to square matrix format
    distance_matrix = squareform(distances)
    
    # Sanity check - look at a small part
    if len(distance_matrix) > 3:
        print("first few dist values:")
        print(distance_matrix[:3, :3])
    
    return distance_matrix


def save_distance_matrix(distance_matrix: np.ndarray, filename: str = "distance_matrix.csv"):
    """
    Save distance matrix to CSV file.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        The distance matrix to save
    filename : str
        Output filename
    """
    # trying to check matrix properties first
    print(f"matrix size: {distance_matrix.shape}")
    
    # save as txt first? nah too hard to read
    # with open(filename.replace('.csv', '.txt'), 'w') as f:
    #     for row in distance_matrix:
    #         f.write(' '.join([f"{x:.2f}" for x in row]) + '\n')
    
    # Should I use pandas? Let me try...
    # df = pd.DataFrame(distance_matrix)
    # df.to_csv(filename)  # this adds index columns we don't need
    
    # simpler to just use numpy
    np.savetxt(filename, distance_matrix, delimiter=",", fmt='%.3f')
    print(f"saved to: {filename}")


def load_coordinates_from_csv(filename: str) -> np.ndarray:
    """
    Load coordinates from CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to CSV file with coordinates
        
    Returns:
    --------
    coordinates : np.ndarray
        Array of coordinates
    """
    return pd.read_csv(filename).values


if __name__ == "__main__":
    # Load the alpha-carbon coordinates
    coordinates = load_coordinates_from_csv("ca_coordinates_6vsb.csv")
    print(f"Loaded {len(coordinates)} alpha-carbon coordinates")
    
    # Calculate distance matrix
    print("Calculating distance matrix...")
    dist_matrix = calculate_distance_matrix(coordinates)
    
    # Display some statistics
    print(f"Distance matrix shape: {dist_matrix.shape}")
    print(f"Min distance (non-zero): {np.min(dist_matrix[dist_matrix > 0]):.3f} Å")
    print(f"Max distance: {np.max(dist_matrix):.3f} Å")
    print(f"Mean distance: {np.mean(dist_matrix[dist_matrix > 0]):.3f} Å")
    
    # Save the matrix
    save_distance_matrix(dist_matrix, "distance_matrix_6vsb.csv")
    
    # Show a small sample of the matrix
    print("\nFirst 5x5 subset of distance matrix:")
    print(dist_matrix[:5, :5])