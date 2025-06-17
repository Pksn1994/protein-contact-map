import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import modules to test
from distance_calculator import calculate_distance_matrix
from contact_matrix import create_contact_matrix, analyze_contact_matrix
from laplacian_calculator import compute_laplacian_matrix
from reconstruction import reconstruct_coordinates
from alignment import center_coordinates, calculate_rmsd, procrustes_alignment
from link_analysis import get_contact_list


class TestDistanceCalculator(unittest.TestCase):
    """Test distance matrix calculation"""
    
    def test_distance_matrix_is_symmetric(self):
        """Distance matrix should be symmetric"""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        result = calculate_distance_matrix(coords)
        
        # Check if matrix equals its transpose
        self.assertTrue(np.allclose(result, result.T))
    
    def test_distance_matrix_diagonal_is_zero(self):
        """Diagonal should be zero (distance from point to itself)"""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        result = calculate_distance_matrix(coords)
        
        # Check diagonal elements
        self.assertAlmostEqual(result[0, 0], 0.0, places=10)
        self.assertAlmostEqual(result[1, 1], 0.0, places=10)
    
    def test_known_distance_calculation(self):
        """Test with known distance - simple 3-4-5 triangle"""
        coords = np.array([[0, 0, 0], [3, 4, 0]])
        result = calculate_distance_matrix(coords)
        
        expected_distance = 5.0  # sqrt(3^2 + 4^2)
        self.assertAlmostEqual(result[0, 1], expected_distance, places=6)


class TestContactMatrix(unittest.TestCase):
    """Test contact matrix creation"""
    
    def test_contact_matrix_threshold_works(self):
        """Contacts should only exist below threshold"""
        dist_matrix = np.array([[0, 5, 10], [5, 0, 15], [10, 15, 0]])
        threshold = 7.0
        
        result = create_contact_matrix(dist_matrix, threshold)
        expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_contact_matrix_no_self_contacts(self):
        """Diagonal should be zero (no self-contacts)"""
        dist_matrix = np.array([[0, 3, 8], [3, 0, 4], [8, 4, 0]])
        result = create_contact_matrix(dist_matrix, 10.0)
        
        # Check diagonal is zero
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[1, 1], 0)
        self.assertEqual(result[2, 2], 0)
    
    def test_analyze_contact_matrix_counts_correctly(self):
        """Check if contact analysis gives right numbers"""
        contact_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        
        stats = analyze_contact_matrix(contact_matrix)
        
        self.assertEqual(stats['n_residues'], 3)
        self.assertEqual(stats['n_contacts'], 3)  # 6 total / 2 for symmetry


class TestLaplacianCalculator(unittest.TestCase):
    """Test Laplacian matrix computation"""
    
    def test_laplacian_matrix_structure(self):
        """Test basic Laplacian structure: L = D - A"""
        contact_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        
        result = compute_laplacian_matrix(contact_matrix)
        expected = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_laplacian_row_sums_zero(self):
        """Laplacian matrix rows should sum to zero"""
        contact_matrix = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        
        result = compute_laplacian_matrix(contact_matrix)
        row_sums = np.sum(result, axis=1)
        
        # Check all row sums are close to zero
        for row_sum in row_sums:
            self.assertAlmostEqual(row_sum, 0.0, places=10)


class TestReconstruction(unittest.TestCase):
    """Test coordinate reconstruction"""
    
    def test_reconstruct_coordinates_correct_shape(self):
        """Reconstructed coordinates should be Nx3"""
        # Make fake eigenvectors (5 residues)
        eigenvectors = np.random.rand(5, 5)
        
        result = reconstruct_coordinates(eigenvectors)
        
        self.assertEqual(result.shape, (5, 3))
    
    def test_reconstruct_uses_right_eigenvectors(self):
        """Should use eigenvectors 1, 2, 3 (skip first)"""
        eigenvectors = np.eye(4)  # Identity for easy testing
        
        result = reconstruct_coordinates(eigenvectors)
        expected = eigenvectors[:, 1:4]
        
        np.testing.assert_array_equal(result, expected)


class TestAlignment(unittest.TestCase):
    """Test alignment and RMSD functions"""
    
    def test_center_coordinates_removes_mean(self):
        """Centering should make mean zero"""
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        centered, centroid = center_coordinates(coords)
        
        # Mean should be zero after centering
        mean_after = np.mean(centered, axis=0)
        self.assertAlmostEqual(mean_after[0], 0.0, places=10)
        self.assertAlmostEqual(mean_after[1], 0.0, places=10)
        self.assertAlmostEqual(mean_after[2], 0.0, places=10)
    
    def test_rmsd_identical_structures(self):
        """RMSD should be zero for identical structures"""
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        
        rmsd = calculate_rmsd(coords, coords)
        
        self.assertAlmostEqual(rmsd, 0.0, places=10)
    
    def test_rmsd_known_displacement(self):
        """Test RMSD with known displacement"""
        coords1 = np.array([[0, 0, 0], [1, 0, 0]])
        coords2 = np.array([[1, 1, 1], [2, 1, 1]])  # Each point moved by (1,1,1)
        
        rmsd = calculate_rmsd(coords1, coords2)
        expected = np.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        
        self.assertAlmostEqual(rmsd, expected, places=6)
    
    def test_procrustes_alignment_improves_rmsd(self):
        """Alignment should reduce RMSD"""
        # Original coordinates
        original = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        # Same coordinates but translated
        translated = original + [2, 3, 1]
        
        # before alignment
        rmsd_before = calculate_rmsd(original, translated)
        
        # after alignment
        aligned, _ = procrustes_alignment(original, translated)
        rmsd_after = calculate_rmsd(original, aligned)
        
        # Should be much better after alignment
        self.assertLess(rmsd_after, rmsd_before)


class TestLinkAnalysis(unittest.TestCase):
    """Test link analysis functions"""
    
    def test_get_contact_list_no_duplicates(self):
        """Contact list should not have duplicate pairs"""
        contact_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        
        contacts = get_contact_list(contact_matrix)
        
        # Should only get upper triangular contacts
        self.assertEqual(len(contacts), 2)  # (0,1) and (0,2)
        # Check specific contacts
        self.assertIn((0, 1), contacts)
        self.assertIn((0, 2), contacts)


class TestPipelineIntegration(unittest.TestCase):
    """Test that pipeline steps work together"""
    
    def test_small_protein_pipeline(self):
        """Test complete pipeline with small example"""
        # Make small synthetic protein (4 residues in a line)
        coords = np.array([[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]])
        
        # Step 2: Distance matrix
        dist_matrix = calculate_distance_matrix(coords)
        self.assertEqual(dist_matrix.shape, (4, 4))
        
        # Step 3: Contact matrix (threshold 2.0 to get neighbors)
        contact_matrix = create_contact_matrix(dist_matrix, threshold=2.0)
        
        # Should have contacts between neighbors
        self.assertEqual(contact_matrix[0, 1], 1)  # 0-1 are neighbors
        self.assertEqual(contact_matrix[1, 2], 1)  # 1-2 are neighbors
        self.assertEqual(contact_matrix[0, 3], 0)  # 0-3 are far apart
        
        # Step 4: Laplacian
        laplacian = compute_laplacian_matrix(contact_matrix)
        
        # Should be symmetric
        self.assertTrue(np.allclose(laplacian, laplacian.T))
        
        # Rows should sum to zero
        row_sums = np.sum(laplacian, axis=1)
        for row_sum in row_sums:
            self.assertAlmostEqual(row_sum, 0.0, places=10)


class TestFileOperations(unittest.TestCase):
    """Test file save/load operations"""
    
    def test_save_load_coordinates_consistent(self):
        """Saved and loaded coordinates should match"""
        coords = np.array([[1.5, 2.7, 3.1], [4.2, 5.8, 6.3]])
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            np.savetxt(tmp.name, coords, delimiter=",", fmt='%.3f')
            
            # Load back
            loaded = np.loadtxt(tmp.name, delimiter=",")
            
            # Should be close (considering precision)
            np.testing.assert_allclose(coords, loaded, rtol=1e-3)
            
        # Clean up
        os.unlink(tmp.name)


if __name__ == '__main__':
    # Run all tests
    print("Running protein analysis tests...")
    print("=" * 50)
    
    unittest.main(verbosity=2)