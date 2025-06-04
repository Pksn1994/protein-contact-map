    import os
    import time
    import numpy as np
    import pandas as pd
    from typing import Dict, Any

    # Import all analysis modules
    from download_pdb import download_pdb_file
    from extractor import extract_ca_coordinates
    from distance_calculator import calculate_distance_matrix, save_distance_matrix
    from contact_matrix import create_contact_matrix, save_contact_matrix, analyze_contact_matrix
    from laplacian_calculator import compute_laplacian_matrix, save_laplacian_matrix
    from reconstruction import calculate_eigendecomposition, reconstruct_coordinates, save_reconstructed_coordinates
    from alignment import procrustes_alignment, calculate_rmsd, save_alignment_results
    from link_analysis import analyze_link_importance, save_results as save_link_results


    class ProteinAnalysisPipeline:
        """
        Complete pipeline for protein structure analysis using graph Laplacian methods.
        """
        
        def __init__(self, pdb_id: str, output_dir: str = "results"):
            """
            Initialize the analysis pipeline.
            
            Parameters:
            -----------
            pdb_id : str
                4-character PDB ID (e.g., '6vsb')
            output_dir : str
                Directory to save all output files
            """
            self.pdb_id = pdb_id.lower()
            self.output_dir = output_dir
            self.results = {}
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Define file paths
            self.file_paths = {
                'pdb_file': f"pdb{self.pdb_id}.ent",
                'coordinates': f"{output_dir}/ca_coordinates_{self.pdb_id}.csv",
                'distance_matrix': f"{output_dir}/distance_matrix_{self.pdb_id}.csv",
                'contact_matrix': f"{output_dir}/contact_matrix_{self.pdb_id}.csv",
                'laplacian_matrix': f"{output_dir}/laplacian_matrix_{self.pdb_id}.csv",
                'reconstructed_coords': f"{output_dir}/reconstructed_coords_{self.pdb_id}.csv",
                'aligned_coords': f"{output_dir}/aligned_coords_{self.pdb_id}.csv",
                'link_analysis': f"{output_dir}/link_importance_{self.pdb_id}.csv",
                'final_report': f"{output_dir}/analysis_report_{self.pdb_id}.txt"
            }
        
        def step_1_download_and_extract(self) -> np.ndarray:
            """
            Step 1: Download PDB file and extract alpha-carbon coordinates.
            
            Returns:
            --------
            coordinates : np.ndarray
                Alpha-carbon coordinates
            """
            print("="*60)
            print("STEP 1: Download PDB and Extract Alpha-Carbon Coordinates")
            print("="*60)
            
            # Download PDB file if it doesn't exist
            if not os.path.exists(self.file_paths['pdb_file']):
                print(f"Downloading PDB file for {self.pdb_id}...")
                download_pdb_file(self.pdb_id)
            else:
                print(f"PDB file {self.file_paths['pdb_file']} already exists.")
            
            # Extract coordinates
            print("Extracting alpha-carbon coordinates...")
            coordinates = extract_ca_coordinates(self.file_paths['pdb_file'])
            
            # Save coordinates
            np.savetxt(self.file_paths['coordinates'], coordinates, 
                    delimiter=",", header="x,y,z", comments='')
            
            print(f"Extracted {len(coordinates)} alpha-carbon atoms")
            print(f"Coordinates saved to: {self.file_paths['coordinates']}")
            
            self.results['coordinates'] = coordinates
            return coordinates
        
        def step_2_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
            """
            Step 2: Calculate pairwise distance matrix.
            
            Parameters:
            -----------
            coordinates : np.ndarray
                Alpha-carbon coordinates
                
            Returns:
            --------
            distance_matrix : np.ndarray
                Pairwise distance matrix
            """
            print("\n" + "="*60)
            print("STEP 2: Calculate Pairwise Distance Matrix")
            print("="*60)
            
            distance_matrix = calculate_distance_matrix(coordinates)
            save_distance_matrix(distance_matrix, self.file_paths['distance_matrix'])
            
            print(f"Distance matrix shape: {distance_matrix.shape}")
            print(f"Min distance (non-zero): {np.min(distance_matrix[distance_matrix > 0]):.3f} Å")
            print(f"Max distance: {np.max(distance_matrix):.3f} Å")
            
            self.results['distance_matrix'] = distance_matrix
            return distance_matrix
        
        def step_3_contact_matrix(self, distance_matrix: np.ndarray, 
                                threshold: float = 7.0) -> np.ndarray:
            """
            Step 3: Create contact matrix using distance threshold.
            
            Parameters:
            -----------
            distance_matrix : np.ndarray
                Pairwise distance matrix
            threshold : float
                Distance threshold in Angstroms
                
            Returns:
            --------
            contact_matrix : np.ndarray
                Binary contact matrix
            """
            print("\n" + "="*60)
            print(f"STEP 3: Create Contact Matrix (threshold = {threshold}Å)")
            print("="*60)
            
            contact_matrix = create_contact_matrix(distance_matrix, threshold)
            save_contact_matrix(contact_matrix, self.file_paths['contact_matrix'])
            
            # Analyze contact matrix
            stats = analyze_contact_matrix(contact_matrix)
            
            self.results['contact_matrix'] = contact_matrix
            self.results['contact_stats'] = stats
            return contact_matrix
        
        def step_4_laplacian_matrix(self, contact_matrix: np.ndarray) -> np.ndarray:
            """
            Step 4: Compute graph Laplacian matrix.
            
            Parameters:
            -----------
            contact_matrix : np.ndarray
                Binary contact matrix
                
            Returns:
            --------
            laplacian_matrix : np.ndarray
                Graph Laplacian matrix
            """
            print("\n" + "="*60)
            print("STEP 4: Compute Graph Laplacian Matrix")
            print("="*60)
            
            laplacian_matrix = compute_laplacian_matrix(contact_matrix)
            save_laplacian_matrix(laplacian_matrix, self.file_paths['laplacian_matrix'])
            
            # Verify Laplacian properties
            row_sums = np.sum(laplacian_matrix, axis=1)
            print(f"Laplacian matrix shape: {laplacian_matrix.shape}")
            print(f"Row sums are zero: {np.allclose(row_sums, 0)}")
            
            self.results['laplacian_matrix'] = laplacian_matrix
            return laplacian_matrix
        
        def step_5_6_reconstruction(self, laplacian_matrix: np.ndarray) -> np.ndarray:
            """
            Steps 5-6: Eigendecomposition and coordinate reconstruction.
            
            Parameters:
            -----------
            laplacian_matrix : np.ndarray
                Graph Laplacian matrix
                
            Returns:
            --------
            reconstructed_coords : np.ndarray
                Reconstructed coordinates
            """
            print("\n" + "="*60)
            print("STEPS 5-6: Eigendecomposition and Coordinate Reconstruction")
            print("="*60)
            
            # Step 5: Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = calculate_eigendecomposition(laplacian_matrix)
            
            # Step 6: Reconstruct coordinates
            reconstructed_coords = reconstruct_coordinates(eigenvectors)
            save_reconstructed_coordinates(reconstructed_coords, 
                                        self.file_paths['reconstructed_coords'])
            
            print(f"Eigendecomposition completed")
            print(f"Number of zero eigenvalues: {np.sum(eigenvalues < 1e-10)}")
            print(f"Coordinates reconstructed using eigenvectors 1, 2, 3")
            
            self.results['eigenvalues'] = eigenvalues
            self.results['eigenvectors'] = eigenvectors
            self.results['reconstructed_coords'] = reconstructed_coords
            return reconstructed_coords
        
        def step_7_8_alignment_rmsd(self, original_coords: np.ndarray, 
                                reconstructed_coords: np.ndarray) -> tuple:
            """
            Steps 7-8: Procrustes alignment and RMSD calculation.
            
            Parameters:
            -----------
            original_coords : np.ndarray
                Original coordinates
            reconstructed_coords : np.ndarray
                Reconstructed coordinates
                
            Returns:
            --------
            aligned_coords : np.ndarray
                Aligned coordinates
            rmsd : float
                Root Mean Square Deviation
            """
            print("\n" + "="*60)
            print("STEPS 7-8: Procrustes Alignment and RMSD Calculation")
            print("="*60)
            
            # Step 7: Procrustes alignment
            aligned_coords, transformation = procrustes_alignment(original_coords, 
                                                                reconstructed_coords)
            
            # Step 8: Calculate RMSD
            rmsd = calculate_rmsd(original_coords, aligned_coords)
            
            # Save results
            save_alignment_results(aligned_coords, transformation, rmsd, 
                                f"{self.output_dir}/alignment_{self.pdb_id}")
            
            print(f"Procrustes alignment completed")
            print(f"Overall RMSD: {rmsd:.6f} Å")
            print(f"Scaling factor: {transformation['scaling']:.6f}")
            
            self.results['aligned_coords'] = aligned_coords
            self.results['rmsd'] = rmsd
            self.results['transformation'] = transformation
            return aligned_coords, rmsd
        
        def step_9_link_analysis(self, original_coords: np.ndarray, 
                                contact_matrix: np.ndarray, 
                                max_links: int = 1000) -> pd.DataFrame:
            """
            Step 9: Analyze link importance by removing contacts one by one.
            
            Parameters:
            -----------
            original_coords : np.ndarray
                Original coordinates
            contact_matrix : np.ndarray
                Contact matrix
            max_links : int
                Maximum number of links to test
                
            Returns:
            --------
            link_results : pd.DataFrame
                Link importance analysis results
            """
            print("\n" + "="*60)
            print("STEP 9: Link Importance Analysis")
            print("="*60)
            
            print(f"Analyzing importance of contacts (testing up to {max_links} links)...")
            link_results = analyze_link_importance(original_coords, contact_matrix, max_links)
            
            # Save results
            save_link_results(link_results, self.file_paths['link_analysis'])
            
            # Display top results
            finite_results = link_results[link_results['rmsd_change'] != np.inf]
            print(f"\nTop 5 most critical contacts:")
            for i, (_, row) in enumerate(finite_results.head(5).iterrows()):
                print(f"{i+1}. Residues {row['residue_i']}-{row['residue_j']}: "
                    f"RMSD change = {row['rmsd_change']:.6f} Å")
            
            self.results['link_analysis'] = link_results
            return link_results
        
        def generate_final_report(self) -> str:
            """
            Generate comprehensive analysis report.
            
            Returns:
            --------
            report : str
                Final analysis report
            """
            print("\n" + "="*60)
            print("GENERATING FINAL REPORT")
            print("="*60)
            
            coordinates = self.results['coordinates']
            contact_stats = self.results['contact_stats']
            rmsd = self.results['rmsd']
            eigenvalues = self.results['eigenvalues']
            link_analysis = self.results['link_analysis']
            
            finite_links = link_analysis[link_analysis['rmsd_change'] != np.inf]
            
            report = f"""
    PROTEIN STRUCTURE ANALYSIS REPORT
    =================================

    PDB ID: {self.pdb_id.upper()}
    Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

    SUMMARY
    -------
    This analysis reconstructs protein 3D structure using graph Laplacian methods
    and evaluates the importance of individual contacts in maintaining structural integrity.

    STRUCTURE INFORMATION
    --------------------
    Number of residues: {len(coordinates)}
    Coordinate range:
    X: {coordinates[:, 0].min():.2f} to {coordinates[:, 0].max():.2f} Å
    Y: {coordinates[:, 1].min():.2f} to {coordinates[:, 1].max():.2f} Å  
    Z: {coordinates[:, 2].min():.2f} to {coordinates[:, 2].max():.2f} Å

    CONTACT ANALYSIS (7Å threshold)
    ------------------------------
    Total contacts: {contact_stats['n_contacts']}
    Contact density: {contact_stats['n_contacts'] / (len(coordinates) * (len(coordinates) - 1) / 2):.6f}
    Average degree: {np.mean(contact_stats['degrees']):.2f}
    Degree range: {np.min(contact_stats['degrees'])} to {np.max(contact_stats['degrees'])}

    EIGENVALUE ANALYSIS
    ------------------
    Number of zero eigenvalues: {np.sum(eigenvalues < 1e-10)}
    Second smallest eigenvalue: {eigenvalues[1]:.6f}
    Largest eigenvalue: {eigenvalues[-1]:.6f}
    Spectral gap: {eigenvalues[1] - eigenvalues[0]:.6f}

    RECONSTRUCTION QUALITY
    ---------------------
    Overall RMSD: {rmsd:.6f} Å
    Scaling factor: {self.results['transformation']['scaling']:.6f}

    The reconstruction quality indicates {'excellent' if rmsd < 2.0 else 'good' if rmsd < 5.0 else 'moderate'} 
    agreement between original and reconstructed structures.

    LINK IMPORTANCE ANALYSIS
    -----------------------
    Links analyzed: {len(link_analysis)}
    Successful reconstructions: {len(finite_links)}
    Failed reconstructions: {len(link_analysis) - len(finite_links)}

    RMSD change statistics:
    Mean: {finite_links['rmsd_change'].mean():.6f} Å
    Std:  {finite_links['rmsd_change'].std():.6f} Å
    Max:  {finite_links['rmsd_change'].max():.6f} Å

    TOP 10 MOST CRITICAL CONTACTS
    -----------------------------
    """
            
            for i, (_, row) in enumerate(finite_links.head(10).iterrows()):
                report += f"{i+1:2d}. Residues {row['residue_i']:4d}-{row['residue_j']:4d} "
                report += f"(separation: {row['distance']:3d}): "
                report += f"RMSD change = {row['rmsd_change']:.6f} Å "
                report += f"({row['relative_change']:+.2f}%)\n"
            
            report += f"""

    INTERPRETATION
    --------------
    • Contacts with large RMSD changes upon removal are critical for structural integrity
    • Short-range contacts (small sequence separation) often show high importance
    • The analysis identifies key structural elements that maintain protein fold
    • Failed reconstructions indicate contacts essential for graph connectivity

    FILES GENERATED
    --------------
    """
            
            for key, path in self.file_paths.items():
                if os.path.exists(path):
                    report += f"• {key}: {path}\n"
            
            report += f"""
    METHODOLOGY
    -----------
    This analysis follows the 9-step pipeline:
    1. Extract alpha-carbon coordinates from PDB
    2. Calculate pairwise distance matrix
    3. Create binary contact matrix (7Å threshold)
    4. Compute graph Laplacian matrix
    5. Calculate eigenvalues and eigenvectors
    6. Reconstruct coordinates using smallest non-zero eigenvectors
    7. Align reconstructed structure using Procrustes analysis
    8. Calculate RMSD between original and reconstructed structures
    9. Evaluate contact importance by systematic removal

    REFERENCES
    ----------
    • Graph Laplacian methods for protein structure analysis
    • Procrustes analysis for structural alignment
    • Contact map analysis in structural biology
    """
            
            # Save report
            with open(self.file_paths['final_report'], 'w') as f:
                f.write(report)
            
            print(f"Final report saved to: {self.file_paths['final_report']}")
            return report
        
        def run_complete_analysis(self, contact_threshold: float = 7.0, 
                                max_links_test: int = 1000) -> Dict[str, Any]:
            """
            Run the complete 9-step analysis pipeline.
            
            Parameters:
            -----------
            contact_threshold : float
                Distance threshold for contacts (Angstroms)
            max_links_test : int
                Maximum number of links to test in step 9
                
            Returns:
            --------
            results : dict
                Complete analysis results
            """
            start_time = time.time()
            
            print(f"PROTEIN STRUCTURE ANALYSIS PIPELINE")
            print(f"PDB ID: {self.pdb_id.upper()}")
            print(f"Output directory: {self.output_dir}")
            print(f"Contact threshold: {contact_threshold}Å")
            print(f"Max links to test: {max_links_test}")
            
            try:
                # Step 1: Download and extract coordinates
                coordinates = self.step_1_download_and_extract()
                
                # Step 2: Distance matrix
                distance_matrix = self.step_2_distance_matrix(coordinates)
                
                # Step 3: Contact matrix
                contact_matrix = self.step_3_contact_matrix(distance_matrix, contact_threshold)
                
                # Step 4: Laplacian matrix
                laplacian_matrix = self.step_4_laplacian_matrix(contact_matrix)
                
                # Steps 5-6: Reconstruction
                reconstructed_coords = self.step_5_6_reconstruction(laplacian_matrix)
                    
                    # Steps 7-8: Alignment and RMSD
                    aligned_coords, rmsd = self.step_7_8_alignment_rmsd(coordinates, reconstructed_coords)
                    
                    # Step 9: Link analysis
                    link_results = self.step_9_link_analysis(coordinates, contact_matrix, max_links_test)
                    
                    # Generate final report
                    report = self.generate_final_report()
                    
                    # Calculate total time
                    total_time = time.time() - start_time
                    
                print(f"\n" + "="*60)
                print("ANALYSIS COMPLETED SUCCESSFULLY!")
                print("="*60)
                print(f"Total execution time: {total_time:.2f} seconds")
                print(f"Overall RMSD: {rmsd:.6f} Å")
                print(f"Results saved in: {self.output_dir}/")
                
                self.results['execution_time'] = total_time
                return self.results
                
            except Exception as e:
                print(f"\nERROR: Analysis failed with exception: {e}")
                import traceback
                traceback.print_exc()
                return None


    def main():
        """
        Main function to run the analysis pipeline.
        """
        # Configuration
        PDB_ID = "6vsb"  # COVID-19 spike protein
        OUTPUT_DIR = "results"
        CONTACT_THRESHOLD = 7.0  # Angstroms
        MAX_LINKS_TEST = 1000  # Limit for computational efficiency
        
        # Create and run analysis pipeline
        pipeline = ProteinAnalysisPipeline(PDB_ID, OUTPUT_DIR)
        results = pipeline.run_complete_analysis(
            contact_threshold=CONTACT_THRESHOLD,
            max_links_test=MAX_LINKS_TEST
        )
        
        if results:
            print(f"\nAnalysis completed! Check the '{OUTPUT_DIR}' folder for all results.")
            print(f"Key result: RMSD = {results['rmsd']:.6f} Å")
        else:
            print("Analysis failed. Check error messages above.")


    if __name__ == "__main__":
        main()