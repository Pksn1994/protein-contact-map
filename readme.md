# Protein Structure Analysis Pipeline

A Python pipeline that conducts protein structure analysis via graph Laplacian methods. The pipeline reconstructs 3D protein structures from contact networks and evaluates the significance of structural contacts within a given protein structure.

## Problem Statement

Traditional analysis of protein structures usually treats global properties, but it is essential to understand which single contacts are crucial for structural stability in drug discovery, protein engineering, and evolutionary studies. This pipeline recasts the following question: **Which contacts in a protein structure are most important to maintaining its 3D fold?**

The pipeline takes advantage of graph theory and spectral analysis to:
- Represent protein structures as contact graphs
- Infer 3D coordinates from connectivity information
- Identify key structural contacts between systematic removal analysis

## Features

- **PDB Processing Automation**: Download and extract coordinates from Protein Data Bank files
- **Contact Network Analysis**: Construct contact matrices with adjustable distance thresholds
- **Graph Laplacian Reconstruction**: Reconstruct 3D structures using spectral graph theory
- **Procrustes Alignment**: Best structural alignment for comparison
- **Link Importance Analysis**: Systematic examination of contact criticality
- **Comprehensive Visualization**: 3D plots, contact maps, and statistical analyses
- **Detailed Reporting**: Automatic generation of analysis reports

## Installation

### Requirements

- Python 3.7 or higher

#### If you use windows just run start.ps1 
#### If you use linux os just run start.sh
(all things added to this bash script and power shell script to automate lib install and start the applicatoin)
- Libraries required:
  ```python
  numpy >= 1.19.0
  scipy >= 1.5.0
  matplotlib >= 3.3.0
  pandas >= 1.1.0
  biopython >= 1.78
  tqdm >= 4.50.0
  ```


### Setup (failsafe manual method)

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd protein-structure-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Run the complete analysis pipeline for a protein (e.g., COVID-19 spike protein):

```python
from main import ProteinAnalysisPipeline

# Create pipeline
pipeline = ProteinAnalysisPipeline("6vsb", output_dir="results")

# Run full analysis
results = pipeline.run_complete_analysis(
    contact_threshold=7.0,   # Angstrom threshold for contacts
    max_links_test=1000      # Computational efficiency limit
)

print(f"Analysis done! RMSD: {results['rmsd']:.6f} Å")
```

### Command Line Usage

```bash
python main.py
```

This will process PDB ID "6vsb" (configurable in main.py) and output all results to the "results/" directory.

### Step-by-Step Usage

Each step of analysis can be run separately:

```python
# Step 1: Download and extract coordinates
from download_pdb import download_pdb_file
from extractor import extract_ca_coordinates

download_pdb_file("6vsb")
coordinates = extract_ca_coordinates("pdb6vsb.ent")

# Step 2: Calculate distance matrix
from distance_calculator import calculate_distance_matrix
distance_matrix = calculate_distance_matrix(coordinates)

# Step 3: Create contact matrix
from contact_matrix import create_contact_matrix
contact_matrix = create_contact_matrix(distance_matrix, threshold=7.0)

# Continue with remaining steps.
```

## Parameters

### Main Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_id` | str | Required | 4-character PDB identifier |
| `output_dir` | str | "results" | Directory for output files |
| `contact_threshold` | float | 7.0 | Threshold for contacts (Angstroms) |
| `max_links_test` | int | 1000 | Max contacts to test in link analysis |

### Contact Matrix Parameters

- **threshold**: Cutoff distance for contact definition (default: 7.0 Å)
  - Typical values: 6-8 Å for alpha-carbon contacts
  - Lower values = less dense networks, higher values = denser networks

### Link Analysis Parameters

- **max_links**: Computational limit for testing contacts
  - Recommended: 1000-5000 for proteins <500 residues
  - Higher values provide more complete analysis but add to the runtime

## Implementation Details

### Algorithms Used

1. **Contact Network Construction**: Euclidean distance thresholding
2. **Graph Laplacian**: L = D - A (degree matrix - adjacency matrix)
3. **Eigendecomposition**: Symmetric eigenvalue decomposition using LAPACK
4. **Coordinate Reconstruction**: Utilizing 2nd, 3rd, and 4th smallest eigenvectors
5. **Procrustes Alignment**: Optimal rotation, translation, and scaling
6. **RMSD Calculation**: Root Mean Square Deviation between structures

### Assumptions and Limitations

- **Input**: Requires PDB files with alpha-carbon atoms
- **Memory**: O(N²) memory complexity for N residues
- **Connectivity**: Assumes protein is connected graph
- **Resolution**: Limited by contact threshold selection
- **Runtime**: Link analysis increases as O(N×M) with M being number of contacts tested

### Performance Considerations

- **Small proteins** (<200 residues): Analysis in minutes
- **Medium proteins** (200-500 residues): 10-30 minutes with limited link testing
- **Large proteins** (>500 residues): Consider reducing `max_links_test` parameter

## Output Files

The pipeline generates extensive output in the provided directory:

### Coordinate Files
- `ca_coordinates_{pdb_id}.csv`: Original alpha-carbon coordinates
- `reconstructed_coords_{pdb_id}.csv`: Reconstructed coordinates
- `aligned_coords_{pdb_id}.csv`: Procrustes-aligned coordinates

### Matrix Files
- `distance_matrix_{pdb_id}.csv`: Pairwise distance matrix
- `contact_matrix_{pdb_id}.csv`: Binary contact matrix
- `laplacian_matrix_{pdb_id}.csv`: Graph Laplacian matrix

### Analysis Results
- `link_importance_{pdb_id}.csv`: Contact criticality rankings
- `alignment_{pdb_id}_transformation.txt`: Alignment parameters
- `analysis_report_{pdb_id}.txt`: Comprehensive analysis report

### Visualizations
- `contact_matrix_{pdb_id}.png`: Contact network visualization
- `eigenvalue_spectrum_{pdb_id}.png`: Eigenvalue distribution
- `coordinate_comparison_{pdb_id}.png`: Original vs reconstructed structures
- `alignment_comparison_{pdb_id}.png`: Alignment quality assessment
- `link_importance_{pdb_id}.png`: Contact importance analysis

## Examples

### Example 1: COVID-19 Spike Protein Analysis

```python
# Analyze COVID-19 spike protein structure
pipeline = ProteinAnalysisPipeline("6vsb")
results = pipeline.run_complete_analysis()
```

print(f"Structure has {len(results['coordinates'])} residues")
print(f"Reconstruction RMSD: {results['rmsd']:.3f} Å")
print(f"Most critical contact: residues {results['link_analysis'].iloc[0]['residue_i']}-{results['link_analysis'].iloc[0]['residue_j']}")

### Example 2: Custom Parameters

```python
# Analysis with custom parameters
pipeline = ProteinAnalysisPipeline("1ubq", output_dir="ubiquitin_analysis")
results = pipeline.run_complete_analysis(
    contact_threshold=6.5,     # Tighter contact definition
    max_links_test=2000        # More detailed link analysis
)
```

### Example 3: Individual Step Analysis

```python
# Load existing data and run only link analysis
from link_analysis import analyze_link_importance
```
import pandas as pd
import numpy as np

original_coords = pd.read_csv("results/ca_coordinates_6vsb.csv").values
contact_matrix = np.loadtxt("results/contact_matrix_6vsb.csv", delimiter=",", dtype=int)

link_results = analyze_link_importance(original_coords, contact_matrix, max_links=500)
print(f"Top critical contact: {link_results.iloc[0]['contact']}")


## References

- Graph Laplacian methods for protein structure analysis
- Procrustes analysis for structural alignment
- Contact map analysis in structural biology
- Protein Data Bank (PDB) file format specification
