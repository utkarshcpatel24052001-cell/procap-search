"""
ProCap Search - Utility Functions
Helper functions for PDB parsing, RMSD calculation, and logging
"""

import logging
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser as BioPDBParser
from Bio.PDB.Polypeptide import three_to_one
from scipy.spatial.distance import cdist

# Configure logging
def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger for the project"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger(__name__)

# PDB Parsing Functions
class PDBParser:
    """Wrapper for Bio.PDB parsing with error handling"""
    
    def __init__(self, pdb_file):
        """Initialize PDB parser"""
        self.pdb_file = Path(pdb_file)
        self.parser = BioPDBParser(QUIET=True)
        self.structure = None
    
    def parse(self):
        """Parse PDB file and return structure"""
        try:
            self.structure = self.parser.get_structure(
                self.pdb_file.stem,
                str(self.pdb_file)
            )
            logger.info(f"Successfully parsed {self.pdb_file.name}")
            return self.structure
        except Exception as e:
            logger.error(f"Error parsing {self.pdb_file.name}: {str(e)}")
            return None
    
    def get_coordinates(self, chain_id='A'):
        """Extract C-alpha coordinates from chain"""
        if not self.structure:
            logger.error("Structure not parsed. Call parse() first.")
            return None
        
        coords = []
        model = self.structure[0]
        
        try:
            chain = model[chain_id]
            for residue in chain:
                if 'CA' in residue:
                    coord = residue['CA'].get_coord()
                    coords.append(coord)
            
            return np.array(coords)
        except Exception as e:
            logger.error(f"Error extracting coordinates: {str(e)}")
            return None
    
    def get_sequence(self, chain_id='A'):
        """Extract amino acid sequence from chain"""
        if not self.structure:
            return None
        
        sequence = []
        model = self.structure[0]
        
        try:
            chain = model[chain_id]
            for residue in chain:
                residue_name = residue.get_resname()
                if residue_name in three_to_one:
                    sequence.append(three_to_one[residue_name])
                else:
                    sequence.append('X')
            
            return ''.join(sequence)
        except Exception as e:
            logger.error(f"Error extracting sequence: {str(e)}")
            return None

# RMSD Calculation
def calculate_rmsd(coords1, coords2):
    """Calculate Root Mean Square Deviation between two structures"""
    if coords1.shape != coords2.shape:
        logger.error("Coordinate arrays have different shapes")
        return None
    
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
    rmsd = np.sqrt(np.mean(distances ** 2))
    
    return rmsd

# Similarity Metrics
def calculate_distance_matrix(coords):
    """Calculate pairwise distance matrix"""
    distances = cdist(coords, coords, metric='euclidean')
    return distances

def distance_matrix_correlation(dm1, dm2):
    """Calculate correlation between two distance matrices"""
    flat1 = dm1[np.triu_indices_from(dm1, k=1)]
    flat2 = dm2[np.triu_indices_from(dm2, k=1)]
    
    correlation = np.corrcoef(flat1, flat2)[0, 1]
    
    return correlation if not np.isnan(correlation) else 0.0

logger.info("Utils module loaded successfully")
