"""
Extract Structural Features
Extract sequence and structural features from PDB files
"""

import argparse
import pandas as pd
from pathlib import Path
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scripts.utils import PDBParser, setup_logger

logger = setup_logger(__name__)

def extract_features_from_pdb(pdb_file, chain_id='A'):
    """Extract features from a single PDB file"""
    parser = PDBParser(str(pdb_file))
    structure = parser.parse()
    
    if not structure:
        return None
    
    sequence = parser.get_sequence(chain_id)
    coords = parser.get_coordinates(chain_id)
    
    if not sequence or coords is None:
        return None
    
    # Calculate protein properties
    analyzer = ProteinAnalysis(sequence)
    
    features = {
        'PDB_ID': pdb_file.stem.upper(),
        'Sequence_Length': len(sequence),
        'Molecular_Weight': analyzer.molecular_weight(),
        'Isoelectric_Point': analyzer.isoelectric_point(),
        'Aromaticity': analyzer.aromaticity(),
        'Instability_Index': analyzer.instability_index(),
        'Helix_Fraction': analyzer.secondary_structure_fraction()[0],
        'Turn_Fraction': analyzer.secondary_structure_fraction()[1],
        'Sheet_Fraction': analyzer.secondary_structure_fraction()[2],
        'Sequence': sequence
    }
    
    return features

def main():
    parser = argparse.ArgumentParser(description='Extract structural features from PDB files')
    parser.add_argument('--pdb_dir', default='data/pdb_structures', help='Input PDB directory')
    parser.add_argument('--output', default='data/structural_features.csv', help='Output CSV file')
    parser.add_argument('--chain', default='A', help='Chain ID to extract')
    
    args = parser.parse_args()
    
    pdb_dir = Path(args.pdb_dir)
    pdb_files = list(pdb_dir.glob('*.pdb'))
    
    logger.info(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
    
    features_list = []
    for pdb_file in pdb_files:
        features = extract_features_from_pdb(pdb_file, args.chain)
        if features:
            features_list.append(features)
            logger.info(f"Extracted features from {pdb_file.name}")
    
    # Save to CSV
    df = pd.DataFrame(features_list)
    df.to_csv(args.output, index=False)
    
    logger.info(f"Saved {len(df)} features to {args.output}")

if __name__ == '__main__':
    main()
