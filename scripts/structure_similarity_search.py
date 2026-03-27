"""
Structure Similarity Search
Implement multiple structure search methods
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from Bio import pairwise2
from scripts.utils import PDBParser, calculate_rmsd, calculate_distance_matrix, distance_matrix_correlation, setup_logger

logger = setup_logger(__name__)

class StructureSimilaritySearch:
    """Compare protein structures using multiple methods"""
    
    def __init__(self, query_pdb, database_dir, threshold=0.7):
        self.query_pdb = query_pdb
        self.database_dir = Path(database_dir)
        self.threshold = threshold
        self.results = []
    
    def rmsd_search(self):
        """RMSD-based alignment search"""
        logger.info("Running RMSD-based search...")
        
        query_parser = PDBParser(self.query_pdb)
        query_structure = query_parser.parse()
        query_coords = query_parser.get_coordinates()
        
        if query_coords is None:
            logger.error("Could not extract query coordinates")
            return []
        
        matches = []
        for pdb_file in self.database_dir.glob('*.pdb'):
            db_parser = PDBParser(str(pdb_file))
            db_structure = db_parser.parse()
            db_coords = db_parser.get_coordinates()
            
            if db_coords is None:
                continue
            
            # Simple RMSD (without alignment)
            if len(query_coords) == len(db_coords):
                rmsd = calculate_rmsd(query_coords, db_coords)
                
                # Normalize RMSD to similarity score (0-1)
                # Higher RMSD = lower similarity
                similarity = max(0, 1 - (rmsd / 10.0))
                
                if similarity >= self.threshold:
                    matches.append({
                        'method': 'RMSD',
                        'target': pdb_file.stem,
                        'score': similarity,
                        'rmsd': rmsd
                    })
        
        return sorted(matches, key=lambda x: x['score'], reverse=True)
    
    def sequence_search(self):
        """Sequence-based similarity search"""
        logger.info("Running sequence-based search...")
        
        query_parser = PDBParser(self.query_pdb)
        query_seq = query_parser.get_sequence()
        
        if not query_seq:
            logger.error("Could not extract query sequence")
            return []
        
        matches = []
        for pdb_file in self.database_dir.glob('*.pdb'):
            db_parser = PDBParser(str(pdb_file))
            db_seq = db_parser.get_sequence()
            
            if not db_seq:
                continue
            
            # Simple identity calculation
            alignment = pairwise2.align.globalxx(query_seq, db_seq)
            
            if alignment:
                best = alignment[0]
                matches_count = sum(a == b for a, b in zip(best[0], best[1]))
                identity = matches_count / max(len(query_seq), len(db_seq))
                
                if identity >= self.threshold:
                    matches.append({
                        'method': 'Sequence',
                        'target': pdb_file.stem,
                        'score': identity,
                        'aligned_length': len(best[0])
                    })
        
        return sorted(matches, key=lambda x: x['score'], reverse=True)
    
    def distance_matrix_search(self):
        """Distance matrix-based topology search"""
        logger.info("Running distance matrix search...")
        
        query_parser = PDBParser(self.query_pdb)
        query_coords = query_parser.get_coordinates()
        
        if query_coords is None:
            logger.error("Could not extract query coordinates")
            return []
        
        query_dm = calculate_distance_matrix(query_coords)
        matches = []
        
        for pdb_file in self.database_dir.glob('*.pdb'):
            db_parser = PDBParser(str(pdb_file))
            db_coords = db_parser.get_coordinates()
            
            if db_coords is None:
                continue
            
            db_dm = calculate_distance_matrix(db_coords)
            
            # Resize to same dimensions if needed
            if query_dm.shape != db_dm.shape:
                continue
            
            correlation = distance_matrix_correlation(query_dm, db_dm)
            similarity = (correlation + 1) / 2  # Normalize to 0-1
            
            if similarity >= self.threshold:
                matches.append({
                    'method': 'Distance_Matrix',
                    'target': pdb_file.stem,
                    'score': similarity,
                    'correlation': correlation
                })
        
        return sorted(matches, key=lambda x: x['score'], reverse=True)
    
    def run_all(self):
        """Run all three search methods"""
        rmsd_results = self.rmsd_search()
        seq_results = self.sequence_search()
        dm_results = self.distance_matrix_search()
        
        return {
            'rmsd': rmsd_results,
            'sequence': seq_results,
            'distance_matrix': dm_results
        }

def main():
    parser = argparse.ArgumentParser(description='Run protein structure similarity searches')
    parser.add_argument('--query', required=True, help='Query PDB file')
    parser.add_argument('--database', default='data/pdb_structures', help='Database directory')
    parser.add_argument('--method', default='all', choices=['rmsd', 'sequence', 'distance_matrix', 'all'])
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    parser.add_argument('--output', default='data/search_results', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    search = StructureSimilaritySearch(args.query, args.database, args.threshold)
    
    if args.method == 'all':
        results = search.run_all()
        
        for method, matches in results.items():
            logger.info(f"\n{method.upper()} Results:")
            for match in matches[:5]:
                logger.info(f"  {match['target']}: {match['score']:.3f}")
            
            # Save to CSV
            if matches:
                df = pd.DataFrame(matches)
                output_file = output_dir / f"{method}_results.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved to {output_file}")
    else:
        if args.method == 'rmsd':
            results = search.rmsd_search()
        elif args.method == 'sequence':
            results = search.sequence_search()
        else:
            results = search.distance_matrix_search()
        
        if results:
            df = pd.DataFrame(results)
            output_file = output_dir / f"{args.method}_results.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} results to {output_file}")

if __name__ == '__main__':
    main()
