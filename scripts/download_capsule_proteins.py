"""
Download Capsule Proteins
Fetch bacterial capsule biosynthesis proteins from RCSB PDB
"""

import argparse
import requests
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known capsule protein PDB IDs from literature
CAPSULE_PROTEINS = {
    'Streptococcus pneumoniae': ['1R3F', '1R3G', '1R3H', '1QFE', '1S24'],
    'Pseudomonas aeruginosa': ['2FJQ', '3FXN', '4NAY', '4NAZ', '4NB0'],
    'Salmonella enterica': ['3A1I', '3A1J', '3A1K', '1W0L', '1W0M'],
    'Neisseria meningitidis': ['2IK5', '2IK6', '2IK7', '1QW2'],
    'Escherichia coli': ['1G0K', '1G0L', '1G0M', '1H8G']
}

def download_pdb_file(pdb_id, output_dir):
    """Download PDB file from RCSB"""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = Path(output_dir) / f"{pdb_id}.pdb"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(output_path, 'w') as f:
            f.write(response.text)
        
        logger.info(f"Downloaded {pdb_id} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {pdb_id}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download capsule proteins from RCSB PDB')
    parser.add_argument('--output', default='data/pdb_structures', help='Output directory')
    parser.add_argument('--species', default='all', help='Specific species to download')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading capsule proteins to {output_dir}")
    
    # Download proteins
    downloaded = 0
    for species, pdb_ids in CAPSULE_PROTEINS.items():
        if args.species != 'all' and species.lower() not in args.species.lower():
            continue
        
        logger.info(f"\nDownloading {species} proteins...")
        for pdb_id in pdb_ids:
            if download_pdb_file(pdb_id, output_dir):
                downloaded += 1
    
    logger.info(f"\nDownload complete. Total files downloaded: {downloaded}")
    
    # Generate manifest
    manifest = {
        'total_downloaded': downloaded,
        'species': list(CAPSULE_PROTEINS.keys()),
        'output_directory': str(output_dir)
    }
    
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest saved to {manifest_path}")

if __name__ == '__main__':
    main()
