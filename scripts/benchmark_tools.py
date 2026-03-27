"""
Benchmark Tools
Compare and benchmark protein structure search tool performance
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import logging
from scripts.utils import setup_logger

logger = setup_logger(__name__)

class ToolBenchmark:
    """Benchmark and compare search tool performance"""
    
    def __init__(self, results_dir, known_homologs_file):
        self.results_dir = Path(results_dir)
        self.known_homologs = pd.read_csv(known_homologs_file)
        self.metrics = {}
    
    def calculate_metrics(self, method_name, results_file):
        """Calculate sensitivity, specificity, precision, F1-score"""
        try:
            results = pd.read_csv(results_file)
        except Exception as e:
            logger.error(f"Could not read {results_file}: {str(e)}")
            return None
        
        # Extract true positives from results
        predictions = set(results['target'].values)
        true_positives = set(self.known_homologs['PDB_ID'].values)
        
        # Calculate metrics
        tp = len(predictions & true_positives)
        fp = len(predictions - true_positives)
        fn = len(true_positives - predictions)
        tn = max(0, len(true_positives) - tp - fn)
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        return {
            'Tool': method_name,
            'True_Positives': tp,
            'False_Positives': fp,
            'False_Negatives': fn,
            'True_Negatives': tn,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'F1_Score': f1
        }
    
    def benchmark_all(self):
        """Benchmark all available method results"""
        methods = ['rmsd', 'sequence', 'distance_matrix']
        
        for method in methods:
            results_file = self.results_dir / f"{method}_results.csv"
            
            if results_file.exists():
                metrics = self.calculate_metrics(method, results_file)
                if metrics:
                    self.metrics[method] = metrics
                    logger.info(f"\n{method.upper()} Metrics:")
                    for key, value in metrics.items():
                        if key != 'Tool':
                            logger.info(f"  {key}: {value:.4f}")
        
        return self.metrics
    
    def generate_report(self, output_file):
        """Generate comprehensive benchmark report"""
        if not self.metrics:
            logger.error("No metrics calculated. Run benchmark_all() first.")
            return
        
        df = pd.DataFrame(list(self.metrics.values()))
        df.to_csv(output_file, index=False)
        
        logger.info(f"\nBenchmark report saved to {output_file}")
        logger.info("\nSummary:")
        logger.info(df.to_string())

def main():
    parser = argparse.ArgumentParser(description='Benchmark protein structure search tools')
    parser.add_argument('--results_dir', default='data/search_results', help='Results directory')
    parser.add_argument('--known_homologs', default='data/known_capsule_proteins.csv', help='Known homologs file')
    parser.add_argument('--output', default='data/benchmark_report.csv', help='Output report file')
    
    args = parser.parse_args()
    
    benchmark = ToolBenchmark(args.results_dir, args.known_homologs)
    benchmark.benchmark_all()
    benchmark.generate_report(args.output)

if __name__ == '__main__':
    main()
