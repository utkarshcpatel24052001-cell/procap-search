# ProCap Search: Methodology Documentation

## Overview

This document outlines the comprehensive methodology used in the ProCap Search project for benchmarking protein structure search tools designed to identify bacterial capsule biosynthesis proteins.

## 1. Data Collection Phase

### 1.1 Reference Dataset Compilation
- Curated 23+ known bacterial capsule biosynthesis proteins from literature and RCSB PDB
- Selected proteins from pathogenic bacteria: *Streptococcus pneumoniae*, *Pseudomonas aeruginosa*, *Salmonella enterica*, *Neisseria meningitidis*, *Escherichia coli*
- Ensured structural diversity (different protein folds and functions)

### 1.2 Protein Structure Acquisition
- Downloaded PDB files from RCSB using automated API calls
- Validated file integrity using checksum verification
- Generated manifest tracking download metadata

## 2. Feature Extraction Phase

### 2.1 Sequence-Based Features
- Protein sequence length
- Molecular weight (Da)
- Isoelectric point (pI)
- Aromaticity index
- Instability index
- Secondary structure composition (% helix, % sheet, % coil)

### 2.2 Structure-Based Features
- C-alpha coordinate extraction
- Distance matrix calculation
- B-factor analysis (temperature factors)
- Radius of gyration (Rg)
- Solvent-accessible surface area (SASA)

## 3. Benchmarking Phase

### 3.1 RMSD-Based Alignment (Physics-Based Method)

**Algorithm:**
1. Extract C-alpha coordinates from query and database proteins
2. Superimpose structures using optimal rotation matrix
3. Calculate Root Mean Square Deviation

**Formula:**
RMSD = sqrt((1/N) * Σ(di²)) where di = distance between atom i in two structures

**Similarity Score Normalization:**
Similarity = max(0, 1 - (RMSD / 10.0))

**Advantages:**
- Biophysically meaningful
- Considers 3D structure directly
- High accuracy for homologous proteins

**Disadvantages:**
- Requires 3D coordinates (slow)
- Sensitive to alignment parameters
- Limited to proteins with similar lengths

### 3.2 Sequence Similarity (BLAST-Like Method)

**Algorithm:**
1. Extract amino acid sequences from proteins
2. Perform global sequence alignment (Smith-Waterman)
3. Calculate sequence identity

**Identity Calculation:**
Identity = (matching residues) / max(seq1_length, seq2_length)

**Advantages:**
- Very fast (< 10 seconds per query)
- Works with sequences alone
- Highly optimized tools available

**Disadvantages:**
- Misses distant homologs
- Cannot detect structural homologs with low sequence similarity
- Sensitive to sequence divergence

### 3.3 Distance Matrix Alignment (Topology-Based Method)

**Algorithm:**
1. Calculate pairwise distance matrix for C-alpha atoms
2. Extract upper triangle (avoid redundancy)
3. Correlate distance matrices

**Correlation Calculation:**
Similarity = (correlation + 1) / 2 Range: [0, 1] where 1 = perfect correlation

**Advantages:**
- Robust to small structural variations
- Detects topologically similar proteins
- Independent of alignment

**Disadvantages:**
- Requires 3D structures
- Slower than sequence methods
- Can have false positives due to similar folds

## 4. Evaluation Metrics

### 4.1 Classification Metrics

**Sensitivity (Recall):**
Sensitivity = TP / (TP + FN) Measures: % of known homologs correctly identified Range: [0, 1] | Higher is better

**Specificity:**
Specificity = TN / (TN + FP) Measures: % of non-homologs correctly rejected Range: [0, 1] | Higher is better

**Precision:**
Precision = TP / (TP + FP) Measures: % of predictions that are correct Range: [0, 1] | Higher is better

**F1-Score:**
F1 = 2 × (Precision × Sensitivity) / (Precision + Sensitivity) Measures: Harmonic mean (balanced metric) Range: [0, 1] | Higher is better

## 5. Validation Strategy

### 5.1 Cross-Validation Approach
- 80% training set (known homologs used for threshold determination)
- 20% test set (held-out for unbiased evaluation)
- Stratified by organism to avoid bias

### 5.2 Negative Controls
- Included metabolic enzymes unrelated to capsule synthesis
- Ensured specificity testing against non-homologs
- Tested with proteins from different folds

## 6. Consensus Integration

### 6.1 Multi-Method Voting
Combines all three methods with weighted averaging: Final_Score = (RMSD_score × 0.33) + (Sequence_score × 0.33) + (DistMatrix_score × 0.34)

### 6.2 Confidence Ranking
- All three methods agree: **High confidence** ⭐⭐⭐
- Two methods agree: **Moderate confidence** ⭐⭐
- One method agrees: **Low confidence** ⭐

## 7. Reproducibility

### 7.1 Code Quality
- Modular, well-documented Python functions
- Comprehensive docstrings for all modules
- Version-controlled dependencies (requirements.txt)
- Automated testing and syntax validation

### 7.2 Data Management
- Reference dataset in CSV format (machine-readable)
- Automatic PDB manifest generation
- Timestamped results and logs
- Clear output directory structure

## 8. Limitations and Considerations

1. **Dataset Size:** Only 23 reference proteins; larger datasets improve reliability
2. **Structure Quality:** PDB files vary in resolution; may affect RMSD calculations
3. **Sequence Divergence:** Very distant homologs may not be detected by sequence methods
4. **Computational Cost:** RMSD-based methods require 3D structure superposition

## 9. Future Improvements

1. Integration with AlphaFold structure predictions
2. Machine learning-based embedding methods (ProtBERT, ESM)
3. GPU acceleration for RMSD calculations
4. Integration with external tools (FoldSeek, TM-align)
5. Expanded reference dataset with 100+ proteins

## 10. Conclusion

This methodology provides a rigorous, reproducible framework for benchmarking protein structure search tools. By comparing three distinct approaches and providing clear performance metrics, it enables informed decision-making for tool selection based on specific research requirements.

---

**Document Version:** 2.0  
**Last Updated:** March 27, 2026  
**Author:** Utkarsh Patel
