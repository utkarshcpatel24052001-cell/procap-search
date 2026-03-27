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
