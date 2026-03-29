---
title: ProCap Search Demo
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.55.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# ProCap Search — Protein Capsule Structure Search Pipeline

**Live App:** https://procap-search-6q7sgrsvhydckpi2hdpb7g.streamlit.app/

ProCap Search is a high-performance **protein structure search tool** for bacterial pathogen capsule identification. It benchmarks **3 independent structure-search methods** (RMSD superposition, sequence alignment, distance-matrix topology) and combines them into a **consensus scoring system** to prioritize candidate proteins for vaccine and therapeutic development.

---

## What this tool does

### 1) RMSD Structure Superposition
- Compares **alpha-carbon (CA) coordinates** between query and database proteins.
- Calculates **Root Mean Square Deviation** for structural similarity.
- Scores proteins by 3D backbone alignment quality.

### 2) Sequence Identity Alignment  
- Performs **global pairwise alignment** (Needleman-Wunsch) on amino acid sequences.
- Reports **% sequence identity** and alignment length.
- Fast screening for evolutionary conservation.

### 3) Distance Matrix Topology Search
- Extracts **pairwise distance matrices** from CA coordinates.
- Compares matrix correlation to detect **fold similarity** regardless of sequence.
- Identifies structurally similar proteins with different sequences.

### 4) Consensus Scoring
- **Integrates all 3 methods** with equal weighting.
- Confidence flags based on agreement across methods:
  - High (3/3): all methods agree
  - Medium (2/3): two methods agree  
  - Low (1/3): single method match

---

## How to interpret results (quick guide)

- **RMSD Score (0–1):** Higher = more similar structure. Formula: `1 - (RMSD / 10)`.
- **Sequence Score (0–1):** Direct % identity (0.70 = 70% identical residues).
- **Distance Matrix Score (0–1):** Correlation of pairwise distances. Higher = same fold topology.
- **Consensus Score:** Weighted average of all three methods.
  - \> 0.70: candidate match
  - 0.50–0.70: borderline  
  - < 0.50: weak similarity

---

## Reference dataset

- **23 bacterial capsule biosynthesis proteins** from 5 pathogenic species:
  - *Streptococcus pneumoniae* (5 proteins)
  - *Pseudomonas aeruginosa* (5 proteins)
  - *Salmonella enterica* (5 proteins)
  - *Neisseria meningitidis* (4 proteins)
  - *Escherichia coli* K-12 (4 proteins)

All structures resolved experimentally (PDB database).

---

## How to run locally

### Requirements
- Python 3.10+
- BioPython, SciPy, Pandas, NumPy

### Install & Run
```bash
pip install -r requirements.txt
streamlit run app.py
