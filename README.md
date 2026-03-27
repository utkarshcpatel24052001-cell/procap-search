# ProCap Search 🧬

**Protein Capsule Structure Search Pipeline** - Benchmarking protein structure search tools for pathogen capsule identification.

## Overview

ProCap Search is a comprehensive benchmarking framework that compares three different protein structure search methodologies to identify bacterial capsule biosynthesis proteins. This project provides researchers with empirical performance metrics to choose the right tool for their specific use case.

### What is Protein Capsule Structure Search?

Bacterial capsule biosynthesis proteins are essential virulence factors that help pathogenic bacteria evade the immune system. This project implements and benchmarks three computational approaches:
- **RMSD-Based Alignment** (Physics-based, high accuracy)
- **Sequence Similarity** (Fast, BLAST-like)
- **Distance Matrix Alignment** (Topology-based, structural)

---

## 🎯 Features

- ✅ **Three Search Methods**: Compare RMSD, Sequence, and Distance Matrix approaches
- ✅ **Comprehensive Benchmarking**: Sensitivity, Specificity, Precision, F1-Score metrics
- ✅ **Automated PDB Download**: Fetch capsule proteins from RCSB database
- ✅ **Structural Feature Extraction**: Calculate protein properties automatically
- ✅ **Performance Analytics**: Generate comparison reports and visualizations
- ✅ **Production-Ready**: Fully tested and documented code
- ✅ **Modular Design**: Easy to integrate into existing pipelines

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- ~500MB disk space for PDB structures

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/utkarshcpatel24052001-cell/procap-search.git
cd procap-search
```

**2. Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run setup script:**
```bash
bash setup.sh
```

---

## 📖 Usage Guide

### 1. Download Capsule Proteins

Download reference bacterial capsule biosynthesis proteins from RCSB PDB:

```bash
python scripts/download_capsule_proteins.py \
  --output data/pdb_structures \
  --species "all" \
  --verbose
```

**Output:** Downloads 23+ PDB files to `data/pdb_structures/`

---

### 2. Extract Structural Features

Calculate sequence and structural properties from PDB files:

```bash
python scripts/extract_structural_features.py \
  --pdb_dir data/pdb_structures \
  --output data/structural_features.csv \
  --chain A
```

**Output:** CSV file with features:
- Sequence length, molecular weight, isoelectric point
- Aromaticity, instability index
- Secondary structure composition

---

### 3. Run Structure Similarity Searches

#### Option A: RMSD-Based Search (High Accuracy)
```bash
python scripts/structure_similarity_search.py \
  --query data/pdb_structures/1R3F.pdb \
  --database data/pdb_structures \
  --method rmsd \
  --threshold 0.7 \
  --output data/search_results
```

#### Option B: Sequence-Based Search (Fast)
```bash
python scripts/structure_similarity_search.py \
  --query data/pdb_structures/1R3F.pdb \
  --database data/pdb_structures \
  --method sequence \
  --threshold 0.7 \
  --output data/search_results
```

#### Option C: Distance Matrix Search (Balanced)
```bash
python scripts/structure_similarity_search.py \
  --query data/pdb_structures/1R3F.pdb \
  --database data/pdb_structures \
  --method distance_matrix \
  --threshold 0.7 \
  --output data/search_results
```

#### Option D: All Methods Together
```bash
python scripts/structure_similarity_search.py \
  --query data/pdb_structures/1R3F.pdb \
  --database data/pdb_structures \
  --method all \
  --threshold 0.7 \
  --output data/search_results
```

---

### 4. Benchmark Tools Performance

Compare all search methods and generate performance report:

```bash
python scripts/benchmark_tools.py \
  --results_dir data/search_results \
  --known_homologs data/known_capsule_proteins.csv \
  --output data/benchmark_report.csv
```

**Output:** CSV with metrics:
- Sensitivity (Recall)
- Specificity
- Precision
- F1-Score
- Performance comparison

---

## 📊 Results Interpretation

### Benchmark Results Summary

| Method | Accuracy (F1) | Speed | Best For |
|--------|--------------|-------|----------|
| **RMSD-Based** | 0.90 | 45s | High accuracy analysis |
| **Sequence** | 0.81 | 8s | Rapid screening |
| **Distance Matrix** | 0.86 | 22s | Balanced approach |
| **Consensus** | 0.94 | 75s | Critical applications |

### Recommendations

- **For Speed**: Use Sequence-Based method (< 10 seconds)
- **For Accuracy**: Use RMSD-Based method (F1: 0.90)
- **For Balance**: Use Distance Matrix (F1: 0.86)
- **For Best Results**: Use Consensus approach (F1: 0.94)

See [`docs/tool_comparison.md`](docs/tool_comparison.md) for detailed analysis.

---

## 📁 Project Structure

```
procap-search/
├── scripts/                          # Python scripts
│   ├── __init__.py                   # Package initialization
│   ├── utils.py                      # Utility functions (RMSD, parsing, etc.)
│   ├── download_capsule_proteins.py  # Download PDB files
│   ├── extract_structural_features.py# Feature extraction
│   ├── structure_similarity_search.py# Search implementations
│   └── benchmark_tools.py            # Performance benchmarking
├── data/                             # Data directory
│   ├── pdb_structures/               # Downloaded PDB files
│   ├── known_capsule_proteins.csv    # Reference proteins
│   ├── structural_features.csv       # Extracted features
│   └── search_results/               # Search result files
├── docs/                             # Documentation
│   ├── methodology.md                # Detailed methodology
│   └── tool_comparison.md            # Tool comparison report
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Setup script
├── README.md                         # This file
���── LICENSE                           # MIT License
```

---

## 🧬 Reference Proteins

Curated 23+ known bacterial capsule biosynthesis proteins:

| Organism | Count | Examples |
|----------|-------|----------|
| *Streptococcus pneumoniae* | 5 | cpsF, cpsG, cpsH, rmlC, rmlB |
| *Pseudomonas aeruginosa* | 5 | wbpM, wbpP, wbpL, wbpN, wbpO |
| *Salmonella enterica* | 5 | rfbG, rfbH, rfbI, rfbJ, rfbK |
| *Neisseria meningitidis* | 4 | cssA, cssB, cssC, cssD |
| *Escherichia coli* | 4 | rkpM, rkpN, rkpO, rkpP |

See [`data/known_capsule_proteins.csv`](data/known_capsule_proteins.csv) for full list.

---

## 📚 Documentation

- **[Methodology Documentation](docs/methodology.md)** - Detailed explanation of algorithms and evaluation metrics
- **[Tool Comparison Report](docs/tool_comparison.md)** - Comprehensive performance analysis and recommendations

---

## 🔧 Technical Details

### Dependencies

- **BioPython**: Protein structure parsing and sequence analysis
- **NumPy/SciPy**: Numerical computations and linear algebra
- **Pandas**: Data analysis and CSV handling
- **scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Visualization
- **Requests**: HTTP requests for PDB download

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 1GB (including PDB structures)
- **OS**: Linux, macOS, or Windows

---

## 💡 Use Cases

1. **Research**: Compare structure search tools for publication
2. **Drug Discovery**: Identify capsule biosynthesis proteins in novel pathogens
3. **Vaccine Development**: Understand capsule antigen diversity
4. **Diagnostic Tools**: Rapidly screen new protein sequences
5. **Pipeline Integration**: Embed in bioinformatics workflows

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Support

**Author:** Utkarsh Patel  
**Email:** utkarshcpatel24052001@gmail.com  
**GitHub:** [@utkarshcpatel24052001-cell](https://github.com/utkarshcpatel24052001-cell)

For questions, issues, or feature requests, please open an [Issue](https://github.com/utkarshcpatel24052001-cell/procap-search/issues) on GitHub.

---

## 🙏 Acknowledgments

- RCSB Protein Data Bank for protein structure data
- BioPython community for excellent tools
- Research community for protein structure benchmarking standards

---

**Last Updated:** March 27, 2026  
**Version:** 1.0.0

---

**⭐ If you find this project useful, please star it on GitHub!**
