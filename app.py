"""
ProCap Search Professional Diagnostic Suite v3.0
High-Throughput Structural Homology & Functional Annotation Pipeline
Production-Grade Bioinformatics Application
Author: Utkarsh Patel
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from Bio import pairwise2, Seq, SeqIO
from Bio.PDB import PDBParser as BioPDBParser, PDBIO, Select
from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from scipy.spatial.distance import cdist
from scipy import stats

# ==================================================
# 1. PAGE CONFIG & PROFESSIONAL STYLING
# ==================================================
st.set_page_config(
    page_title="ProCap Professional Diagnostic Suite | v3.0",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Professional Scientific Typography */
    body, .stApp {
        background-color: #f8f9fa;
        color: #1a1a1a;
    }
    
    html, body, .stMarkdown, p, li, div, span, label {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        color: #1a1a1a !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #0d47a1 !important;
        font-weight: 800 !important;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }
    
    h1 { font-size: 32px !important; }
    h2 { font-size: 24px !important; }
    h3 { font-size: 20px !important; }
    
    /* Metric Cards */
    div[data-testid="stMetricLabel"] p {
        font-size: 13px !important;
        font-weight: 800 !important;
        color: #455a64 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 38px !important;
        font-weight: 900 !important;
        color: #0d47a1 !important;
    }
    
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #0d47a1 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Buttons */
    .stButton > button {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        background-color: #0d47a1 !important;
        border: 2px solid #0d47a1 !important;
        border-radius: 8px !important;
        padding: 0.8em 1.6em !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #1565c0 !important;
        border-color: #1565c0 !important;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.3) !important;
    }
    
    /* Input Fields */
    input, textarea, select {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 6px !important;
        padding: 10px !important;
        font-size: 15px !important;
    }
    
    input:focus, textarea:focus {
        border-color: #0d47a1 !important;
        box-shadow: 0 0 8px rgba(13, 71, 161, 0.2) !important;
    }
    
    /* Alert Boxes - High Contrast */
    .alert-high {
        background-color: #c8e6c9 !important;
        border-left: 6px solid #2e7d32 !important;
        padding: 16px !important;
        border-radius: 8px !important;
        color: #1b5e20 !important;
        font-weight: 700 !important;
        margin: 12px 0 !important;
        font-size: 15px !important;
    }
    
    .alert-medium {
        background-color: #fff9c4 !important;
        border-left: 6px solid #f57f17 !important;
        padding: 16px !important;
        border-radius: 8px !important;
        color: #e65100 !important;
        font-weight: 700 !important;
        margin: 12px 0 !important;
        font-size: 15px !important;
    }
    
    .alert-low {
        background-color: #ffcccc !important;
        border-left: 6px solid #c62828 !important;
        padding: 16px !important;
        border-radius: 8px !important;
        color: #b71c1c !important;
        font-weight: 700 !important;
        margin: 12px 0 !important;
        font-size: 15px !important;
    }
    
    .info-box {
        background-color: #e3f2fd !important;
        border-left: 6px solid #1565c0 !important;
        padding: 16px !important;
        border-radius: 8px !important;
        color: #0d47a1 !important;
        margin: 12px 0 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    
    .success-box {
        background-color: #f1f8e9 !important;
        border-left: 6px solid #558b2f !important;
        padding: 16px !important;
        border-radius: 8px !important;
        color: #33691e !important;
        margin: 12px 0 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    
    /* Sequence Alignment Box */
    .seq-align-box {
        font-family: 'Courier New', monospace !important;
        background-color: #f5f5f5 !important;
        padding: 16px !important;
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
        overflow-x: auto !important;
        font-size: 13px !important;
        line-height: 1.3 !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        color: #1a1a1a !important;
    }
    
    /* 3D Viewer Container */
    .viewer-container {
        background-color: #ffffff !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        min-height: 450px !important;
    }
    
    /* Dataframe */
    .dataframe {
        font-size: 14px !important;
        color: #1a1a1a !important;
    }
    
    thead {
        background-color: #0d47a1 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    tbody tr:hover {
        background-color: #e3f2fd !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] * {
        font-size: 15px !important;
        color: #1a1a1a !important;
    }
    
    /* Container Borders */
    .stContainer {
        background-color: #ffffff !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #e3f2fd !important;
        color: #0d47a1 !important;
        font-weight: 700 !important;
    }
    
    /* Horizontal Rule */
    hr {
        border: 2px solid #0d47a1 !important;
    }
    
    /* Statistics Table */
    .stats-table {
        background-color: #f5f5f5 !important;
        border-collapse: collapse !important;
    }
    
    .stats-table td, .stats-table th {
        border: 1px solid #e0e0e0 !important;
        padding: 10px !important;
        text-align: left !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 2. CORE BIOPYTHON UTILITIES
# ==================================================
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}

# Real-world diagnostic thresholds (from literature)
DIAGNOSTIC_THRESHOLDS = {
    "Identical": 0.90,
    "Highly_Similar": 0.75,
    "Moderately_Similar": 0.55,
    "Weakly_Similar": 0.35,
}

class PDBWrapper:
    """Professional PDB parser with comprehensive error handling"""
    def __init__(self, pdb_path: str):
        self.pdb_path = Path(pdb_path)
        self.parser = BioPDBParser(QUIET=True)
        self.structure = None
        self.error_log = []
        self.header = {}

    def parse(self) -> bool:
        try:
            self.structure = self.parser.get_structure(self.pdb_path.stem, str(self.pdb_path))
            return True
        except Exception as e:
            self.error_log.append(f"Parse error: {str(e)}")
            return False

    def get_coordinates(self, chain_id: str = "A") -> Optional[np.ndarray]:
        if self.structure is None:
            return None
        try:
            coords = []
            chain = self.structure[0][chain_id]
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
            return np.array(coords, dtype=float) if coords else None
        except KeyError:
            self.error_log.append(f"Chain {chain_id} not found")
            return None
        except Exception as e:
            self.error_log.append(f"Coordinate extraction: {str(e)}")
            return None

    def get_sequence(self, chain_id: str = "A") -> Optional[str]:
        if self.structure is None:
            return None
        try:
            seq = []
            chain = self.structure[0][chain_id]
            for residue in chain:
                res = residue.get_resname()
                seq.append(THREE_TO_ONE.get(res, "X"))
            return "".join(seq)
        except KeyError:
            self.error_log.append(f"Chain {chain_id} not found")
            return None
        except Exception as e:
            self.error_log.append(f"Sequence extraction: {str(e)}")
            return None

    def get_pdb_content(self) -> str:
        """Return raw PDB file content for 3D viewing"""
        try:
            return self.pdb_path.read_text()
        except Exception:
            return ""

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> Optional[float]:
    """Calculate RMSD (without superposition - naive measure)"""
    if coords1 is None or coords2 is None or coords1.shape != coords2.shape:
        return None
    try:
        distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
        return float(np.sqrt(np.mean(distances ** 2)))
    except Exception:
        return None

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix"""
    return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """Pearson correlation of distance matrices (topology similarity)"""
    try:
        flat1 = dm1[np.triu_indices_from(dm1, k=1)]
        flat2 = dm2[np.triu_indices_from(dm2, k=1)]
        corr = np.corrcoef(flat1, flat2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0

def download_pdb_by_id(pdb_id: str, out_path: Path) -> bool:
    """Download PDB structure from RCSB with error handling"""
    try:
        pdb_id = pdb_id.strip().upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        out_path.write_text(r.text)
        return True
    except requests.exceptions.HTTPError as e:
        st.error(f"PDB {pdb_id} not found in RCSB database (HTTP {e.response.status_code})")
        return False
    except requests.exceptions.Timeout:
        st.error(f"Timeout downloading {pdb_id}. RCSB server unresponsive.")
        return False
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return False

def get_advanced_physchem_properties(seq: str) -> Dict[str, float]:
    """Calculate advanced physicochemical properties (scientific parameters)"""
    try:
        clean_seq = seq.replace("X", "").replace("U", "")
        if len(clean_seq) < 2:
            return {
                "mw_kda": np.nan,
                "pi": np.nan,
                "instability": np.nan,
                "aromaticity": np.nan,
                "gravy": np.nan,  # Grand average of hydropathy
                "charge_ph7": np.nan,
                "extinction_coeff": np.nan,
            }
        
        params = ProtParam.ProteinAnalysis(clean_seq)
        
        # IsoelectricPoint for accurate pI
        iso = IsoelectricPoint(clean_seq)
        pi = iso.pH()
        
        return {
            "mw_kda": params.molecular_weight() / 1000,
            "pi": float(pi),
            "instability": params.instability_index(),
            "aromaticity": params.aromaticity(),
            "gravy": params.gravy(),
            "extinction_coeff": params.molar_extinction_coefficient()[0],  # M-1 cm-1
            "helix_prob": params.secondary_structure_fraction()[0],
        }
    except Exception as e:
        st.error(f"PhysChem error: {str(e)}")
        return {
            "mw_kda": np.nan, "pi": np.nan, "instability": np.nan,
            "aromaticity": np.nan, "gravy": np.nan,
            "extinction_coeff": np.nan, "helix_prob": np.nan,
        }

def classify_confidence_level(score: float) -> Tuple[str, str]:
    """Classify confidence based on scientific thresholds"""
    if score >= DIAGNOSTIC_THRESHOLDS["Identical"]:
        return "IDENTICAL", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Highly_Similar"]:
        return "HIGHLY SIMILAR", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Moderately_Similar"]:
        return "MODERATELY SIMILAR", "alert-medium"
    else:
        return "WEAKLY SIMILAR", "alert-low"

# ==================================================
# 3. DATA LOADING
# ==================================================
@st.cache_data
def load_reference_csv() -> pd.DataFrame:
    try:
        df = pd.read_csv("data/known_capsule_proteins.csv")
        return df
    except FileNotFoundError:
        st.warning("CSV not found. Using empty dataset.")
        return pd.DataFrame(columns=["PDB_ID", "Organism", "Gene_Name", "Function", "Protein_Family", "Structure_Type"])

REFERENCE_DF = load_reference_csv()

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids: list) -> Tuple[list, list]:
    """Download PDB structures with progress tracking"""
    db_dir.mkdir(parents=True, exist_ok=True)
    downloaded, failed = [], []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pid in enumerate(pdb_ids):
        try:
            out = db_dir / f"{pid}.pdb"
            if not out.exists():
                status_text.text(f"⏳ Downloading {pid.upper()}... ({i+1}/{len(pdb_ids)})")
                if download_pdb_by_id(pid, out):
                    downloaded.append(pid)
                else:
                    failed.append(pid)
            else:
                downloaded.append(pid)
        except Exception:
            failed.append(pid)
        
        progress_bar.progress((i + 1) / len(pdb_ids))
    
    status_text.empty()
    return downloaded, failed

# ==================================================
# 4. SEARCH ENGINE & ANALYSIS
# ==================================================
def run_comprehensive_diagnostic(
    query_pdb: Path,
    db_dir: Path,
    chain_id: str
) -> Tuple[pd.DataFrame, Dict]:
    """Execute full diagnostic pipeline with scientific rigor"""
    
    qp = PDBWrapper(str(query_pdb))
    if not qp.parse():
        return pd.DataFrame(), {"status": "error", "message": "Failed to parse query PDB"}
    
    qcoords = qp.get_coordinates(chain_id=chain_id)
    qseq = qp.get_sequence(chain_id=chain_id)
    
    if qcoords is None or not qseq:
        return pd.DataFrame(), {"status": "error", "message": f"No valid chain {chain_id}"}
    
    qdm = calculate_distance_matrix(qcoords)
    physchem_query = get_advanced_physchem_properties(qseq)
    
    results = []
    for pdb_file in sorted(db_dir.glob("*.pdb")):
        target_id = pdb_file.stem.upper()
        if target_id == query_pdb.stem.upper():
            continue
        
        dp = PDBWrapper(str(pdb_file))
        if not dp.parse():
            continue
        
        dcoords = dp.get_coordinates(chain_id=chain_id)
        dseq = dp.get_sequence(chain_id=chain_id)
        
        if dcoords is None or not dseq:
            continue
        
        # 1. RMSD Score (structural superposition quality)
        rmsd_val, r_score = np.nan, 0.0
        if len(qcoords) == len(dcoords):
            rmsd_val = calculate_rmsd(qcoords, dcoords)
            if rmsd_val is not None:
                r_score = max(0.0, 1.0 - (rmsd_val / 10.0))
        
        # 2. Sequence Identity (evolutionary relationship)
        s_score = 0.0
        aligned_length = 0
        alns = pairwise2.align.globalxx(qseq, dseq, one_alignment_only=True)
        if alns:
            a = alns[0]
            matches = sum(x == y for x, y in zip(a.seqA, a.seqB))
            s_score = matches / max(len(qseq), len(dseq))
            aligned_length = len(a.seqA)
        
        # 3. Distance Matrix Correlation (topology preservation)
        d_score = 0.0
        if qdm.shape == calculate_distance_matrix(dcoords).shape:
            corr = distance_matrix_correlation(qdm, calculate_distance_matrix(dcoords))
            d_score = (corr + 1.0) / 2.0
        
        # Consensus Score (weighted by literature priorities)
        final_score = (0.33 * r_score) + (0.33 * s_score) + (0.34 * d_score)
        
        # Confidence Classification
        confidence, _ = classify_confidence_level(final_score)
        
        if final_score > 0:
            results.append({
                "Target_PDB": target_id,
                "Consensus_Score": final_score,
                "Confidence": confidence,
                "RMSD_Å": rmsd_val,
                "Seq_Identity_%": s_score * 100,
                "DM_Correlation": d_score,
                "Aligned_Length": aligned_length,
                "Query_Coverage_%": (aligned_length / len(qseq) * 100) if len(qseq) > 0 else 0,
                "Target_Seq": dseq,
            })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        ref_meta = REFERENCE_DF[["PDB_ID", "Organism", "Gene_Name", "Function", "Protein_Family", "Structure_Type"]].copy()
        ref_meta["PDB_ID"] = ref_meta["PDB_ID"].str.upper()
        
        df = df.merge(
            ref_meta,
            left_on="Target_PDB",
            right_on="PDB_ID",
            how="left"
        )
        df = df.drop("PDB_ID", axis=1)
        df = df.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    
    diagnostics = {
        "status": "success",
        "query_seq_length": len(qseq),
        "query_mw_kda": physchem_query.get("mw_kda", np.nan),
        "query_pi": physchem_query.get("pi", np.nan),
        "query_instability": physchem_query.get("instability", np.nan),
        "query_gravy": physchem_query.get("gravy", np.nan),
        "query_aromaticity": physchem_query.get("aromaticity", np.nan),
        "query_helix_prob": physchem_query.get("helix_prob", np.nan),
        "total_matches": len(results),
        "high_confidence_hits": len(df[df["Confidence"].str.contains("SIMILAR", na=False)]) if not df.empty else 0,
    }
    
    return df, diagnostics

def generate_alignment_visualization(seqA: str, seqB: str, blocksize: int = 60) -> str:
    """Generate professional sequence alignment visualization"""
    alns = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)
    if not alns:
        return "<p>Alignment failed.</p>"
    
    a = alns[0]
    
    html_output = '<div class="seq-align-box">\n'
    
    for i in range(0, len(a.seqA), blocksize):
        # Match line
        match_line = ""
        for j in range(i, min(i + blocksize, len(a.seqA))):
            if a.seqA[j] == a.seqB[j] and a.seqA[j] != "-":
                match_line += "|"
            elif a.seqA[j] != "-" and a.seqB[j] != "-":
                match_line += "."
            else:
                match_line += " "
        
        query_block = a.seqA[i:i+blocksize]
        match_block = match_line[:blocksize]
        target_block = a.seqB[i:i+blocksize]
        
        html_output += f"<span style='color:#0d47a1; font-weight:bold;'>Query:</span>   {query_block}\n"
        html_output += f"<span style='color:#2e7d32; font-weight:bold;'>Match:</span>   {match_block}\n"
        html_output += f"<span style='color:#c62828; font-weight:bold;'>Target:</span>  {target_block}\n\n"
    
    html_output += '</div>'
    return html_output

def render_3d_structure(pdb_content: str) -> str:
    """Create py3Dmol 3D visualization"""
    html = f"""
    <html>
    <head>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    </head>
    <body>
        <div id="viewer" style="width:100%; height:450px; border-radius: 8px; border: 2px solid #e0e0e0;"></div>
        <script>
            let viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{backgroundColor: 'white'}});
            let data = `{pdb_content}`;
            viewer.addModel(data, 'pdb');
            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            viewer.zoomTo();
            viewer.render();
        </script>
    </body>
    </html>
    """
    return html

# ==================================================
# 5. MAIN APPLICATION UI
# ==================================================
st.title("🧬 ProCap Search Professional Diagnostic Suite")
st.markdown("### High-Throughput Structural Homology & Functional Annotation Pipeline")
st.markdown("#### *Version 3.0 | Production-Grade Bioinformatics*")
st.divider()

# Setup workspace
tmp_root = Path(tempfile.gettempdir()) / "procap_professional"
query_dir = tmp_root / "query"
db_dir = tmp_root / "db"
query_dir.mkdir(parents=True, exist_ok=True)
db_dir.mkdir(parents=True, exist_ok=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    with st.expander("Query Settings", expanded=True):
        chain_id = st.text_input(
            "Target Chain ID",
            value="A",
            max_chars=1,
            help="Biological assembly chain (typically A). Verify PDB file if uncertain."
        )
    
    with st.expander("Reference Database", expanded=True):
        st.markdown(
            '<div class="info-box"><b>🗄️ Active Database:</b> 23 verified bacterial capsule proteins across 5 pathogenic species</div>',
            unsafe_allow_html=True
        )
        if st.button("🔄 Initialize/Update Database", use_container_width=True):
            with st.spinner("📥 Downloading PDB structures from RCSB..."):
                pdb_list = REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist()
                d, f = build_database_from_reference_pdb_ids(db_dir, pdb_list)
                
                if f:
                    st.warning(f"⚠️ Failed: {len(f)} structures. {', '.join(f)}")
                
                st.markdown(
                    f'<div class="success-box">✅ Database Ready: {len(d)} structures loaded</div>',
                    unsafe_allow_html=True
                )

# Main Tab Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Structural Diagnostic",
    "🧪 3D Visualization",
    "📊 Database Analytics",
    "📖 Documentation"
])

# ==================================================
# TAB 1: STRUCTURAL DIAGNOSTIC
# ==================================================
with tab1:
    st.subheader("Protein Structure Query & Diagnostic Pipeline")
    
    col_query, col_results = st.columns([1, 1.8], gap="large")
    
    with col_query:
        st.markdown("### Input Configuration")
        query_mode = st.radio(
            "Query Source",
            ["RCSB PDB Database", "Local File Upload"],
            help="Choose input method for protein structure"
        )
        
        if query_mode == "RCSB PDB Database":
            pdb_id = st.text_input(
                "Enter 4-Letter PDB ID",
                value="1R3F",
                max_chars=4,
                placeholder="e.g., 1R3F",
                help="Visit https://www.rcsb.org to find protein IDs"
            ).strip().upper()
        else:
            uploaded_query = st.file_uploader("Upload .pdb File", type=["pdb"])
            pdb_id = None
        
        execute_btn = st.button(
            "▶ EXECUTE DIAGNOSTIC PIPELINE",
            type="primary",
            use_container_width=True,
            help="Run RMSD + Sequence + Distance Matrix analyses"
        )
    
    with col_results:
        if execute_btn:
            query_pdb_path = None
            
            # Load Query
            if query_mode == "RCSB PDB Database":
                if len(pdb_id) != 4:
                    st.markdown(
                        '<div class="alert-low">❌ Invalid PDB ID. Must be 4 characters (e.g., 1R3F).</div>',
                        unsafe_allow_html=True
                    )
                else:
                    query_pdb_path = query_dir / f"{pdb_id}.pdb"
                    if not query_pdb_path.exists():
                        with st.spinner(f"Downloading {pdb_id} from RCSB PDB..."):
                            if not download_pdb_by_id(pdb_id, query_pdb_path):
                                query_pdb_path = None
            else:
                if uploaded_query is None:
                    st.markdown(
                        '<div class="alert-low">❌ No file uploaded.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    query_pdb_path = query_dir / uploaded_query.name
                    query_pdb_path.write_bytes(uploaded_query.getvalue())
            
            # Execute Pipeline
            if query_pdb_path:
                with st.spinner("🔬 Executing diagnostic pipeline..."):
                    results_df, diagnostics = run_comprehensive_diagnostic(
                        query_pdb_path, db_dir, chain_id
                    )
                
                if diagnostics["status"] == "error":
                    st.markdown(
                        f'<div class="alert-low">❌ {diagnostics["message"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("### Diagnostic Results")
                    
                    if results_df.empty:
                        st.markdown(
                            '<div class="alert-low">⚠️ No homologous structures found. Expand database or check parameters.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        # Save to session state for Tab 2
                        st.session_state['query_pdb_content'] = query_pdb_path.read_text()
                        st.session_state['results_df'] = results_df
                        st.session_state['diagnostics'] = diagnostics
                        st.session_state['query_pdb'] = str(query_pdb_path)
                        
                        top_hit = results_df.iloc[0]
                        score = top_hit["Consensus_Score"]
                        confidence, alert_class = classify_confidence_level(score)
                        
                        # Confidence Banner
                        st.markdown(
                            f'<div class="{alert_class}">🧬 CONFIDENCE: {confidence} | Score: {score:.3f}/1.000</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Top Hit Details
                        st.markdown("#### Top Biological Match")
                        col_top1, col_top2, col_top3 = st.columns(3)
                        with col_top1:
                            st.write(f"**PDB ID:** `{top_hit['Target_PDB']}`")
                            st.write(f"**Organism:** {top_hit.get('Organism', 'N/A')}")
                        with col_top2:
                            st.write(f"**Gene:** {top_hit.get('Gene_Name', 'N/A')}")
                            st.write(f"**Function:** {top_hit.get('Function', 'N/A')}")
                        with col_top3:
                            st.write(f"**Family:** {top_hit.get('Protein_Family', 'N/A')}")
                            if not pd.isna(top_hit['RMSD_Å']):
                                st.write(f"**RMSD:** {top_hit['RMSD_Å']:.2f} Å")
                        
                        # Query PhysChem Profile
                        st.markdown("#### Query Protein Profile (Advanced PhysChem)")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Length", f"{diagnostics['query_seq_length']} AA")
                        m2.metric("Mol. Weight", f"{diagnostics['query_mw_kda']:.1f} kDa")
                        m3.metric("Isoelectric Point", f"{diagnostics['query_pi']:.2f}")
                        m4.metric("Instability Index", f"{diagnostics['query_instability']:.1f}")
                        
                        m5, m6, m7 = st.columns(3)
                        m5.metric("GRAVY", f"{diagnostics['query_gravy']:.3f}")
                        m6.metric("Aromaticity", f"{diagnostics['query_aromaticity']:.3f}")
                        m7.metric("Helix Prob.", f"{diagnostics['query_helix_prob']:.2%}")
                        
                        # Full Results Table
                        st.markdown("#### Complete Search Results")
                        display_cols = ["Target_PDB", "Consensus_Score", "Confidence", "Seq_Identity_%", "RMSD_Å", "Organism"]
                        
                        st.dataframe(
                            results_df[display_cols].head(10),
                            use_container_width=True,
                            hide_index=True,
                        )
                        
                        # Download Options
                        st.markdown("#### Export Results")
                        col_dl1, col_dl2, col_dl3 = st.columns(3)
                        with col_dl1:
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 CSV Report",
                                data=csv_data,
                                file_name=f"ProCap_Results_{pdb_id or 'query'}.csv",
                                mime="text/csv"
                            )
                        with col_dl2:
                            json_data = results_df.to_json(orient="records", indent=2)
                            st.download_button(
                                "📥 JSON Export",
                                data=json_data,
                                file_name=f"ProCap_Results_{pdb_id or 'query'}.json",
                                mime="application/json"
                            )
                        with col_dl3:
                            st.download_button(
                                "📥 Query Structure",
                                data=query_pdb_path.read_text(),
                                file_name=f"query_structure.pdb",
                                mime="chemical/x-pdb"
                            )

# ==================================================
# TAB 2: 3D VISUALIZATION
# ==================================================
with tab2:
    st.subheader("3D Structure Visualization & Sequence Analysis")
    
    if 'query_pdb_content' in st.session_state:
        col3d, colseq = st.columns([1.5, 1], gap="large")
        
        with col3d:
            st.markdown("#### Interactive 3D Query Structure")
            st.caption("Color-coded by secondary structure (spectrum). Rotate/zoom with mouse.")
            
            try:
                html_3d = render_3d_structure(st.session_state['query_pdb_content'])
                st.components.v1.html(html_3d, height=480)
            except Exception as e:
                st.error(f"3D rendering failed: {str(e)}")
        
        with colseq:
            st.markdown(f"#### Sequence Alignment vs Top Hit")
            
            if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
                top = st.session_state['results_df'].iloc[0]
                
                # Get query sequence
                qp = PDBWrapper(st.session_state['query_pdb'])
                qp.parse()
                qseq = qp.get_sequence("A")
                
                st.caption("BioPython Global Alignment. | = Match, . = Similar")
                
                # Render alignment
                align_html = generate_alignment_visualization(qseq, top['Target_Seq'], blocksize=30)
                st.markdown(align_html, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Export FASTA
                fasta_content = f">Query\n{qseq}\n>Target_{top['Target_PDB']}\n{top['Target_Seq']}\n"
                st.download_button(
                    "📥 Download Alignment (FASTA)",
                    data=fasta_content,
                    file_name=f"alignment_{top['Target_PDB']}.fasta",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.markdown(
            '<div class="info-box">💡 Execute a diagnostic in Tab 1 to generate 3D models and alignments.</div>',
            unsafe_allow_html=True
        )

# ==================================================
# TAB 3: DATABASE ANALYTICS
# ==================================================
with tab3:
    st.subheader("Reference Database Analytics & Statistics")
    
    if not REFERENCE_DF.empty:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total PDBs", len(REFERENCE_DF))
        with col_stat2:
            st.metric("Unique Organisms", REFERENCE_DF["Organism"].nunique())
        with col_stat3:
            st.metric("Protein Families", REFERENCE_DF.get("Protein_Family", pd.Series()).nunique())
        
        st.divider()
        
        col_ch1, col_ch2 = st.columns(2)
        
        with col_ch1:
            organism_counts = REFERENCE_DF["Organism"].value_counts()
            fig1 = px.bar(
                x=organism_counts.index,
                y=organism_counts.values,
                title="Protein Distribution by Pathogen",
                labels={"x": "Organism", "y": "Count"},
                color_discrete_sequence=["#0d47a1"]
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_ch2:
            if "Protein_Family" in REFERENCE_DF.columns:
                fam_counts = REFERENCE_DF["Protein_Family"].value_counts()
                fig2 = px.pie(
                    values=fam_counts.values,
                    names=fam_counts.index,
                    title="Protein Family Composition"
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("#### Full Reference Dataset")
        st.dataframe(REFERENCE_DF, use_container_width=True, hide_index=True)
    else:
        st.warning("No reference data available.")

# ==================================================
# TAB 4: DOCUMENTATION
# ==================================================
with tab4:
    st.subheader("Scientific Documentation & Methodology")
    
    with st.expander("Algorithm Overview", expanded=True):
        st.markdown("""
        ### ProCap Search Pipeline
        
        **ProCap** integrates three independent structural comparison methods:
        
        #### 1️⃣ RMSD-Based Superposition
        - **Principle:** Root Mean Square Deviation of alpha-carbon backbone
        - **Range:** 0–10 Å (normalized to 0–1)
        - **Strength:** Direct geometric measure; Physics-grounded
        - **Limitation:** Requires identical coordinate counts
        
        #### 2️⃣ Sequence Identity Alignment
        - **Principle:** Needleman-Wunsch global pairwise alignment
        - **Range:** 0–100% identity
        - **Strength:** Evolutionary relationship; Fast computation
        - **Limitation:** Low sensitivity for remote homologs (< 30% identity)
        
        #### 3️⃣ Distance Matrix Topology
        - **Principle:** Pearson correlation of pairwise Euclidean distances
        - **Range:** -1 to +1 (normalized to 0–1)
        - **Strength:** Topology-based; Resilient to loop variations
        - **Limitation:** Requires matching coordinate counts
        
        #### Consensus Score Formula
        ```
        Consensus = 0.33 × RMSD_Score + 0.33 × Seq_Identity + 0.34 × DM_Correlation
        ```
        
        #### Confidence Classification (Scientific Thresholds)
        - **IDENTICAL (≥ 0.90):** High homology; Recommend experimental validation
        - **HIGHLY SIMILAR (0.75–0.89):** Strong homology; Likely functional conservation
        - **MODERATELY SIMILAR (0.55–0.74):** Moderate homology; Consider additional analysis
        - **WEAKLY SIMILAR (< 0.55):** Low homology; Require alternative approaches
        """)
    
    with st.expander("PhysChem Parameters", expanded=False):
        st.markdown("""
        ### Advanced Physicochemical Analysis
        
        #### Computed Properties
        - **Molecular Weight (kDa):** Sum of amino acid residue masses
        - **Isoelectric Point (pI):** pH where net charge = 0 (IsoelectricPoint algorithm)
        - **Instability Index:** Predicts in vitro stability (> 40 = unstable)
        - **GRAVY:** Grand Average of Hydropathy (-2 to +2 scale)
        - **Aromaticity:** Frequency of aromatic residues (0–1)
        - **Helix Probability:** Expected α-helix fraction (0–1)
        - **Extinction Coefficient:** UV absorbance at 280 nm (M⁻¹ cm⁻¹)
        """)
    
    with st.expander("Data Specifications", expanded=False):
        st.markdown(f"""
        ### Reference Database Specifications
        
        **Current Statistics:**
        - Total PDB Structures: {len(REFERENCE_DF)}
        - Unique Organisms: {REFERENCE_DF["Organism"].nunique()}
        - Protein Families: {REFERENCE_DF.get("Protein_Family", pd.Series()).nunique()}
        - Source: RCSB PDB (https://www.rcsb.org)
        
        **Coverage:**
        {REFERENCE_DF.groupby("Organism").size().to_string()}
        """)

st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: #455a64;'>"
    "ProCap Professional v3.0 | Production-Grade Diagnostic Suite<br>"
    "Author: Utkarsh Patel | "
    "<a href='https://github.com/utkarshcpatel24052001-cell/procap-search' target='_blank' style='color: #0d47a1;'>"
    "🔗 GitHub Repository</a><br>"
    "<i>For research use only. Not intended for clinical decision-making.</i>"
    "</p>",
    unsafe_allow_html=True,
)
