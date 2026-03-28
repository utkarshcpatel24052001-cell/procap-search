"""
ProCap Search Professional Diagnostic Suite
Operational Intelligence for Bacterial Capsule Biosynthesis Protein Identification
Author: Utkarsh Patel | Version: 2.0 (Production-Grade)
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from Bio import pairwise2
from Bio.PDB import PDBParser as BioPDBParser
from Bio.SeqUtils import ProtParam
from scipy.spatial.distance import cdist

# ==================================================
# 1. PAGE CONFIG & PROFESSIONAL STYLING
# ==================================================
st.set_page_config(
    page_title="ProCap Search Professional | Diagnostic Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Background & Text Colors */
    body, .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    /* Global Typography - Clear & Readable */
    html, body, .stMarkdown, p, li, div, span, label, button {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        color: #1a1a1a !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #0d47a1 !important;
        font-weight: 700 !important;
    }
    
    /* Metric Styling - Clear Numbers */
    div[data-testid="stMetricLabel"] p {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #424242 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 800 !important;
        color: #0d47a1 !important;
    }
    
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 700 !important;
        color: #0d47a1 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Button Styling - High Contrast */
    .stButton > button {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 0.7em 1.5em !important;
        background-color: #0d47a1 !important;
        border: 2px solid #0d47a1 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #1565c0 !important;
        border-color: #1565c0 !important;
    }
    
    /* Input Fields */
    input, textarea, select {
        color: #1a1a1a !important;
        background-color: #f5f5f5 !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 6px !important;
        padding: 10px !important;
        font-size: 15px !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: #0d47a1 !important;
        background-color: #ffffff !important;
    }
    
    /* Alert/Info Boxes - High Visibility */
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
    
    .warning-box {
        background-color: #ffe0b2 !important;
        border-left: 6px solid #e65100 !important;
        padding: 16px !important;
        border-radius: 8px !important;
        color: #bf360c !important;
        margin: 12px 0 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f5f5f5 !important;
    }
    
    section[data-testid="stSidebar"] * {
        font-size: 15px !important;
        color: #1a1a1a !important;
    }
    
    /* Dataframe Styling */
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
    
    /* Divider */
    hr {
        border: 2px solid #0d47a1 !important;
    }
    
    /* Container Styling */
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
    
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 2. CORE UTILITIES & CONSTANTS
# ==================================================
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

DIAGNOSTIC_THRESHOLDS = {
    "High_Confidence": 0.75,
    "Medium_Confidence": 0.55,
    "Low_Confidence": 0.35,
}

class PDBWrapper:
    """Robust PDB file parser with error handling"""
    def __init__(self, pdb_path: str):
        self.pdb_path = Path(pdb_path)
        self.parser = BioPDBParser(QUIET=True)
        self.structure = None
        self.error_log = []

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
            self.error_log.append(f"Chain {chain_id} not found in structure")
            return None
        except Exception as e:
            self.error_log.append(f"Coordinate extraction error: {str(e)}")
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
            self.error_log.append(f"Sequence extraction error: {str(e)}")
            return None

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> Optional[float]:
    """Calculate RMSD between two coordinate sets"""
    if coords1 is None or coords2 is None or coords1.shape != coords2.shape:
        return None
    try:
        distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
        return float(np.sqrt(np.mean(distances ** 2)))
    except Exception:
        return None

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix"""
    return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """Compare distance matrices via Pearson correlation"""
    try:
        flat1 = dm1[np.triu_indices_from(dm1, k=1)]
        flat2 = dm2[np.triu_indices_from(dm2, k=1)]
        corr = np.corrcoef(flat1, flat2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0

def download_pdb_by_id(pdb_id: str, out_path: Path) -> bool:
    """Download PDB structure from RCSB"""
    try:
        pdb_id = pdb_id.strip().upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        out_path.write_text(r.text)
        return True
    except Exception as e:
        st.error(f"Failed to download {pdb_id}: {str(e)}")
        return False

def get_physchem_properties(seq: str) -> Dict[str, float]:
    """Calculate physicochemical properties of protein sequence"""
    try:
        clean_seq = seq.replace("X", "").replace("U", "")
        if len(clean_seq) < 2:
            return {"mw_kda": np.nan, "pi": np.nan, "instability": np.nan, "aromaticity": np.nan}
        params = ProtParam.ProteinAnalysis(clean_seq)
        return {
            "mw_kda": params.molecular_weight() / 1000,
            "pi": params.isoelectric_point(),
            "instability": params.instability_index(),
            "aromaticity": params.aromaticity(),
        }
    except Exception:
        return {"mw_kda": np.nan, "pi": np.nan, "instability": np.nan, "aromaticity": np.nan}

# ==================================================
# 3. DATA LOADING & DATABASE
# ==================================================
@st.cache_data
def load_reference_csv() -> pd.DataFrame:
    try:
        df = pd.read_csv("data/known_capsule_proteins.csv")
        return df
    except FileNotFoundError:
        st.warning("CSV file not found. Using empty reference dataset.")
        return pd.DataFrame(columns=["PDB_ID", "Organism", "Gene_Name", "Function", "Protein_Family"])

REFERENCE_DF = load_reference_csv()

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids: list) -> Tuple[list, list]:
    """Download PDB structures from RCSB"""
    db_dir.mkdir(parents=True, exist_ok=True)
    downloaded, failed = [], []
    
    progress_bar = st.progress(0)
    for i, pid in enumerate(pdb_ids):
        try:
            out = db_dir / f"{pid}.pdb"
            if not out.exists():
                if download_pdb_by_id(pid, out):
                    downloaded.append(pid)
                else:
                    failed.append(pid)
            else:
                downloaded.append(pid)
        except Exception:
            failed.append(pid)
        progress_bar.progress((i + 1) / len(pdb_ids))
    
    return downloaded, failed

# ==================================================
# 4. COMPREHENSIVE SEARCH ENGINE
# ==================================================
def run_diagnostic_pipeline(
    query_pdb: Path,
    db_dir: Path,
    threshold: float,
    chain_id: str
) -> Tuple[pd.DataFrame, Dict]:
    """Execute full diagnostic pipeline"""
    
    qp = PDBWrapper(str(query_pdb))
    if not qp.parse():
        return pd.DataFrame(), {"status": "error", "message": "Failed to parse query PDB"}
    
    qcoords = qp.get_coordinates(chain_id=chain_id)
    qseq = qp.get_sequence(chain_id=chain_id)
    
    if qcoords is None or not qseq:
        return pd.DataFrame(), {"status": "error", "message": f"No valid coordinates/sequence for chain {chain_id}"}
    
    qdm = calculate_distance_matrix(qcoords)
    physchem_query = get_physchem_properties(qseq)
    
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
        
        # RMSD Score
        rmsd_val, r_score = np.nan, 0.0
        if len(qcoords) == len(dcoords):
            rmsd_val = calculate_rmsd(qcoords, dcoords)
            if rmsd_val is not None:
                r_score = max(0.0, 1.0 - (rmsd_val / 10.0))
        
        # Sequence Identity Score
        s_score = 0.0
        aligned_length = 0
        alns = pairwise2.align.globalxx(qseq, dseq, one_alignment_only=True)
        if alns:
            a = alns[0]
            matches = sum(x == y for x, y in zip(a.seqA, a.seqB))
            s_score = matches / max(len(qseq), len(dseq))
            aligned_length = len(a.seqA)
        
        # Distance Matrix Correlation
        d_score = 0.0
        if qdm.shape == calculate_distance_matrix(dcoords).shape:
            corr = distance_matrix_correlation(qdm, calculate_distance_matrix(dcoords))
            d_score = (corr + 1.0) / 2.0
        
        # Consensus Score
        final_score = (0.33 * r_score) + (0.33 * s_score) + (0.34 * d_score)
        
        # Confidence Classification
        if final_score >= DIAGNOSTIC_THRESHOLDS["High_Confidence"]:
            confidence = "HIGH"
        elif final_score >= DIAGNOSTIC_THRESHOLDS["Medium_Confidence"]:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
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
            })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        ref_meta = REFERENCE_DF[["PDB_ID", "Organism", "Gene_Name", "Function"]].copy()
        ref_meta.columns = ["Target_PDB_Meta", "Organism", "Gene_Name", "Function"]
        ref_meta["Target_PDB_Meta"] = ref_meta["Target_PDB_Meta"].str.upper()
        
        df = df.merge(
            ref_meta,
            left_on="Target_PDB",
            right_on="Target_PDB_Meta",
            how="left"
        )
        df = df.drop("Target_PDB_Meta", axis=1)
        df = df.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    
    diagnostics = {
        "status": "success",
        "query_seq_length": len(qseq),
        "query_mw_kda": physchem_query.get("mw_kda", np.nan),
        "query_pi": physchem_query.get("pi", np.nan),
        "query_instability": physchem_query.get("instability", np.nan),
        "total_matches": len(results),
        "high_confidence_hits": len(df[df["Confidence"] == "HIGH"]) if not df.empty else 0,
    }
    
    return df, diagnostics

# ==================================================
# 5. REAL-WORLD DIAGNOSTIC SUGGESTIONS
# ==================================================
def generate_real_world_suggestions(results_df: pd.DataFrame, diagnostics: Dict, query_pdb: str) -> str:
    """Generate actionable recommendations"""
    
    suggestions = ""
    
    if diagnostics["status"] == "error":
        return f"⚠️ **Diagnostic Error:** {diagnostics['message']}"
    
    query_instability = diagnostics.get("query_instability", 0)
    high_conf_hits = diagnostics.get("high_confidence_hits", 0)
    total_matches = diagnostics.get("total_matches", 0)
    
    # Suggestion 1: Protein Stability
    if query_instability > 40:
        suggestions += "🔴 <b>CRITICAL - Protein Stability Issue:</b> Your query protein is predicted UNSTABLE in vitro (Instability Index > 40).<br><br>"
        suggestions += "<b>Recommendations:</b><br>"
        suggestions += "• Consider N-terminal stabilization tags (His6, MBP) for expression<br>"
        suggestions += "• Conduct CD spectroscopy to confirm secondary structure<br>"
        suggestions += "• Test buffer conditions (pH, salt, reducing agents) empirically<br><br>"
    elif query_instability > 30:
        suggestions += "🟡 <b>MODERATE - Stability Concern:</b> Instability Index 30-40 suggests potential aggregation risk.<br><br>"
    else:
        suggestions += "🟢 <b>EXCELLENT - Stability Profile:</b> Query protein appears suitable for standard expression systems.<br><br>"
    
    # Suggestion 2: Homolog Discovery
    if high_conf_hits == 0 and total_matches > 0:
        suggestions += "⚠️ <b>Limited High-Confidence Matches:</b> No proteins scored above 75% consensus.<br><br>"
        suggestions += "<b>Next Steps:</b><br>"
        suggestions += "• Expand database with orthologs from related species (use PSI-BLAST)<br>"
        suggestions += "• Consider relaxing sequence thresholds for paralogs<br>"
        suggestions += "• Perform AlphaFold2 modeling for functional insights<br><br>"
    elif high_conf_hits > 3:
        suggestions += f"✅ <b>EXCELLENT - Strong Match Profile:</b> Found {high_conf_hits} high-confidence homologs across species.<br><br>"
    
    # Suggestion 3: Vaccine Development
    if not results_df.empty:
        top_hit = results_df.iloc[0]
        organism = top_hit.get("Organism", "Unknown")
        
        suggestions += f"💊 <b>Development Pathway:</b> Top match identified in <i>{organism}</i>.<br><br>"
        suggestions += "<b>Vaccine Development Considerations:</b><br>"
        suggestions += "• Assess epitope conservation across matched homologs (MSA)<br>"
        suggestions += "• Prioritize B-cell and T-cell epitope prediction (NetMHC/BepiPred)<br>"
        suggestions += "• Consider cross-reactivity testing against non-target species<br><br>"
        suggestions += "<b>Structural Biology Next Steps:</b><br>"
        suggestions += "• Perform cryo-EM or X-ray crystallography if needed<br>"
        suggestions += "• Map protein-protein interactions using co-IP<br>"
        suggestions += "• Validate surface accessibility via hydroxyl radical footprinting<br><br>"
    
    # Suggestion 4: Experimental Validation
    suggestions += "🧪 <b>Recommended Experimental Validation:</b><br>"
    suggestions += "1. <b>Expression & Purification:</b> Test recombinant expression in E. coli/yeast/mammalian systems<br>"
    suggestions += "2. <b>Biophysical Characterization:</b> Perform SEC, TSA, and intrinsic fluorescence<br>"
    suggestions += "3. <b>Functional Assays:</b> Conduct ELISA, Western blot with species-specific antibodies<br>"
    suggestions += "4. <b>Clinical Relevance:</b> Correlate in vitro findings with in vivo protection in animal models<br><br>"
    
    return suggestions

# ==================================================
# 6. UI - MAIN APPLICATION
# ==================================================
st.title("🧬 ProCap Professional Diagnostic Suite")
st.markdown("### Operational Intelligence for Bacterial Capsule Biosynthesis Protein Identification")
st.divider()

# Setup Workspace
tmp_root = Path(tempfile.gettempdir()) / "procap_professional"
query_dir = tmp_root / "query"
db_dir = tmp_root / "db"
query_dir.mkdir(parents=True, exist_ok=True)
db_dir.mkdir(parents=True, exist_ok=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Diagnostic Configuration")
    
    with st.expander("Pipeline Settings", expanded=True):
        chain_id = st.text_input(
            "Target Chain ID",
            value="A",
            help="Most bacterial structures use chain A"
        )
        threshold = st.slider(
            "Global Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05,
            help="Lower = more sensitive. Higher = more stringent"
        )
    
    with st.expander("Reference Database", expanded=True):
        st.markdown(
            '<div class="info-box">🗄️ <b>Active Database:</b> 23 verified bacterial capsule proteins across 5 pathogenic species</div>',
            unsafe_allow_html=True
        )
        if st.button("🔄 Initialize/Update Database", use_container_width=True):
            with st.spinner("Downloading reference structures from RCSB PDB..."):
                pdb_list = REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist()
                d, f = build_database_from_reference_pdb_ids(db_dir, pdb_list)
                if f:
                    st.warning(f"⚠️ Failed to download {len(f)} structures")
                st.success(f"✅ Database ready: {len(d)} structures loaded")

# Main Tab Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Diagnostic Query",
    "📊 Database Analytics",
    "📖 Methodology",
    "📋 About"
])

with tab1:
    st.subheader("Protein Structure Query & Diagnostic Pipeline")
    
    col_query, col_results = st.columns([1, 1.8], gap="large")
    
    with col_query:
        st.markdown("### Input Configuration")
        query_mode = st.radio(
            "Query Source",
            ["RCSB PDB Database", "Local File Upload"],
            help="Select query input method"
        )
        
        if query_mode == "RCSB PDB Database":
            pdb_id = st.text_input(
                "Enter 4-Letter PDB ID",
                value="1R3F",
                placeholder="e.g., 1R3F, 2MJQ",
            ).strip().upper()
        else:
            uploaded_query = st.file_uploader("Upload .pdb File", type=["pdb"])
            pdb_id = None
        
        execute_btn = st.button(
            "▶ Execute Diagnostic Pipeline",
            type="primary",
            use_container_width=True,
            help="Run full RMSD + Sequence + Distance Matrix analysis"
        )
    
    with col_results:
        if execute_btn:
            query_pdb_path = None
            
            if query_mode == "RCSB PDB Database":
                if len(pdb_id) != 4:
                    st.markdown(
                        '<div class="alert-low">❌ Invalid PDB ID format. Must be 4 characters.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    query_pdb_path = query_dir / f"{pdb_id}.pdb"
                    if not query_pdb_path.exists():
                        with st.spinner(f"Downloading {pdb_id}..."):
                            if not download_pdb_by_id(pdb_id, query_pdb_path):
                                st.markdown(
                                    '<div class="alert-low">❌ Failed to download. Check PDB ID.</div>',
                                    unsafe_allow_html=True
                                )
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
            
            if query_pdb_path:
                with st.spinner("🔬 Executing diagnostic pipeline..."):
                    results_df, diagnostics = run_diagnostic_pipeline(
                        query_pdb_path, db_dir, threshold, chain_id
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
                            '<div class="warning-box">⚠️ No homologous structures found matching criteria.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        top_hit = results_df.iloc[0]
                        score = top_hit["Consensus_Score"]
                        confidence = top_hit["Confidence"]
                        
                        # Confidence Banner
                        if confidence == "HIGH":
                            st.markdown(
                                f'<div class="alert-high">✅ HIGH CONFIDENCE HOMOLOG DETECTED | Score: {score:.3f}/1.000</div>',
                                unsafe_allow_html=True
                            )
                        elif confidence == "MEDIUM":
                            st.markdown(
                                f'<div class="alert-medium">⚠️ MEDIUM CONFIDENCE MATCH | Score: {score:.3f}/1.000</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="alert-low">🔴 LOW CONFIDENCE | Score: {score:.3f}/1.000</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Top Hit Details
                        st.markdown("#### Top Biological Match")
                        col_top1, col_top2 = st.columns(2)
                        with col_top1:
                            st.write(f"**PDB ID:** {top_hit['Target_PDB']}")
                            st.write(f"**Organism:** {top_hit.get('Organism', 'N/A')}")
                            st.write(f"**Gene Name:** {top_hit.get('Gene_Name', 'N/A')}")
                        with col_top2:
                            st.write(f"**Function:** {top_hit.get('Function', 'N/A')}")
                            if not pd.isna(top_hit['RMSD_Å']):
                                st.write(f"**RMSD:** {top_hit['RMSD_Å']:.2f} Å")
                            st.write(f"**Query Coverage:** {top_hit.get('Query_Coverage_%', 0):.1f}%")
                        
                        # Query PhysChem Profile
                        st.markdown("#### Query Protein Profile")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Sequence Length", f"{diagnostics['query_seq_length']} AA")
                        m2.metric("Molecular Weight", f"{diagnostics['query_mw_kda']:.1f} kDa")
                        m3.metric("Isoelectric Point", f"{diagnostics['query_pi']:.2f}")
                        m4.metric("Instability Index", f"{diagnostics['query_instability']:.1f}")
                        
                        # Full Results Table
                        st.markdown("#### Complete Search Results (Top 15)")
                        display_cols = ["Target_PDB", "Consensus_Score", "Confidence", "Seq_Identity_%", "RMSD_Å", "Organism", "Gene_Name"]
                        st.dataframe(
                            results_df[display_cols].head(15),
                            use_container_width=True,
                            hide_index=True,
                        )
                        
                        # Download Results
                        st.markdown("#### Export Results")
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download CSV",
                                data=csv_data,
                                file_name=f"ProCap_Results_{pdb_id}.csv",
                                mime="text/csv"
                            )
                        with col_dl2:
                            json_data = results_df.to_json(orient="records", indent=2)
                            st.download_button(
                                "📥 Download JSON",
                                data=json_data,
                                file_name=f"ProCap_Results_{pdb_id}.json",
                                mime="application/json"
                            )
                        
                        # Real-World Suggestions
                        st.markdown("---")
                        st.markdown("#### 🎯 Real-World Diagnostic Recommendations")
                        suggestions = generate_real_world_suggestions(results_df, diagnostics, pdb_id or "UPLOAD")
                        st.markdown(f'<div class="info-box">{suggestions}</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Reference Database Analytics")
    
    if REFERENCE_DF.empty:
        st.warning("No reference data loaded.")
    else:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Reference Proteins", len(REFERENCE_DF))
        with col_s2:
            st.metric("Unique Organisms", REFERENCE_DF["Organism"].nunique())
        with col_s3:
            st.metric("Protein Families", REFERENCE_DF.get("Protein_Family", pd.Series()).nunique())
        
        col_ch1, col_ch2 = st.columns(2)
        
        with col_ch1:
            organism_counts = REFERENCE_DF["Organism"].value_counts()
            fig1 = px.bar(
                x=organism_counts.index,
                y=organism_counts.values,
                title="Protein Distribution by Pathogen",
                labels={"x": "Organism", "y": "Count"},
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

with tab3:
    st.subheader("Scientific Methodology & Algorithm Description")
    
    st.markdown("""
    ### Overview
    **ProCap Search** integrates three complementary computational methods:
    
    ---
    
    ### 1️⃣ RMSD-Based Structural Superposition
    
    **Principle:** Measures 3D backbone alignment quality
    
    - **Strength:** Direct geometric similarity; Physics-grounded
    - **Limitation:** Requires identical coordinate counts
    - **Formula:** `RMSD_Score = max(0, 1 - (RMSD / 10 Å))`
    
    ---
    
    ### 2️⃣ Sequence Identity Alignment
    
    **Principle:** Needleman-Wunsch global alignment
    
    - **Strength:** Fast; Detects evolutionary relationships
    - **Limitation:** Low sensitivity for remote homologs (< 30% identity)
    - **Formula:** `Seq_Identity = (Matching_Residues / Max_Sequence_Length)`
    
    ---
    
    ### 3️⃣ Distance Matrix Topology Correlation
    
    **Principle:** Pearson correlation of pairwise distances
    
    - **Strength:** Topology-based; Resilient to loop movements
    - **Limitation:** Requires matching coordinate count
    - **Formula:** `DM_Score = (Pearson_Correlation + 1) / 2`
    
    ---
    
    ### Consensus Scoring
    
    **Final Formula:**
    ```
    Consensus_Score = 0.33 × RMSD + 0.33 × Seq_Identity + 0.34 × DM
    ```
    
    **Confidence Classification:**
    - **HIGH (≥ 0.75):** All methods agree
    - **MEDIUM (0.55-0.75):** Two methods agree
    - **LOW (< 0.55):** Single method match
    """)

with tab4:
    st.subheader("About ProCap Professional")
    
    st.markdown("""
    ### Project Overview
    **ProCap Search** is a computational diagnostic suite for identifying bacterial capsule biosynthesis proteins.
    Designed for researchers in vaccine development, structural biology, and antimicrobial drug discovery.
    
    ### Key Features
    ✅ Multi-method consensus scoring
    ✅ Real-world diagnostic suggestions
    ✅ Physicochemical property profiling
    ✅ Professional-grade UI
    ✅ Export functionality (CSV, JSON)
    
    ### Developer
    - **Name:** Utkarsh Patel
    - **GitHub:** [procap-search](https://github.com/utkarshcpatel24052001-cell/procap-search)
    - **Institution:** Coventry University
    - **Specialization:** Computational Biology & Drug Discovery
    
    ### References
    - RCSB PDB: https://www.rcsb.org
    - BioPython: Cock et al. (2009)
    - SciPy: Virtanen et al. (2020)
    """)

st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: #424242;'>"
    "ProCap Professional v2.0 | Production-Grade Diagnostic Suite<br>"
    "<a href='https://github.com/utkarshcpatel24052001-cell/procap-search' target='_blank' style='color: #0d47a1; text-decoration: none;'>"
    "📍 GitHub Repository</a>"
    "</p>",
    unsafe_allow_html=True,
)
