"""
ProCap Search Professional Diagnostic Suite v3.0
High-Throughput Structural Homology & Functional Annotation Pipeline
Production-Grade Bioinformatics Application (Dark Mode Edition)
Author: Utkarsh Patel
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
from Bio import pairwise2
from Bio.PDB import PDBParser as BioPDBParser
from Bio.SeqUtils import ProtParam
from scipy.spatial.distance import cdist

# ==================================================
# 1. PAGE CONFIG & DARK MODE STYLING
# ==================================================
st.set_page_config(
    page_title="ProCap Professional | v3.0",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Dark Background and White Text */
    html, body, [class*="css"], .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    html, body, .stMarkdown, p, li, span, label {
        font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif !important;
        font-size: 16px !important;
        color: #ffffff !important;
    }
    
    /* Headers in Red */
    h1, h2, h3, h4 { color: #ff3333 !important; font-weight: 700 !important; }
    h3 { font-size: 20px !important; border-bottom: 1px solid #333333; padding-bottom: 8px;}

    /* Metric Cards */
    div[data-testid="stMetricLabel"] p {
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #cccccc !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #ff3333 !important;
    }

    /* Status Banners for Dark Mode */
    .alert-high { background-color: #002200; border-left: 6px solid #00ff00; padding: 16px; border-radius: 6px; color: #00ff00; font-weight: 600; margin-bottom: 15px;}
    .alert-medium { background-color: #331a00; border-left: 6px solid #ffa500; padding: 16px; border-radius: 6px; color: #ffa500; font-weight: 600; margin-bottom: 15px;}
    .alert-low { background-color: #330000; border-left: 6px solid #ff0000; padding: 16px; border-radius: 6px; color: #ff3333; font-weight: 600; margin-bottom: 15px;}
    .info-box { background-color: #001133; border-left: 6px solid #3399ff; padding: 16px; border-radius: 6px; color: #3399ff; font-weight: 600; margin-bottom: 15px;}
    .success-box { background-color: #001a00; border-left: 6px solid #00ff00; padding: 16px; border-radius: 6px; color: #00ff00; font-weight: 600; margin-bottom: 15px;}
    .method-box { background-color: #1a1a2e; border-left: 6px solid #16c784; padding: 16px; border-radius: 6px; color: #16c784; font-weight: 600; margin-bottom: 15px; font-family: 'Courier New', monospace;}

    /* Sequence Alignment Box */
    .seq-align-box {
        font-family: 'Fira Code', 'Courier New', monospace !important;
        background-color: #111111;
        padding: 16px;
        border-radius: 6px;
        border: 1px solid #333333;
        overflow-x: auto;
        font-size: 14px !important;
        line-height: 1.4 !important;
        white-space: pre !important;
        color: #ffffff;
    }
    
    /* Buttons */
    .stButton > button { 
        font-weight: 600 !important; 
        border-radius: 6px !important; 
        background-color: #222222 !important;
        color: #ffffff !important;
        border: 1px solid #ff3333 !important;
    }
    .stButton > button:hover {
        background-color: #ff3333 !important;
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 2. CORE BIOPYTHON & MATH UTILITIES - ALL THREE METHODS
# ==================================================
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

DIAGNOSTIC_THRESHOLDS = {"Identical": 0.90, "Highly_Similar": 0.75, "Moderately_Similar": 0.55}

class PDBWrapper:
    """Professional PDB parser with multi-chain support"""
    def __init__(self, pdb_path: str):
        self.pdb_path = Path(pdb_path)
        self.parser = BioPDBParser(QUIET=True)
        self.structure = None
        self.header = {}

    def parse(self) -> bool:
        try:
            self.structure = self.parser.get_structure(self.pdb_path.stem, str(self.pdb_path))
            self.header = self.parser.get_header()
            return True
        except Exception as e:
            st.error(f"Parse error: {str(e)}")
            return False

    def get_available_chains(self) -> list:
        """Get all available chains in the PDB structure"""
        if not self.structure:
            return []
        try:
            chains = [chain.id for chain in self.structure[0]]
            return sorted(chains)
        except Exception:
            return []

    def get_coordinates(self, chain_id="A") -> Optional[np.ndarray]:
        """Extract alpha-carbon coordinates for a specific chain"""
        if not self.structure:
            return None
        try:
            coords = []
            chain = self.structure[0][chain_id]
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
            return np.array(coords, dtype=float) if coords else None
        except (KeyError, Exception) as e:
            return None

    def get_sequence(self, chain_id="A") -> Optional[str]:
        """Extract amino acid sequence for a specific chain"""
        if not self.structure:
            return None
        try:
            seq = []
            chain = self.structure[0][chain_id]
            for residue in chain:
                res_name = residue.get_resname()
                seq.append(THREE_TO_ONE.get(res_name, "X"))
            return "".join(seq) if seq else None
        except (KeyError, Exception):
            return None
        
    def get_experimental_metadata(self, chain_id="A"):
        """Extract crystallographic parameters"""
        resolution = self.header.get("resolution", np.nan)
        b_factors = []
        try:
            if self.structure:
                chain = self.structure[0][chain_id]
                for residue in chain:
                    if "CA" in residue:
                        b_factors.append(residue["CA"].get_bfactor())
        except (KeyError, Exception):
            pass
        return {
            "resolution": resolution,
            "mean_b_factor": np.mean(b_factors) if b_factors else np.nan
        }

# ==================================================
# 3. THREE COMPLEMENTARY SCORING METHODS
# ==================================================

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> Optional[float]:
    """METHOD 1: RMSD-Based Structural Superposition"""
    if coords1 is None or coords2 is None or coords1.shape != coords2.shape:
        return None
    try:
        distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
        return float(np.sqrt(np.mean(distances ** 2)))
    except Exception:
        return None

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Helper: Compute pairwise distance matrix"""
    return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """METHOD 3: Distance Matrix Topology Correlation"""
    try:
        flat1 = dm1[np.triu_indices_from(dm1, k=1)]
        flat2 = dm2[np.triu_indices_from(dm2, k=1)]
        corr = np.corrcoef(flat1, flat2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0

def calculate_sequence_identity(seqA: str, seqB: str) -> Tuple[float, int]:
    """METHOD 2: Sequence Identity Alignment (Needleman-Wunsch)"""
    try:
        # Fix: Ensure sequences are strings
        seqA = str(seqA).strip()
        seqB = str(seqB).strip()
        
        if not seqA or not seqB:
            return 0.0, 0
        
        alns = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)
        if not alns:
            return 0.0, 0
        a = alns[0]
        matches = sum(x == y for x, y in zip(a.seqA, a.seqB))
        s_score = matches / max(len(seqA), len(seqB))
        aligned_length = len(a.seqA)
        return float(s_score), aligned_length
    except Exception as e:
        st.warning(f"Sequence alignment error: {str(e)}")
        return 0.0, 0

def download_pdb_by_id(pdb_id: str, out_path: Path) -> bool:
    """Download PDB structure from RCSB with error handling"""
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id.strip().upper()}.pdb", timeout=15)
        r.raise_for_status()
        out_path.write_text(r.text)
        return True
    except Exception as e:
        st.error(f"Failed to download {pdb_id}: {str(e)}")
        return False

def get_advanced_physchem_properties(seq: str) -> Dict[str, float]:
    """Calculate advanced biochemical properties"""
    clean_seq = seq.replace("X", "").replace("U", "")
    default_props = {
        "mw_kda": np.nan, "pi": np.nan, "instability": np.nan,
        "aromaticity": np.nan, "gravy": np.nan, "extinction_coeff": np.nan
    }
    if len(clean_seq) < 2:
        return default_props
    try:
        params = ProtParam.ProteinAnalysis(clean_seq)
        pi = np.nan
        try:
            from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
            pi = IsoelectricPoint(clean_seq).pH()
        except ImportError:
            pi = params.isoelectric_point()

        return {
            "mw_kda": params.molecular_weight() / 1000,
            "pi": float(pi),
            "instability": params.instability_index(),
            "aromaticity": params.aromaticity(),
            "gravy": params.gravy(),
            "extinction_coeff": params.molar_extinction_coefficient()[0],
        }
    except Exception:
        return default_props

def classify_confidence_level(score: float) -> Tuple[str, str]:
    """Classify match quality based on consensus score"""
    if score >= DIAGNOSTIC_THRESHOLDS["Identical"]:
        return "IDENTICAL MATCH", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Highly_Similar"]:
        return "HIGHLY SIMILAR", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Moderately_Similar"]:
        return "MODERATELY SIMILAR", "alert-medium"
    else:
        return "WEAKLY SIMILAR", "alert-low"

# ==================================================
# 4. DATA LOADING & DATABASE PIPELINE
# ==================================================
@st.cache_data
def load_reference_csv() -> pd.DataFrame:
    try:
        return pd.read_csv("data/known_capsule_proteins.csv")
    except FileNotFoundError:
        st.warning("CSV not found. Using empty reference dataset.")
        return pd.DataFrame(columns=[
            "PDB_ID", "Organism", "Gene_Name", "Function",
            "Protein_Family", "Structure_Type"
        ])

REFERENCE_DF = load_reference_csv()

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids: list):
    """Download PDB structures from RCSB"""
    db_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pid in enumerate(pdb_ids):
        out = db_dir / f"{pid}.pdb"
        if out.exists():
            downloaded.append(pid)
        else:
            status_text.text(f"⏳ Downloading {pid}... ({i+1}/{len(pdb_ids)})")
            if download_pdb_by_id(pid, out):
                downloaded.append(pid)
        progress_bar.progress((i + 1) / len(pdb_ids))
    
    status_text.empty()
    return downloaded

def run_comprehensive_diagnostic(
    query_pdb: Path, db_dir: Path, chain_id: str
) -> Tuple[pd.DataFrame, Dict]:
    """Execute comprehensive 3-method diagnostic pipeline"""
    
    # Parse query structure
    qp = PDBWrapper(str(query_pdb))
    if not qp.parse():
        return pd.DataFrame(), {
            "status": "error",
            "message": "Failed to parse query PDB"
        }
    
    # Get query data
    qcoords = qp.get_coordinates(chain_id)
    qseq = qp.get_sequence(chain_id)
    
    if qcoords is None or not qseq:
        available_chains = qp.get_available_chains()
        return pd.DataFrame(), {
            "status": "error",
            "message": f"No valid chain '{chain_id}'. Available chains: {', '.join(available_chains) if available_chains else 'NONE'}"
        }
    
    qdm = calculate_distance_matrix(qcoords)
    physchem_query = get_advanced_physchem_properties(qseq)
    
    results = []
    
    # Compare against all database structures
    for pdb_file in sorted(db_dir.glob("*.pdb")):
        target_id = pdb_file.stem.upper()
        if target_id == query_pdb.stem.upper():
            continue
        
        dp = PDBWrapper(str(pdb_file))
        if not dp.parse():
            continue
        
        dcoords = dp.get_coordinates(chain_id)
        dseq = dp.get_sequence(chain_id)
        
        if dcoords is None or not dseq:
            continue
        
        # ===== METHOD 1: RMSD-Based Superposition =====
        rmsd_val, r_score = np.nan, 0.0
        if len(qcoords) == len(dcoords):
            rmsd_val = calculate_rmsd(qcoords, dcoords)
            if rmsd_val is not None:
                r_score = max(0.0, 1.0 - (rmsd_val / 10.0))
        
        # ===== METHOD 2: Sequence Identity (Needleman-Wunsch) =====
        s_score, aligned_length = calculate_sequence_identity(qseq, dseq)
        
        # ===== METHOD 3: Distance Matrix Topology =====
        d_score = 0.0
        if qdm.shape == calculate_distance_matrix(dcoords).shape:
            d_score = (
                distance_matrix_correlation(qdm, calculate_distance_matrix(dcoords)) + 1.0
            ) / 2.0
        
        # Consensus Score (weighted average of all three methods)
        final_score = (0.33 * r_score) + (0.33 * s_score) + (0.34 * d_score)
        confidence, _ = classify_confidence_level(final_score)
        
        if final_score > 0:
            results.append({
                "Target_PDB": target_id,
                "Consensus_Score": final_score,
                "Confidence": confidence,
                "RMSD_Å": rmsd_val,
                "RMSD_Score": r_score,
                "Seq_Identity_%": s_score * 100,
                "Seq_Score": s_score,
                "DM_Correlation": d_score,
                "Query_Coverage_%": (aligned_length / len(qseq) * 100) if len(qseq) > 0 else 0,
                "Target_Seq": dseq,
            })
    
    df = pd.DataFrame(results)
    if not df.empty:
        ref_meta = REFERENCE_DF.copy()
        ref_meta["PDB_ID"] = ref_meta["PDB_ID"].str.upper()
        df = df.merge(ref_meta, left_on="Target_PDB", right_on="PDB_ID", how="inner")
        df = df.drop("PDB_ID", axis=1, errors="ignore")
        df = df.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    
    diagnostics = {
        "status": "success",
        "query_seq_length": len(qseq),
        "total_matches": len(results),
        **physchem_query
    }
    return df, diagnostics

def predict_biological_function(results_df: pd.DataFrame) -> dict:
    """Predict function from homolog consensus"""
    top_3 = results_df.head(3).dropna(
        subset=['Protein_Family', 'Structure_Type', 'Function']
    )
    if top_3.empty:
        return {
            "family": "Unknown", "structure": "Unknown",
            "function": "Unknown", "confidence": "Low"
        }
    return {
        "family": top_3['Protein_Family'].mode()[0],
        "structure": top_3['Structure_Type'].mode()[0],
        "function": top_3['Function'].mode()[0],
        "confidence": "High" if top_3['Protein_Family'].nunique() == 1 else "Moderate"
    }

# ==================================================
# 5. VISUALIZATION ENGINES (Dark Mode)
# ==================================================
def generate_alignment_visualization(seqA: str, seqB: str, blocksize: int = 60) -> str:
    """Generate sequence alignment with all three methods info"""
    try:
        # Ensure sequences are proper strings
        seqA = str(seqA).strip()
        seqB = str(seqB).strip()
        
        if not seqA or not seqB:
            return "<p style='color:#ff3333;'>Alignment error: Empty sequence</p>"
        
        alns = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)
        if not alns:
            return "<p style='color:#ff3333;'>No alignment found</p>"
        a = alns[0]
        
        html = '<div class="seq-align-box">\n'
        for i in range(0, len(a.seqA), blocksize):
            match_line = "".join([
                "|" if a.seqA[j] == a.seqB[j] and a.seqA[j] != "-" else
                "." if a.seqA[j] != "-" and a.seqB[j] != "-" else " "
                for j in range(i, min(i + blocksize, len(a.seqA)))
            ])
            html += f"<span style='color:#66b3ff; font-weight:bold;'>Query: </span> {a.seqA[i:i+blocksize]}\n"
            html += f"<span style='color:#00ff00; font-weight:bold;'>Match: </span> {match_line}\n"
            html += f"<span style='color:#ff3333; font-weight:bold;'>Target:</span> {a.seqB[i:i+blocksize]}\n\n"
        html += '</div>'
        return html
    except Exception as e:
        return f"<p style='color:#ff3333;'>Alignment Error: {str(e)}</p>"

def render_3d_structure(pdb_content: str):
    """Render 3D structure with dark background"""
    try:
        view = py3Dmol.view(width=800, height=450)
        view.addModel(pdb_content, 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.setBackgroundColor('#000000')
        view.zoomTo()
        return view
    except Exception as e:
        st.error(f"3D Viewer Error: {str(e)}")
        return None

# ==================================================
# 6. APPLICATION UI
# ==================================================
st.title("🧬 ProCap Search Diagnostic Suite")
st.markdown("##### High-Throughput Structural Homology & Functional Annotation Pipeline v3.0")
st.markdown("##### **All THREE Methods: RMSD + Sequence Identity + Distance Matrix**")
st.divider()

tmp_root = Path(tempfile.gettempdir()) / "procap_professional"
query_dir = tmp_root / "query"
query_dir.mkdir(parents=True, exist_ok=True)
db_dir = tmp_root / "db"
db_dir.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    # Get available chains dynamically
    st.markdown('<div class="info-box"><b>🔗 Chain Selection:</b><br>Select the chain to analyze in your query structure.</div>', unsafe_allow_html=True)
    chain_id = st.text_input(
        "Target Chain ID",
        value="A",
        max_chars=1,
        help="Chain ID (A, B, C, etc.). Leave as A if unsure."
    ).strip().upper() or "A"
    
    st.markdown('<div class="info-box"><b>🗄️ Core Database:</b><br>Strictly utilizes verified bacterial capsule proteins from known_capsule_proteins.csv.</div>', unsafe_allow_html=True)
    if st.button("🔄 Initialize Local CSV Database", use_container_width=True):
        with st.spinner("📥 Syncing PDB structures from RCSB..."):
            for f in db_dir.glob("*.pdb"):
                f.unlink()
            d = build_database_from_reference_pdb_ids(
                db_dir, REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist()
            )
            st.markdown(
                f'<div class="success-box">✅ Database Ready: {len(d)} verified structures loaded.</div>',
                unsafe_allow_html=True
            )

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Structural Diagnostic",
    "🧪 3D Viewer & Alignments",
    "📊 Database Analytics",
    "📖 Algorithm Details",
    "🧬 Scoring Breakdown"
])

# =========================
# TAB 1: DIAGNOSTIC
# =========================
with tab1:
    col_query, col_results = st.columns([1, 1.8], gap="large")
    
    with col_query:
        st.subheader("Query Configuration")
        query_mode = st.radio("Query Source", ["RCSB PDB Database", "Local File Upload"])
        
        if query_mode == "RCSB PDB Database":
            pdb_id = st.text_input(
                "Enter 4-Letter PDB ID",
                value="1R3F",
                max_chars=4
            ).strip().upper()
        else:
            uploaded_query = st.file_uploader("Upload .pdb File", type=["pdb"])
            pdb_id = None
        
        st.markdown("<br>", unsafe_allow_html=True)
        execute_btn = st.button("▶ EXECUTE ALL 3 METHODS", type="primary", use_container_width=True)
    
    with col_results:
        if execute_btn:
            if not list(db_dir.glob("*.pdb")):
                st.error("❌ Database is empty. Click 'Initialize Local CSV Database' in sidebar.")
            else:
                query_pdb_path = None
                if query_mode == "RCSB PDB Database" and len(pdb_id) == 4:
                    query_pdb_path = query_dir / f"{pdb_id}.pdb"
                    if not query_pdb_path.exists():
                        with st.spinner(f"Downloading {pdb_id}..."):
                            if not download_pdb_by_id(pdb_id, query_pdb_path):
                                query_pdb_path = None
                elif query_mode == "Local File Upload" and uploaded_query:
                    query_pdb_path = query_dir / uploaded_query.name
                    query_pdb_path.write_bytes(uploaded_query.getvalue())
                
                if not query_pdb_path:
                    st.error("❌ Failed to load Query Structure.")
                else:
                    with st.spinner("🔬 Executing 3 structural methods + sequence analysis..."):
                        results_df, diagnostics = run_comprehensive_diagnostic(
                            query_pdb_path, db_dir, chain_id
                        )
                        
                        qp = PDBWrapper(str(query_pdb_path))
                        qp.parse()
                        exp_meta = qp.get_experimental_metadata(chain_id)
                    
                    if diagnostics["status"] == "error":
                        st.markdown(
                            f'<div class="alert-low">❌ ERROR: {diagnostics["message"]}</div>',
                            unsafe_allow_html=True
                        )
                    elif results_df.empty:
                        st.markdown(
                            '<div class="alert-low">⚠️ No homologous structures found in database.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.session_state['query_pdb_content'] = query_pdb_path.read_text()
                        st.session_state['query_seq'] = qp.get_sequence(chain_id)
                        st.session_state['results_df'] = results_df
                        
                        top_hit = results_df.iloc[0]
                        confidence, alert_class = classify_confidence_level(
                            top_hit["Consensus_Score"]
                        )
                        prediction = predict_biological_function(results_df)
                        
                        st.markdown(
                            f'<div class="{alert_class}">🧬 CONSENSUS SCORE: {top_hit["Consensus_Score"]:.3f}/1.000 | {confidence}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # ===== DISPLAY ALL THREE METHODS RESULTS =====
                        st.markdown("### 📊 Individual Method Scores (Top Hit)")
                        
                        method_col1, method_col2, method_col3 = st.columns(3)
                        
                        with method_col1:
                            st.markdown(
                                f'<div class="method-box">'
                                f'<b>METHOD 1: RMSD</b><br>'
                                f'Score: {top_hit["RMSD_Score"]:.3f}/1.000<br>'
                                f'RMSD Value: {top_hit["RMSD_Å"]:.2f} Å<br>'
                                f'<small>Structural alignment quality</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        with method_col2:
                            st.markdown(
                                f'<div class="method-box">'
                                f'<b>METHOD 2: SEQUENCE</b><br>'
                                f'Score: {top_hit["Seq_Score"]:.3f}/1.000<br>'
                                f'Identity: {top_hit["Seq_Identity_%"]:.1f}%<br>'
                                f'<small>Evolutionary conservation</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        with method_col3:
                            st.markdown(
                                f'<div class="method-box">'
                                f'<b>METHOD 3: TOPOLOGY</b><br>'
                                f'Score: {top_hit["DM_Correlation"]:.3f}/1.000<br>'
                                f'Corr: {top_hit["DM_Correlation"]:.3f}<br>'
                                f'<small>Distance matrix correlation</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        with st.container(border=True):
                            st.markdown("### Biological Annotation")
                            st.write(
                                f"This query functions as a **{prediction['function']}**, "
                                f"belonging to **{prediction['family']}** family "
                                f"({prediction['structure']})."
                            )
                            st.markdown("**Top Reference Hit**")
                            col_top1, col_top2, col_top3 = st.columns(3)
                            col_top1.write(
                                f"**PDB ID:** `{top_hit['Target_PDB']}`\n\n"
                                f"**Organism:** {top_hit.get('Organism', 'Unknown')}"
                            )
                            col_top2.write(
                                f"**Gene:** {top_hit.get('Gene_Name', 'Unknown')}\n\n"
                                f"**Coverage:** {top_hit['Query_Coverage_%']:.1f}%"
                            )
                            col_top3.write(
                                f"**Function:** {top_hit.get('Function', 'Unknown')}\n\n"
                                f"**Family:** {top_hit.get('Protein_Family', 'Unknown')}"
                            )
                        
                        with st.container(border=True):
                            st.markdown("### Structural & PhysChem Profile (Query)")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric(
                                "Length",
                                f"{diagnostics.get('query_seq_length', 0)} AA"
                            )
                            m2.metric(
                                "Mol. Weight",
                                f"{diagnostics.get('mw_kda', 0):.1f} kDa"
                            )
                            m3.metric(
                                "Isoelectric Pt",
                                f"{diagnostics.get('pi', 0):.2f}"
                            )
                            m4.metric(
                                "Instability Idx",
                                f"{diagnostics.get('instability', 0):.1f}"
                            )
                            
                            m5, m6, m7, m8 = st.columns(4)
                            m5.metric(
                                "GRAVY",
                                f"{diagnostics.get('gravy', 0):.3f}",
                                help="Hydropathy Index"
                            )
                            m6.metric(
                                "Extinction Coeff.",
                                f"{diagnostics.get('extinction_coeff', 0):.0f}",
                                help="M⁻¹ cm⁻¹"
                            )
                            m7.metric(
                                "X-Ray Resolution",
                                f"{exp_meta.get('resolution', np.nan):.2f} Å"
                                if not pd.isna(exp_meta.get('resolution', np.nan)) else "N/A"
                            )
                            m8.metric(
                                "Mean B-Factor",
                                f"{exp_meta.get('mean_b_factor', np.nan):.1f}"
                                if not pd.isna(exp_meta.get('mean_b_factor', np.nan)) else "N/A"
                            )
                        
                        st.markdown("### Search Results (All 3 Methods Combined)")
                        display_cols = [
                            "Target_PDB", "Organism", "Protein_Family",
                            "Consensus_Score", "Confidence",
                            "RMSD_Score", "Seq_Score", "DM_Correlation",
                            "Seq_Identity_%", "RMSD_Å"
                        ]
                        st.dataframe(
                            results_df[display_cols].head(15),
                            use_container_width=True,
                            hide_index=True
                        )

# ==================================================
# TAB 2: 3D VIEWER & ALIGNMENTS
# ==================================================
with tab2:
    st.subheader("3D Structure Visualization & Sequence Interrogation")
    
    if 'query_pdb_content' in st.session_state:
        col3d, colseq = st.columns([1.3, 1], gap="large")
        
        with col3d:
            st.markdown("### Interactive 3D Query Topology")
            st.caption("Color-coded by secondary structure. Use mouse to rotate/zoom.")
            with st.container(border=True):
                view = render_3d_structure(st.session_state['query_pdb_content'])
                if view:
                    components.html(view._make_html(), height=480)
        
        with colseq:
            if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
                top = st.session_state['results_df'].iloc[0]
                st.markdown(f"### Alignment vs {top['Target_PDB']}")
                st.caption("BioPython Global Alignment (| = Exact Match)")
                
                qseq = st.session_state.get('query_seq')
                if qseq and top.get('Target_Seq'):
                    align_html = generate_alignment_visualization(qseq, top['Target_Seq'], blocksize=40)
                    st.markdown(align_html, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    fasta_str = (
                        f">Query\n{qseq}\n>Target_{top['Target_PDB']}\n{top['Target_Seq']}\n"
                    )
                    st.download_button(
                        "📥 Export FASTA Alignment",
                        data=fasta_str,
                        file_name=f"alignment_{top['Target_PDB']}.fasta",
                        mime="text/plain",
                        use_container_width=True
                    )
    else:
        st.markdown(
            '<div class="info-box">💡 Execute diagnostic in Tab 1 to generate 3D models.</div>',
            unsafe_allow_html=True
        )

# ==================================================
# TAB 3: DATABASE ANALYTICS
# ==================================================
with tab3:
    st.subheader("Database Population Analytics")
    if not REFERENCE_DF.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.pie(
                    REFERENCE_DF, names="Organism",
                    title="Pathogen Representation", hole=0.3,
                    template="plotly_dark"
                ),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                px.bar(
                    REFERENCE_DF["Protein_Family"].value_counts().reset_index(),
                    x="Protein_Family", y="count",
                    title="Domain Families", template="plotly_dark"
                ),
                use_container_width=True
            )
        st.dataframe(REFERENCE_DF, use_container_width=True, hide_index=True)

# ==================================================
# TAB 4: ALGORITHM DETAILS
# ==================================================
with tab4:
    st.subheader("THREE-METHOD Consensus Algorithm")
    st.markdown("""
    ### Method 1: RMSD-Based Structural Superposition
    **What:** Root Mean Square Deviation of alpha-carbon backbone atoms
    - **Range:** 0–10 Å (normalized to 0–1 score)
    - **Formula:** `Score = max(0, 1 - RMSD/10)`
    - **Strength:** Direct 3D geometric measurement, physics-grounded
    - **Limitation:** Requires identical coordinate counts
    
    ### Method 2: Sequence Identity (Needleman-Wunsch Global Alignment)
    **What:** Evolutionary primary structure conservation
    - **Range:** 0–100% identity (0–1 score)
    - **Formula:** `Score = Matching_Residues / max(Seq_A, Seq_B)`
    - **Strength:** Detects remote homologs, evolutionary relationships
    - **Limitation:** Low sensitivity for divergent sequences (< 20% identity)
    
    ### Method 3: Distance Matrix Topology Correlation
    **What:** Pearson correlation of pairwise Euclidean distances
    - **Range:** -1 to +1 (normalized to 0–1)
    - **Formula:** `Score = (Correlation + 1) / 2`
    - **Strength:** Topology-based, resilient to loop variations
    - **Limitation:** Requires matching coordinate counts
    
    ### Consensus Scoring Formula:
    ```
    Consensus_Score = (0.33 × RMSD_Score) + (0.33 × Seq_Score) + (0.34 × DM_Score)
    ```
    
    ### Confidence Classification:
    - **IDENTICAL (≥ 0.90):** All methods agree; Very strong homology
    - **HIGHLY SIMILAR (0.75–0.89):** Two methods agree; Strong homology
    - **MODERATELY SIMILAR (0.55–0.74):** Single method agreement; Moderate
    - **WEAKLY SIMILAR (< 0.55):** Methods diverge; Weak match
    """)

# ==================================================
# TAB 5: SCORING BREAKDOWN
# ==================================================
with tab5:
    st.subheader("Individual Method Scores Breakdown")
    if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
        results = st.session_state['results_df'].head(10).copy()
        
        fig = px.bar(
            results,
            x="Target_PDB",
            y=["RMSD_Score", "Seq_Score", "DM_Correlation"],
            title="Individual Method Contributions to Consensus Score",
            labels={"value": "Score", "variable": "Method"},
            template="plotly_dark",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Top 10 Hits - Detailed Breakdown")
        breakdown_cols = [
            "Target_PDB", "Organism", "Consensus_Score",
            "RMSD_Score", "Seq_Score", "DM_Correlation",
            "RMSD_Å", "Seq_Identity_%"
        ]
        st.dataframe(
            results[breakdown_cols],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Run diagnostic first to see method breakdown")
