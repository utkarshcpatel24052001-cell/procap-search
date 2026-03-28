"""
ProCap Search Professional Diagnostic Suite v3.0
High-Throughput Structural Homology & Functional Annotation Pipeline
Production-Grade Bioinformatics Application
Author: Utkarsh Patel
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
from Bio import pairwise2
from Bio.PDB import PDBParser as BioPDBParser
from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from scipy.spatial.distance import cdist

# ==================================================
# 1. PAGE CONFIG & PROFESSIONAL STYLING
# ==================================================
st.set_page_config(
    page_title="ProCap Professional | v3.0",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern LIMS CSS - Clean Sans-Serif, High Contrast, Card-based
st.markdown(
    """
    <style>
    /* Clean, modern typography for dense data dashboards */
    html, body, .stMarkdown, p, li, span, label {
        font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif !important;
        font-size: 16px !important;
        color: #1e293b;
    }
    
    /* Headers */
    h1, h2, h3, h4 { color: #0f172a !important; font-weight: 700 !important; }
    h3 { font-size: 20px !important; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;}

    /* Metric Cards */
    div[data-testid="stMetricLabel"] p {
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #2563eb !important;
    }

    /* Status Banners */
    .alert-high { background-color: #f0fdf4; border-left: 6px solid #16a34a; padding: 16px; border-radius: 6px; color: #166534; font-weight: 600; margin-bottom: 15px;}
    .alert-medium { background-color: #fffbeb; border-left: 6px solid #d97706; padding: 16px; border-radius: 6px; color: #92400e; font-weight: 600; margin-bottom: 15px;}
    .alert-low { background-color: #fef2f2; border-left: 6px solid #dc2626; padding: 16px; border-radius: 6px; color: #991b1b; font-weight: 600; margin-bottom: 15px;}
    .info-box { background-color: #f0f9ff; border-left: 6px solid #2563eb; padding: 16px; border-radius: 6px; color: #1e40af; font-weight: 600; margin-bottom: 15px;}

    /* Sequence Alignment Box */
    .seq-align-box {
        font-family: 'Fira Code', 'Courier New', monospace !important;
        background-color: #f8fafc;
        padding: 16px;
        border-radius: 6px;
        border: 1px solid #cbd5e1;
        overflow-x: auto;
        font-size: 14px !important;
        line-height: 1.4 !important;
        white-space: pre !important;
        color: #334155;
    }
    
    .stButton > button { font-weight: 600 !important; border-radius: 6px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 2. CORE BIOPYTHON & MATH UTILITIES
# ==================================================
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

DIAGNOSTIC_THRESHOLDS = {"Identical": 0.90, "Highly_Similar": 0.75, "Moderately_Similar": 0.55}

class PDBWrapper:
    """Professional PDB parser with robust error handling"""
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
        except Exception: return False

    def get_coordinates(self, chain_id="A") -> Optional[np.ndarray]:
        if not self.structure: return None
        try:
            coords = [res["CA"].get_coord() for res in self.structure[0][chain_id] if "CA" in res]
            return np.array(coords, dtype=float) if coords else None
        except KeyError: return None

    def get_sequence(self, chain_id="A") -> Optional[str]:
        if not self.structure: return None
        try:
            return "".join([THREE_TO_ONE.get(res.get_resname(), "X") for res in self.structure[0][chain_id]])
        except KeyError: return None
        
    def get_experimental_metadata(self, chain_id="A"):
        resolution = self.header.get("resolution", np.nan)
        b_factors = []
        try:
            if self.structure:
                b_factors = [res["CA"].get_bfactor() for res in self.structure[0][chain_id] if "CA" in res]
        except KeyError: pass
        return {"resolution": resolution, "mean_b_factor": np.mean(b_factors) if b_factors else np.nan}

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> Optional[float]:
    if coords1 is None or coords2 is None or coords1.shape != coords2.shape: return None
    return float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1))))

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1: np.ndarray, dm2: np.ndarray) -> float:
    flat1, flat2 = dm1[np.triu_indices_from(dm1, k=1)], dm2[np.triu_indices_from(dm2, k=1)]
    corr = np.corrcoef(flat1, flat2)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def download_pdb_by_id(pdb_id: str, out_path: Path) -> bool:
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id.strip().upper()}.pdb", timeout=15)
        r.raise_for_status()
        out_path.write_text(r.text)
        return True
    except Exception: return False

def get_advanced_physchem_properties(seq: str) -> Dict[str, float]:
    """Calculates advanced biochemical properties."""
    clean_seq = seq.replace("X", "").replace("U", "")
    if len(clean_seq) < 2:
        return {k: np.nan for k in ["mw_kda", "pi", "instability", "aromaticity", "gravy", "extinction_coeff", "helix_prob"]}
    try:
        params = ProtParam.ProteinAnalysis(clean_seq)
        pi = IsoelectricPoint(clean_seq).pH()
        return {
            "mw_kda": params.molecular_weight() / 1000,
            "pi": float(pi),
            "instability": params.instability_index(),
            "aromaticity": params.aromaticity(),
            "gravy": params.gravy(),
            "extinction_coeff": params.molar_extinction_coefficient()[0],
            "helix_prob": params.secondary_structure_fraction()[0],
        }
    except Exception: return {k: np.nan for k in ["mw_kda", "pi", "instability", "aromaticity", "gravy", "extinction_coeff", "helix_prob"]}

def classify_confidence_level(score: float) -> Tuple[str, str]:
    if score >= DIAGNOSTIC_THRESHOLDS["Identical"]: return "IDENTICAL MATCH", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Highly_Similar"]: return "HIGHLY SIMILAR", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Moderately_Similar"]: return "MODERATELY SIMILAR", "alert-medium"
    else: return "WEAKLY SIMILAR", "alert-low"

# ==================================================
# 3. DATA LOADING & CUSTOM ZIP DATABASE PIPELINE
# ==================================================
@st.cache_data
def load_reference_csv() -> pd.DataFrame:
    try: return pd.read_csv("data/known_capsule_proteins.csv")
    except FileNotFoundError: return pd.DataFrame(columns=["PDB_ID", "Organism", "Gene_Name", "Function", "Protein_Family", "Structure_Type"])

REFERENCE_DF = load_reference_csv()

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids: list):
    db_dir.mkdir(parents=True, exist_ok=True)
    d, f = [], []
    for pid in pdb_ids:
        out = db_dir / f"{pid}.pdb"
        if out.exists() or download_pdb_by_id(pid, out): d.append(pid)
        else: f.append(pid)
    return d, f

def extract_zip_to_db(zip_bytes: bytes, db_dir: Path):
    """Extracts custom .pdb files from a user's uploaded .zip archive."""
    db_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for file_info in z.infolist():
            if file_info.filename.endswith('.pdb'):
                # Extract to flat directory regardless of zip folders
                out_path = db_dir / Path(file_info.filename).name
                out_path.write_bytes(z.read(file_info.filename))
                extracted_count += 1
    return extracted_count

def run_comprehensive_diagnostic(query_pdb: Path, db_dir: Path, chain_id: str) -> Tuple[pd.DataFrame, Dict]:
    qp = PDBWrapper(str(query_pdb))
    if not qp.parse(): return pd.DataFrame(), {"status": "error", "message": "Failed to parse query PDB"}
    
    qcoords, qseq = qp.get_coordinates(chain_id), qp.get_sequence(chain_id)
    if qcoords is None or not qseq: return pd.DataFrame(), {"status": "error", "message": f"No valid chain {chain_id}"}
    
    qdm = calculate_distance_matrix(qcoords)
    physchem_query = get_advanced_physchem_properties(qseq)
    
    results = []
    for pdb_file in sorted(db_dir.glob("*.pdb")):
        target_id = pdb_file.stem.upper()
        if target_id == query_pdb.stem.upper(): continue
        
        dp = PDBWrapper(str(pdb_file))
        if not dp.parse(): continue
        dcoords, dseq = dp.get_coordinates(chain_id), dp.get_sequence(chain_id)
        if dcoords is None or not dseq: continue
        
        # 1. RMSD
        rmsd_val, r_score = np.nan, 0.0
        if len(qcoords) == len(dcoords):
            rmsd_val = calculate_rmsd(qcoords, dcoords)
            if rmsd_val is not None: r_score = max(0.0, 1.0 - (rmsd_val / 10.0))
        
        # 2. Sequence Alignment
        alns = pairwise2.align.globalxx(qseq, dseq, one_alignment_only=True)
        s_score, aligned_length = 0.0, 0
        if alns:
            a = alns[0]
            s_score = sum(x == y for x, y in zip(a.seqA, a.seqB)) / max(len(qseq), len(dseq))
            aligned_length = len(a.seqA)
        
        # 3. Distance Matrix
        d_score = 0.0
        if qdm.shape == calculate_distance_matrix(dcoords).shape:
            d_score = (distance_matrix_correlation(qdm, calculate_distance_matrix(dcoords)) + 1.0) / 2.0
        
        final_score = (0.33 * r_score) + (0.33 * s_score) + (0.34 * d_score)
        confidence, _ = classify_confidence_level(final_score)
        
        if final_score > 0:
            results.append({
                "Target_PDB": target_id, "Consensus_Score": final_score, "Confidence": confidence,
                "RMSD_Å": rmsd_val, "Seq_Identity_%": s_score * 100, "DM_Correlation": d_score,
                "Query_Coverage_%": (aligned_length / len(qseq) * 100) if len(qseq) > 0 else 0, "Target_Seq": dseq,
            })
    
    df = pd.DataFrame(results)
    if not df.empty:
        # Merge with CSV data. Hits from custom ZIPs will simply show NaN for CSV metadata.
        ref_meta = REFERENCE_DF[["PDB_ID", "Organism", "Gene_Name", "Function", "Protein_Family", "Structure_Type"]].copy()
        ref_meta["PDB_ID"] = ref_meta["PDB_ID"].str.upper()
        df = df.merge(ref_meta, left_on="Target_PDB", right_on="PDB_ID", how="left").drop("PDB_ID", axis=1)
        
        # Default unknown organisms to "Custom Upload"
        df["Organism"] = df["Organism"].fillna("Custom Upload")
        df["Gene_Name"] = df["Gene_Name"].fillna("Unknown")
        
        df = df.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    
    diagnostics = {
        "status": "success", "query_seq_length": len(qseq), "total_matches": len(results),
        **physchem_query
    }
    return df, diagnostics

# ==================================================
# 4. VISUALIZATION ENGINES
# ==================================================
def generate_alignment_visualization(seqA: str, seqB: str, blocksize: int = 60) -> str:
    alns = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)
    if not alns: return "<p>Alignment failed.</p>"
    a = alns[0]
    
    html = '<div class="seq-align-box">\n'
    for i in range(0, len(a.seqA), blocksize):
        match_line = "".join(["|" if a.seqA[j] == a.seqB[j] and a.seqA[j] != "-" else "." if a.seqA[j] != "-" and a.seqB[j] != "-" else " " for j in range(i, min(i + blocksize, len(a.seqA)))])
        html += f"<span style='color:#2563eb; font-weight:bold;'>Query: </span> {a.seqA[i:i+blocksize]}\n"
        html += f"<span style='color:#16a34a; font-weight:bold;'>Match: </span> {match_line}\n"
        html += f"<span style='color:#dc2626; font-weight:bold;'>Target:</span> {a.seqB[i:i+blocksize]}\n\n"
    html += '</div>'
    return html

def render_3d_structure(pdb_content: str):
    import py3Dmol
    view = py3Dmol.view(width=800, height=450)
    view.addModel(pdb_content, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.setBackgroundColor('#ffffff')
    view.zoomTo()
    return view

# ==================================================
# 5. APPLICATION UI
# ==================================================
st.title("🧬 ProCap Search Diagnostic Suite")
st.markdown("##### High-Throughput Structural Homology & Functional Annotation Pipeline v3.0")
st.divider()

tmp_root = Path(tempfile.gettempdir()) / "procap_professional"
query_dir = tmp_root / "query"; query_dir.mkdir(parents=True, exist_ok=True)
db_dir = tmp_root / "db"; db_dir.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    chain_id = st.text_input("Target Chain ID", value="A", max_chars=1, help="Biological assembly chain.")
    
    st.subheader("Database Target Selection")
    use_reference = st.checkbox("Curated CSV Database (RCSB)", value=True, help="Include the 23 verified bacterial capsule proteins.")
    uploaded_db_zip = st.file_uploader("Custom Database (.zip)", type=["zip"], help="Upload a ZIP of custom .pdb models (e.g., AlphaFold).")
    
    if st.button("🔄 Sync Database Memory", use_container_width=True):
        with st.spinner("📥 Preparing execution environment..."):
            # Clear old DB state
            for f in db_dir.glob("*.pdb"): f.unlink()
            
            db_count = 0
            if use_reference:
                d, _ = build_database_from_reference_pdb_ids(db_dir, REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist())
                db_count += len(d)
            if uploaded_db_zip:
                db_count += extract_zip_to_db(uploaded_db_zip.getvalue(), db_dir)
                
            st.success(f"✅ Database Ready: {db_count} structures targeted.")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Structural Diagnostic", "🧪 3D Viewer & Alignments", "📊 Database Analytics", "📖 Documentation"])

# =========================
# TAB 1: DIAGNOSTIC
# =========================
with tab1:
    col_query, col_results = st.columns([1, 1.8], gap="large")
    
    with col_query:
        st.subheader("Query Configuration")
        query_mode = st.radio("Query Source", ["RCSB PDB Database", "Local File Upload"])
        
        if query_mode == "RCSB PDB Database":
            pdb_id = st.text_input("Enter 4-Letter PDB ID", value="1R3F", max_chars=4).strip().upper()
        else:
            uploaded_query = st.file_uploader("Upload .pdb File", type=["pdb"])
            pdb_id = None
        
        st.markdown("<br>", unsafe_allow_html=True)
        execute_btn = st.button("▶ EXECUTE PIPELINE", type="primary", use_container_width=True)
    
    with col_results:
        if execute_btn:
            # Enforce Database Selection
            if not list(db_dir.glob("*.pdb")):
                st.error("❌ Database empty. Please click 'Sync Database Memory' in the sidebar first.")
            else:
                query_pdb_path = None
                if query_mode == "RCSB PDB Database" and len(pdb_id) == 4:
                    query_pdb_path = query_dir / f"{pdb_id}.pdb"
                    if not query_pdb_path.exists():
                        with st.spinner(f"Downloading {pdb_id}..."):
                            if not download_pdb_by_id(pdb_id, query_pdb_path): query_pdb_path = None
                elif query_mode == "Local File Upload" and uploaded_query:
                    query_pdb_path = query_dir / uploaded_query.name
                    query_pdb_path.write_bytes(uploaded_query.getvalue())
                
                if not query_pdb_path:
                    st.error("❌ Failed to load Query Structure. Verify PDB ID.")
                else:
                    with st.spinner("🔬 Executing structural alignments against database..."):
                        results_df, diagnostics = run_comprehensive_diagnostic(query_pdb_path, db_dir, chain_id)
                        
                        # Extract Meta
                        qp = PDBWrapper(str(query_pdb_path))
                        qp.parse()
                        exp_meta = qp.get_experimental_metadata(chain_id)
                    
                    if diagnostics["status"] == "error":
                        st.error(f"❌ {diagnostics['message']}")
                    elif results_df.empty:
                        st.warning("⚠️ No homologous structures found matching the minimum criteria.")
                    else:
                        st.session_state['query_pdb_content'] = query_pdb_path.read_text()
                        st.session_state['results_df'] = results_df
                        
                        top_hit = results_df.iloc[0]
                        confidence, alert_class = classify_confidence_level(top_hit["Consensus_Score"])
                        
                        # Functional Deduction
                        func_str = str(top_hit.get('Function', 'Unknown'))
                        fam_str = str(top_hit.get('Protein_Family', 'Unknown'))
                        struc_str = str(top_hit.get('Structure_Type', 'Unknown'))
                        
                        st.markdown(f'<div class="{alert_class}">🧬 CONFIDENCE: {confidence} | Score: {top_hit["Consensus_Score"]:.3f}/1.000</div>', unsafe_allow_html=True)
                        
                        with st.container(border=True):
                            st.markdown("### Biological Triage & Annotation")
                            st.write(f"Based on the top cluster consensus, this query functions as a **{func_str}**, belonging to the **{fam_str}** family ({struc_str}).")
                            st.markdown("**Top Database Hit**")
                            col_top1, col_top2, col_top3 = st.columns(3)
                            col_top1.write(f"**PDB ID:** `{top_hit['Target_PDB']}`\n\n**Organism:** {top_hit.get('Organism', 'Custom Upload')}")
                            col_top2.write(f"**Gene:** {top_hit.get('Gene_Name', 'Unknown')}\n\n**Coverage:** {top_hit['Query_Coverage_%']:.1f}%")
                            col_top3.write(f"**Seq Identity:** {top_hit['Seq
