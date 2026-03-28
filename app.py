"""
ProCap Search Professional Diagnostic Suite v3.0
High-Throughput Structural Homology & Functional Annotation Pipeline
Production-Grade Bioinformatics Application
Author: Utkarsh Patel
"""

from __future__ import annotations

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
    default_props = {"mw_kda": np.nan, "pi": np.nan, "instability": np.nan, "aromaticity": np.nan, "gravy": np.nan, "extinction_coeff": np.nan}
    if len(clean_seq) < 2: return default_props
    try:
        params = ProtParam.ProteinAnalysis(clean_seq)
        return {
            "mw_kda": params.molecular_weight() / 1000,
            "pi": params.isoelectric_point(),
            "instability": params.instability_index(),
            "aromaticity": params.aromaticity(),
            "gravy": params.gravy(),
            "extinction_coeff": params.molar_extinction_coefficient()[0],
        }
    except Exception: return default_props

def classify_confidence_level(score: float) -> Tuple[str, str]:
    if score >= DIAGNOSTIC_THRESHOLDS["Identical"]: return "IDENTICAL MATCH", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Highly_Similar"]: return "HIGHLY SIMILAR", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Moderately_Similar"]: return "MODERATELY SIMILAR", "alert-medium"
    else: return "WEAKLY SIMILAR", "alert-low"

# ==================================================
# 3. DATA LOADING & DATABASE PIPELINE
# ==================================================
@st.cache_data
def load_reference_csv() -> pd.DataFrame:
    try: return pd.read_csv("data/known_capsule_proteins.csv")
    except FileNotFoundError: return pd.DataFrame(columns=["PDB_ID", "Organism", "Gene_Name", "Function", "Protein_Family", "Structure_Type"])

REFERENCE_DF = load_reference_csv()

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids: list):
    db_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for pid in pdb_ids:
        out = db_dir / f"{pid}.pdb"
        if out.exists() or download_pdb_by_id(pid, out): downloaded.append(pid)
    return downloaded

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
        ref_meta = REFERENCE_DF.copy()
        ref_meta["PDB_ID"] = ref_meta["PDB_ID"].str.upper()
        df = df.merge(ref_meta, left_on="Target_PDB", right_on="PDB_ID", how="inner").drop("PDB_ID", axis=1)
        df = df.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    
    diagnostics = {
        "status": "success", "query_seq_length": len(qseq), "total_matches": len(results),
        **physchem_query
    }
    return df, diagnostics

def predict_biological_function(results_df: pd.DataFrame) -> dict:
    top_3 = results_df.head(3).dropna(subset=['Protein_Family', 'Structure_Type', 'Function'])
    if top_3.empty: return {"family": "Unknown", "structure": "Unknown", "function": "Unknown", "confidence": "Low"}
    return {
        "family": top_3['Protein_Family'].mode()[0], "structure": top_3['Structure_Type'].mode()[0],
        "function": top_3['Function'].mode()[0], "confidence": "High" if top_3['Protein_Family'].nunique() == 1 else "Moderate"
    }

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
    
    st.markdown('<div class="info-box"><b>🗄️ Core Database:</b><br>Strictly utilizes verified bacterial capsule proteins from known_capsule_proteins.csv.</div>', unsafe_allow_html=True)
    if st.button("🔄 Initialize Local CSV Database", use_container_width=True):
        with st.spinner("📥 Syncing PDB structures from RCSB..."):
            for f in db_dir.glob("*.pdb"): f.unlink() # Clear DB
            d = build_database_from_reference_pdb_ids(db_dir, REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist())
            st.success(f"✅ Database Ready: {len(d)} verified structures loaded.")

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
            if not list(db_dir.glob("*.pdb")):
                st.error("❌ Database is empty. Please click 'Initialize Local CSV Database' in the sidebar first.")
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
                    with st.spinner("🔬 Executing structural alignments against CSV database..."):
                        results_df, diagnostics = run_comprehensive_diagnostic(query_pdb_path, db_dir, chain_id)
                        
                        qp = PDBWrapper(str(query_pdb_path))
                        qp.parse()
                        exp_meta = qp.get_experimental_metadata(chain_id)
                    
                    if diagnostics["status"] == "error":
                        st.error(f"❌ {diagnostics['message']}")
                    elif results_df.empty:
                        st.warning("⚠️ No homologous structures found within the reference database.")
                    else:
                        st.session_state['query_pdb_content'] = query_pdb_path.read_text()
                        st.session_state['results_df'] = results_df
                        
                        top_hit = results_df.iloc[0]
                        confidence, alert_class = classify_confidence_level(top_hit["Consensus_Score"])
                        prediction = predict_biological_function(results_df)
                        
                        st.markdown(f'<div class="{alert_class}">🧬 CONFIDENCE: {confidence} | Score: {top_hit["Consensus_Score"]:.3f}/1.000</div>', unsafe_allow_html=True)
                        
                        with st.container(border=True):
                            st.markdown("### Biological Triage & Annotation")
                            st.write(f"Based on the top cluster consensus, this query functions as a **{prediction['function']}**, belonging to the **{prediction['family']}** family ({prediction['structure']}).")
                            st.markdown("**Top Reference Hit**")
                            col_top1, col_top2, col_top3 = st.columns(3)
                            col_top1.write(f"**PDB ID:** `{top_hit['Target_PDB']}`\n\n**Organism:** {top_hit.get('Organism', 'Unknown')}")
                            col_top2.write(f"**Gene:** {top_hit.get('Gene_Name', 'Unknown')}\n\n**Coverage:** {top_hit['Query_Coverage_%']:.1f}%")
                            col_top3.write(f"**Seq Identity:** {top_hit['Seq_Identity_%']:.1f}%\n\n**RMSD:** {top_hit['RMSD_Å']:.2f} Å" if not pd.isna(top_hit['RMSD_Å']) else "**RMSD:** N/A")
                        
                        with st.container(border=True):
                            st.markdown("### Structural & PhysChem Profile (Query)")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Length", f"{diagnostics.get('query_seq_length', 0)} AA")
                            m2.metric("Mol. Weight", f"{diagnostics.get('mw_kda', 0):.1f} kDa")
                            m3.metric("Isoelectric Pt", f"{diagnostics.get('pi', 0):.2f}")
                            m4.metric("Instability Idx", f"{diagnostics.get('instability', 0):.1f}")
                            
                            m5, m6, m7, m8 = st.columns(4)
                            m5.metric("GRAVY", f"{diagnostics.get('gravy', 0):.3f}", help="Hydropathy. Positive = Hydrophobic. Negative = Soluble.")
                            m6.metric("Extinction Coeff.", f"{diagnostics.get('extinction_coeff', 0):.0f}", help="M⁻¹ cm⁻¹")
                            m7.metric("X-Ray Resolution", f"{exp_meta.get('resolution', np.nan):.2f} Å" if not pd.isna(exp_meta.get('resolution', np.nan)) else "N/A")
                            m8.metric("Mean B-Factor", f"{exp_meta.get('mean_b_factor', np.nan):.1f}" if not pd.isna(exp_meta.get('mean_b_factor', np.nan)) else "N/A")
                        
                        st.markdown("### Search Results Topology Matrix")
                        display_cols = ["Target_PDB", "Organism", "Protein_Family", "Consensus_Score", "Confidence", "Seq_Identity_%", "RMSD_Å"]
                        st.dataframe(results_df[display_cols].head(15), use_container_width=True, hide_index=True)

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
                components.html(view._make_html(), height=480)
        
        with colseq:
            if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
                top = st.session_state['results_df'].iloc[0]
                st.markdown(f"### Alignment vs {top['Target_PDB']}")
                st.caption("BioPython Global Alignment (| = Exact Match)")
                
                qp = PDBWrapper("temp.pdb")
                qp.structure = BioPDBParser(QUIET=True).get_structure("tmp", io.StringIO(st.session_state['query_pdb_content']))
                qseq = qp.get_sequence(chain_id)
                
                align_html = generate_alignment_visualization(qseq, top['Target_Seq'], blocksize=40)
                st.markdown(align_html, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                fasta_str = f">Query\n{qseq}\n>Target_{top['Target_PDB']}\n{top['Target_Seq']}\n"
                st.download_button("📥 Export FASTA Alignment", data=fasta_str, file_name=f"alignment_{top['Target_PDB']}.fasta", mime="text/plain", use_container_width=True)
    else:
        st.markdown('<div class="info-box">💡 Execute a diagnostic in Tab 1 to generate 3D models and alignments.</div>', unsafe_allow_html=True)

# ==================================================
# TAB 3 & 4: ANALYTICS & DOCS
# ==================================================
with tab3:
    st.subheader("Database Population Analytics")
    if not REFERENCE_DF.empty:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.pie(REFERENCE_DF, names="Organism", title="Pathogen Representation", hole=0.3), use_container_width=True)
        with c2: st.plotly_chart(px.bar(REFERENCE_DF["Protein_Family"].value_counts().reset_index(), x="Protein_Family", y="count", title="Domain Families"), use_container_width=True)
        st.dataframe(REFERENCE_DF, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Algorithm Specifications")
    st.markdown("""
    **1. RMSD-Based Superposition:** Physics-based 3D geometric measurement across alpha-carbons.
    **2. Sequence Identity:** Evolutionary primary structure alignment via global pairing.
    **3. Distance Matrix Topology:** Internal Euclidian correlation, heavily resilient to structural loop mutations and insertions.
    **Consensus Equation:** `(0.33 × RMSD_Norm) + (0.33 × SeqID) + (0.34 × Distance_Matrix_Corr)`
    """)
