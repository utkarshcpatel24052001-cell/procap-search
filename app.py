"""
ProCap Search Professional Diagnostic Suite v3.0
High-Throughput Structural Homology & Functional Annotation Pipeline
Production-Grade Bioinformatics Application (Dark Mode Edition)
Author: Utkarsh Patel
"""

from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List

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
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
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
    
    /* Highlighted Method Box */
    .method-box { 
        background-color: #111111; 
        border: 1px solid #333333; 
        border-left: 6px solid #ff3333; 
        padding: 16px; 
        border-radius: 6px; 
        color: #ffffff; 
        font-weight: 500; 
        margin-bottom: 15px;
    }

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
    
    /* Container Borders */
    .stContainer {
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 2. WEB DATABASE FUNCTIONS
# ==================================================

def search_pdb_rcsb(keyword: str, limit: int = 50) -> List[Dict]:
    """Search RCSB PDB database"""
    try:
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        query = {
            "query": {"type": "terminal", "service": "text", "parameters": {"value": keyword}},
            "return_type": "entry", "num_result_set": limit
        }
        response = requests.post(url, json=query, headers={"Content-Type": "application/json"}, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return [{"PDB_ID": res["identifier"], "Source": "RCSB PDB"} for res in data.get("result_set", [])[:limit]]
        st.warning(f"RCSB search failed: {response.status_code}")
        return []
    except Exception as e:
        st.error(f"RCSB search error: {str(e)}")
        return []

def search_uniprot_for_pdb(protein_name: str, organism: str = "") -> List[Dict]:
    """Search UniProt for PDB structures"""
    try:
        query = f"{protein_name}" + (f" AND organism:{organism}" if organism else "")
        params = {"query": query, "format": "json", "size": 50, "fields": "accession,protein_name,xref_pdb,organism_name"}
        response = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params, timeout=15)
        if response.status_code == 200:
            results = []
            for entry in response.json().get("results", []):
                for xref in entry.get("uniProtKBCrossReferences", []):
                    if xref.get("database") == "PDB":
                        results.append({
                            "PDB_ID": xref.get("id"),
                            "Protein_Name": entry.get("proteins", [{}])[0].get("recommendedName", {}).get("fullName", {}).get("value", "Unknown"),
                            "Organism": entry.get("organism", {}).get("scientificName", "Unknown"),
                            "Source": "UniProt"
                        })
            return results
        st.warning(f"UniProt search failed: {response.status_code}")
        return []
    except Exception as e:
        st.error(f"UniProt search error: {str(e)}")
        return []

# ==================================================
# 3. CORE BIOPYTHON & MATH UTILITIES
# ==================================================
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

DIAGNOSTIC_THRESHOLDS = {"Identical": 0.90, "Highly_Similar": 0.75, "Moderately_Similar": 0.55}

class PDBWrapper:
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
        except Exception: 
            return False

    def get_available_chains(self) -> list:
        if not self.structure: 
            return []
        try: 
            return sorted([chain.id for chain in self.structure[0]])
        except Exception: 
            return []

    def get_best_chain(self, requested_chain: str) -> Optional[str]:
        chains = self.get_available_chains()
        if requested_chain in chains: 
            return requested_chain
        if chains: 
            return chains[0]
        return None

    def get_coordinates(self, chain_id="A") -> Optional[np.ndarray]:
        if not self.structure or not chain_id: 
            return None
        try:
            coords = [res["CA"].get_coord() for res in self.structure[0][chain_id] if "CA" in res]
            return np.array(coords, dtype=float) if coords else None
        except Exception: 
            return None

    def get_sequence(self, chain_id="A") -> Optional[str]:
        if not self.structure or not chain_id: 
            return None
        try:
            seq = []
            for residue in self.structure[0][chain_id]:
                if residue.id[0] != " ": 
                    continue
                seq.append(THREE_TO_ONE.get(residue.get_resname(), "X"))
            return "".join(seq) if seq else None
        except Exception: 
            return None
        
    def get_experimental_metadata(self, chain_id="A"):
        resolution = self.header.get("resolution", np.nan)
        b_factors = []
        try:
            if self.structure and chain_id:
                b_factors = [res["CA"].get_bfactor() for res in self.structure[0][chain_id] if "CA" in res]
        except Exception: 
            pass
        return {"resolution": resolution, "mean_b_factor": np.mean(b_factors) if b_factors else np.nan}

# ==================================================
# 4. SCORING ALGORITHMS
# ==================================================

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> Optional[float]:
    if coords1 is None or coords2 is None or coords1.shape != coords2.shape: 
        return None
    try: 
        return float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1))))
    except Exception: 
        return None

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1: np.ndarray, dm2: np.ndarray) -> float:
    try:
        flat1, flat2 = dm1[np.triu_indices_from(dm1, k=1)], dm2[np.triu_indices_from(dm2, k=1)]
        corr = np.corrcoef(flat1, flat2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception: 
        return 0.0

def calculate_sequence_identity(seqA: str, seqB: str) -> Tuple[float, int]:
    try:
        if not seqA or not seqB or str(seqA) == "None" or str(seqB) == "None": 
            return 0.0, 0
        alns = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)
        if not alns: 
            return 0.0, 0
        a = alns[0]
        matches = sum(x == y for x, y in zip(a.seqA, a.seqB))
        return float(matches / max(len(seqA), len(seqB))), len(a.seqA)
    except Exception: 
        return 0.0, 0

def download_pdb_by_id(pdb_id: str, out_path: Path) -> bool:
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id.strip().upper()}.pdb", timeout=15)
        r.raise_for_status()
        out_path.write_text(r.text)
        return True
    except Exception: 
        return False

def get_advanced_physchem_properties(seq: str) -> Dict[str, float]:
    default_props = {"mw_kda": np.nan, "pi": np.nan, "instability": np.nan, "aromaticity": np.nan, "gravy": np.nan, "extinction_coeff": np.nan}
    if not seq or str(seq) == "None": 
        return default_props
    
    clean_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq.upper())
    if len(clean_seq) < 2: 
        return default_props
    
    try:
        params = ProtParam.ProteinAnalysis(clean_seq)
        pi = np.nan
        try: 
            pi = IsoelectricPoint(clean_seq).pH()
        except Exception: 
            pi = params.isoelectric_point()
        
        return {
            "mw_kda": float(params.molecular_weight() / 1000), 
            "pi": float(pi), 
            "instability": float(params.instability_index()),
            "aromaticity": float(params.aromaticity()), 
            "gravy": float(params.gravy()), 
            "extinction_coeff": float(params.molar_extinction_coefficient()[0])
        }
    except Exception: 
        return default_props

def classify_confidence_level(score: float) -> Tuple[str, str]:
    if score >= DIAGNOSTIC_THRESHOLDS["Identical"]: 
        return "IDENTICAL MATCH", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Highly_Similar"]: 
        return "HIGHLY SIMILAR", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Moderately_Similar"]: 
        return "MODERATELY SIMILAR", "alert-medium"
    else: 
        return "WEAKLY SIMILAR", "alert-low"

# ==================================================
# 5. DATABASE PIPELINE
# ==================================================
@st.cache_data
def load_reference_csv() -> pd.DataFrame:
    try: 
        return pd.read_csv("data/known_capsule_proteins.csv")
    except FileNotFoundError: 
        return pd.DataFrame(columns=["PDB_ID", "Organism", "Gene_Name", "Function", "Protein_Family", "Structure_Type"])

REFERENCE_DF = load_reference_csv()

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids: list):
    db_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    progress_bar = st.progress(0)
    for i, pid in enumerate(pdb_ids):
        out = db_dir / f"{pid}.pdb"
        if out.exists() or download_pdb_by_id(pid, out): 
            downloaded.append(pid)
        progress_bar.progress((i + 1) / len(pdb_ids))
    return downloaded

def run_comprehensive_diagnostic(query_pdb: Path, db_dir: Path, chain_id: str) -> Tuple[pd.DataFrame, Dict]:
    qp = PDBWrapper(str(query_pdb))
    if not qp.parse(): 
        return pd.DataFrame(), {"status": "error", "message": "Failed to parse query PDB"}
    
    q_chain = qp.get_best_chain(chain_id)
    if not q_chain: 
        return pd.DataFrame(), {"status": "error", "message": f"No valid chains found in query PDB."}
    
    qcoords, qseq = qp.get_coordinates(q_chain), qp.get_sequence(q_chain)
    qdm = calculate_distance_matrix(qcoords) if qcoords is not None else None
    physchem_query = get_advanced_physchem_properties(qseq)
    
    chain_warning = f"Requested Chain '{chain_id}' missing. Auto-resolved to Chain '{q_chain}'." if q_chain != chain_id else None

    results, total_files = [], len(list(db_dir.glob("*.pdb")))
    progress_bar = st.progress(0)
    
    for idx, pdb_file in enumerate(sorted(db_dir.glob("*.pdb"))):
        progress_bar.progress((idx + 1) / total_files)
        target_id = pdb_file.stem.upper()
        if target_id == query_pdb.stem.upper(): 
            continue
        
        dp = PDBWrapper(str(pdb_file))
        if not dp.parse(): 
            continue
        
        d_chain = dp.get_best_chain(q_chain)
        if not d_chain: 
            continue
        
        dcoords, dseq = dp.get_coordinates(d_chain), dp.get_sequence(d_chain)
        if dcoords is None or not dseq: 
            continue
        
        rmsd_val, r_score = np.nan, 0.0
        if qcoords is not None and len(qcoords) == len(dcoords):
            rmsd_val = calculate_rmsd(qcoords, dcoords)
            if rmsd_val is not None: 
                r_score = max(0.0, 1.0 - (rmsd_val / 10.0))
        
        s_score, aligned_length = calculate_sequence_identity(qseq, dseq)
        
        d_score = 0.0
        if qdm is not None and dcoords is not None:
            ddm = calculate_distance_matrix(dcoords)
            if qdm.shape == ddm.shape:
                d_score = (distance_matrix_correlation(qdm, ddm) + 1.0) / 2.0
        
        final_score = (0.33 * r_score) + (0.33 * s_score) + (0.34 * d_score)
        confidence, _ = classify_confidence_level(final_score)
        
        if final_score > 0:
            results.append({
                "Target_PDB": target_id, "Target_Chain": d_chain, "Consensus_Score": final_score, "Confidence": confidence,
                "RMSD_Å": rmsd_val, "RMSD_Score": r_score, "Seq_Identity_%": s_score * 100, "Seq_Score": s_score,
                "DM_Correlation": d_score, "Query_Coverage_%": (aligned_length / len(qseq) * 100) if qseq else 0, "Target_Seq": dseq,
            })
    
    df = pd.DataFrame(results)
    if not df.empty:
        ref_meta = REFERENCE_DF.copy()
        ref_meta["PDB_ID"] = ref_meta["PDB_ID"].str.upper()
        df = df.merge(ref_meta, left_on="Target_PDB", right_on="PDB_ID", how="left").drop("PDB_ID", axis=1, errors="ignore")
        df["Organism"] = df["Organism"].fillna("Web Database Search")
        df = df.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    
    return df, {"status": "success", "resolved_chain": q_chain, "chain_warning": chain_warning, "query_seq_length": len(qseq) if qseq else 0, "total_matches": len(results), **physchem_query}

def predict_biological_function(results_df: pd.DataFrame) -> dict:
    top_3 = results_df.head(3).dropna(subset=['Protein_Family', 'Structure_Type', 'Function'])
    if top_3.empty: 
        return {"family": "Unknown", "structure": "Unknown", "function": "Unknown", "confidence": "Low"}
    return {"family": top_3['Protein_Family'].mode()[0], "structure": top_3['Structure_Type'].mode()[0], "function": top_3['Function'].mode()[0], "confidence": "High" if top_3['Protein_Family'].nunique() == 1 else "Moderate"}

# ==================================================
# 6. VISUALIZATION ENGINES
# ==================================================
def generate_alignment_visualization(seqA: str, seqB: str, blocksize: int = 60) -> str:
    if not seqA or not seqB or str(seqA) == "None" or str(seqB) == "None":
        return "<p style='color:#ff3333;'>Alignment error: One or both sequences are unavailable.</p>"
    try:
        alns = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)
        if not alns: 
            return "<p style='color:#ff3333;'>No alignment found</p>"
        a = alns[0]
        
        html = '<div class="seq-align-box">\n'
        for i in range(0, len(a.seqA), blocksize):
            match_line = "".join(["|" if a.seqA[j] == a.seqB[j] and a.seqA[j] != "-" else "." if a.seqA[j] != "-" and a.seqB[j] != "-" else " " for j in range(i, min(i + blocksize, len(a.seqA)))])
            html += f"<span style='color:#66b3ff; font-weight:bold;'>Query: </span> {a.seqA[i:i+blocksize]}\n"
            html += f"<span style='color:#00ff00; font-weight:bold;'>Match: </span> {match_line}\n"
            html += f"<span style='color:#ff3333; font-weight:bold;'>Target:</span> {a.seqB[i:i+blocksize]}\n\n"
        html += '</div>'
        return html
    except Exception as e: 
        return f"<p style='color:#ff3333;'>Error: {str(e)}</p>"

def render_3d_structure(pdb_content: str):
    try:
        view = py3Dmol.view(width=800, height=450)
        view.addModel(pdb_content, 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.setBackgroundColor('#000000')
        view.zoomTo()
        return view
    except Exception: 
        return None

# ==================================================
# 7. APPLICATION UI
# ==================================================
st.title("🧬 ProCap Search Diagnostic Suite")
st.markdown("##### High-Throughput Structural Homology & Functional Annotation Pipeline v3.0")
st.markdown("##### **All THREE Methods: RMSD + Sequence Identity + Distance Matrix**")
st.divider()

tmp_root = Path(tempfile.gettempdir()) / "procap_professional"
query_dir = tmp_root / "query"; query_dir.mkdir(parents=True, exist_ok=True)
db_dir = tmp_root / "db"; db_dir.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    st.markdown('<div class="info-box"><b>🔗 Chain Selection:</b><br>Select the chain to analyze.</div>', unsafe_allow_html=True)
    chain_id = st.text_input("Target Chain ID", value="A", max_chars=1, help="Chain ID (A, B, C, etc.)").strip().upper() or "A"
    
    st.divider()
    st.markdown("### 🗄️ Database Management")
    st.subheader("Option 1: Local CSV Database")
    if st.button("📥 Sync Local CSV Database", use_container_width=True):
        with st.spinner("Syncing local CSV structures..."):
            for f in db_dir.glob("*.pdb"): 
                f.unlink()
            d = build_database_from_reference_pdb_ids(db_dir, REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist())
            st.markdown(f'<div class="success-box">✅ Local DB: {len(d)} structures</div>', unsafe_allow_html=True)
    
    st.divider()
    st.subheader("Option 2: Expand from WEB")
    web_search_type = st.selectbox("🌐 Search Type", ["Keyword Search", "Protein Name (UniProt)"])
    web_search_term = st.text_input("Search Term", placeholder="e.g., 'capsule protein'")
    
    if st.button("🔍 Search & Download", use_container_width=True):
        if web_search_term:
            with st.spinner(f"Searching {web_search_type}..."):
                results = search_pdb_rcsb(web_search_term, 20) if web_search_type == "Keyword Search" else search_uniprot_for_pdb(web_search_term, "")
                if results:
                    st.session_state['web_results'] = results
                else: 
                    st.warning("⚠️ No results found")
        else: 
            st.warning("⚠️ Enter a search term")
        
    if 'web_results' in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state['web_results']), use_container_width=True, hide_index=True)
        if st.button("⬇️ Download All to Local Database", use_container_width=True):
            with st.spinner("Downloading..."):
                pdb_ids = [r["PDB_ID"] for r in st.session_state['web_results']]
                added = 0
                for pid in pdb_ids:
                    out = db_dir / f"{pid}.pdb"
                    if not out.exists() and download_pdb_by_id(pid, out): 
                        added += 1
                st.success(f"✅ Added {added} structures to database")
    
    st.divider()
    db_count = len(list(db_dir.glob("*.pdb")))
    st.metric("Total Database Structures Ready", db_count)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Structural Diagnostic", "🧪 3D Viewer & Alignments", "📊 Database Analytics", "📖 Algorithm Details", "🧬 Scoring Breakdown"])

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
        execute_btn = st.button("▶ EXECUTE ALL 3 METHODS", type="primary", use_container_width=True)
    
    with col_results:
        if execute_btn:
            if db_count == 0:
                st.markdown('<div class="alert-low">❌ Database is empty! Initialize from sidebar.</div>', unsafe_allow_html=True)
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
                    with st.spinner(f"🔬 Comparing against {db_count} database structures..."):
                        results_df, diagnostics = run_comprehensive_diagnostic(query_pdb_path, db_dir, chain_id)
                        
                        qp = PDBWrapper(str(query_pdb_path))
                        qp.parse()
                        exp_meta = qp.get_experimental_metadata(diagnostics.get('resolved_chain', 'A'))
                    
                    if diagnostics["status"] == "error":
                        st.markdown(f'<div class="alert-low">❌ ERROR: {diagnostics["message"]}</div>', unsafe_allow_html=True)
                    elif results_df.empty:
                        st.markdown('<div class="alert-low">⚠️ No homologous structures found in database.</div>', unsafe_allow_html=True)
                    else:
                        st.session_state['query_pdb_content'] = query_pdb_path.read_text()
                        st.session_state['query_seq'] = qp.get_sequence(diagnostics.get('resolved_chain', 'A'))
                        st.session_state['results_df'] = results_df
                        st.session_state['resolved_chain'] = diagnostics.get('resolved_chain', 'A')
                        
                        top_hit = results_df.iloc[0]
                        confidence, alert_class = classify_confidence_level(top_hit["Consensus_Score"])
                        prediction = predict_biological_function(results_df)
                        
                        if diagnostics.get("chain_warning"): 
                            st.warning(f"⚠️ {diagnostics['chain_warning']}")
                        
                        st.markdown(f'<div class="{alert_class}">🧬 CONSENSUS SCORE: {top_hit["Consensus_Score"]:.3f}/1.000 | {confidence}</div>', unsafe_allow_html=True)
                        
                        st.markdown("### 📊 Individual Method Scores (Top Hit)")
                        method_col1, method_col2, method_col3 = st.columns(3)
                        with method_col1: 
                            st.markdown(f'<div class="method-box"><b>METHOD 1: RMSD</b><br>Score: {top_hit["RMSD_Score"]:.3f}<br>Value: {top_hit["RMSD_Å"]:.2f} Å</div>', unsafe_allow_html=True)
                        with method_col2: 
                            st.markdown(f'<div class="method-box"><b>METHOD 2: SEQUENCE</b><br>Score: {top_hit["Seq_Score"]:.3f}<br>Identity: {top_hit["Seq_Identity_%"]:.1f}%</div>', unsafe_allow_html=True)
                        with method_col3: 
                            st.markdown(f'<div class="method-box"><b>METHOD 3: TOPOLOGY</b><br>Score: {top_hit["DM_Correlation"]:.3f}<br>Corr: {top_hit["DM_Correlation"]:.3f}</div>', unsafe_allow_html=True)
                        
                        with st.container(border=True):
                            st.markdown("### Biological Annotation")
                            st.write(f"This query functions as a **{prediction['function']}**, belonging to **{prediction['family']}** family ({prediction['structure']}).")
                            st.markdown("**Top Reference Hit**")
                            col_top1, col_top2, col_top3 = st.columns(3)
                            col_top1.write(f"**PDB ID:** `{top_hit['Target_PDB']}` (Chain {top_hit['Target_Chain']})\n\n**Organism:** {top_hit.get('Organism', 'Unknown')}")
                            col_top2.write(f"**Gene:** {top_hit.get('Gene_Name', 'Unknown')}\n\n**Coverage:** {top_hit['Query_Coverage_%']:.1f}%")
                            col_top3.write(f"**Function:** {top_hit.get('Function', 'Unknown')}\n\n**Family:** {top_hit.get('Protein_Family', 'Unknown')}")
                        
                        with st.container(border=True):
                            st.markdown(f"### Structural & PhysChem Profile (Query Chain {diagnostics.get('resolved_chain', 'A')})")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Length", f"{diagnostics.get('query_seq_length', 0)} AA")
                            m2.metric("Mol. Weight", f"{diagnostics.get('mw_kda', 0):.1f} kDa")
                            m3.metric("Isoelectric Pt", f"{diagnostics.get('pi', 0):.2f}")
                            m4.metric("Instability Idx", f"{diagnostics.get('instability', 0):.1f}")
                            
                            m5, m6, m7, m8 = st.columns(4)
                            m5.metric("GRAVY", f"{diagnostics.get('gravy', 0):.3f}")
                            m6.metric("Aromaticity", f"{diagnostics.get('aromaticity', 0):.3f}")
                            m7.metric("Extinction Coeff.", f"{diagnostics.get('extinction_coeff', 0):.0f} M⁻¹cm⁻¹")
                            m8.metric("X-Ray Resolution", f"{exp_meta.get('resolution', np.nan):.2f} Å" if not pd.isna(exp_meta.get('resolution', np.nan)) else "N/A")
                        
                        st.markdown("### Search Results Matrix")
                        st.markdown(f"**Found {len(results_df)} structurally viable matching structures in database.**")
                        display_cols = ["Target_PDB", "Organism", "Consensus_Score", "Confidence", "RMSD_Score", "Seq_Score", "DM_Correlation", "Seq_Identity_%", "RMSD_Å"]
                        st.dataframe(results_df[display_cols].head(15), use_container_width=True, hide_index=True)

# ==================================================
# TAB 2: 3D VIEWER & ALIGNMENTS
# ==================================================
with tab2:
    st.subheader("3D Structure Visualization & Sequence Interrogation")
    if 'query_pdb_content' in st.session_state:
        col3d, colseq = st.columns([1.3, 1], gap="large")
        with col3d:
            st.markdown(f"### Interactive 3D Query Topology (Chain {st.session_state.get('resolved_chain', 'A')})")
            with st.container(border=True):
                view = render_3d_structure(st.session_state['query_pdb_content'])
                if view: 
                    components.html(view._make_html(), height=480)
        with colseq:
            if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
                top = st.session_state['results_df'].iloc[0]
                st.markdown(f"### Alignment vs {top['Target_PDB']} (Chain {top['Target_Chain']})")
                qseq = st.session_state.get('query_seq')
                tseq = top.get('Target_Seq')
                st.markdown(generate_alignment_visualization(qseq, tseq, blocksize=40), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                fasta_str = f">Query_Chain_{st.session_state.get('resolved_chain', 'A')}\n{qseq}\n>Target_{top['Target_PDB']}_Chain_{top['Target_Chain']}\n{tseq}\n"
                st.download_button("📥 Export FASTA Alignment", data=fasta_str, file_name=f"alignment_{top['Target_PDB']}.fasta", mime="text/plain", use_container_width=True)
    else:
        st.markdown('<div class="info-box">💡 Execute diagnostic in Tab 1 to generate 3D models.</div>', unsafe_allow_html=True)

# ==================================================
# TAB 3: DATABASE ANALYTICS
# ==================================================
with tab3:
    st.subheader("Database Population Analytics")
    if not REFERENCE_DF.empty:
        c1, c2 = st.columns(2)
        with c1: 
            st.plotly_chart(px.pie(REFERENCE_DF, names="Organism", title="Pathogen Representation", hole=0.3, template="plotly_dark"), use_container_width=True)
        with c2: 
            st.plotly_chart(px.bar(REFERENCE_DF["Protein_Family"].value_counts().reset_index(), x="Protein_Family", y="count", title="Domain Families", template="plotly_dark"), use_container_width=True)
        st.dataframe(REFERENCE_DF, use_container_width=True, hide_index=True)

# ==================================================
# TAB 4: ALGORITHM DETAILS
# ==================================================
with tab4:
    st.subheader("THREE-METHOD Consensus Algorithm")
    st.markdown("""
    ### Method 1: RMSD-Based Structural Superposition
    - Root Mean Square Deviation of alpha-carbon backbone atoms
    - **Formula:** `Score = max(0, 1 - RMSD/10)`
    
    ### Method 2: Sequence Identity (Needleman-Wunsch)
    - Evolutionary primary structure conservation
    - **Formula:** `Score = Matching_Residues / max(Seq_A, Seq_B)`
    
    ### Method 3: Distance Matrix Topology Correlation
    - Pearson correlation of pairwise Euclidean distances
    - **Formula:** `Score = (Correlation + 1) / 2`
    
    ### Consensus Scoring:
    ```text
    Consensus = (0.33 × RMSD_Score) + (0.33 × Seq_Score) + (0.34 × DM_Score)
    ```
    """)

# ==================================================
# TAB 5: SCORING BREAKDOWN
# ==================================================
with tab5:
    st.subheader("Individual Method Scores Breakdown")
    if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
        results = st.session_state['results_df'].head(10).copy()
        fig = px.bar(
            results, x="Target_PDB", y=["RMSD_Score", "Seq_Score", "DM_Correlation"],
            title="Individual Method Contributions", template="plotly_dark", barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(results[["Target_PDB", "Organism", "Consensus_Score", "RMSD_Score", "Seq_Score", "DM_Correlation", "RMSD_Å", "Seq_Identity_%"]], use_container_width=True, hide_index=True)
    else:
        st.info("Run diagnostic first")
