"""
ProCap Search Professional Diagnostic Suite v3.0
High-Throughput Structural Homology & Functional Annotation Pipeline
Production-Grade Bioinformatics Application (Dark Mode Edition)
Author: Utkarsh Patel
FIXED: Web database integration + error handling
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import json

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
# 2. SIMPLIFIED WEB DATABASE FUNCTIONS (FIXED!)
# ==================================================

@st.cache_data(ttl=3600)
def search_pdb_by_keyword(keyword: str, limit: int = 10) -> List[str]:
    """Search RCSB PDB for structures by keyword - SIMPLIFIED"""
    try:
        st.info(f"🔍 Searching PDB for: {keyword}")
        
        # Simple RCSB search
        url = "https://www.rcsb.org/search"
        params = {
            "q": keyword,
            "return_type": "entry"
        }
        
        # Use advanced API directly
        search_url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={{\"query\":{{\"type\":\"terminal\",\"service\":\"text\",\"parameters\":{{\"value\":\"{keyword}\"}}}},\"return_type\":\"entry\",\"num_result_set\":{limit}}}"
        
        response = requests.get(search_url, timeout=10)
        
        if response.status_code == 200:
            try:
                data = response.json()
                pdb_ids = []
                
                if "result_set" in data:
                    for result in data["result_set"][:limit]:
                        try:
                            pdb_id = result.get("identifier", "")
                            if pdb_id:
                                pdb_ids.append(pdb_id)
                        except:
                            continue
                
                return pdb_ids
            except:
                st.warning("Could not parse PDB response")
                return []
        else:
            st.warning(f"PDB search returned status: {response.status_code}")
            return []
    
    except Exception as e:
        st.error(f"❌ PDB Search Error: {str(e)}")
        return []

def download_pdb_by_id(pdb_id: str, out_path: Path) -> bool:
    """Download PDB structure from RCSB"""
    try:
        pdb_id = pdb_id.strip().upper()
        r = requests.get(
            f"https://files.rcsb.org/download/{pdb_id}.pdb",
            timeout=15
        )
        r.raise_for_status()
        out_path.write_text(r.text)
        return True
    except Exception as e:
        st.warning(f"⚠️ Download failed for {pdb_id}: {str(e)}")
        return False

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
            return False

    def get_available_chains(self) -> list:
        """Get all available chains"""
        if not self.structure:
            return []
        try:
            chains = [chain.id for chain in self.structure[0]]
            return sorted(chains)
        except:
            return []

    def get_coordinates(self, chain_id="A") -> Optional[np.ndarray]:
        """Extract alpha-carbon coordinates"""
        if not self.structure:
            return None
        try:
            coords = []
            chain = self.structure[0][chain_id]
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
            return np.array(coords, dtype=float) if coords else None
        except:
            return None

    def get_sequence(self, chain_id="A") -> Optional[str]:
        """Extract amino acid sequence"""
        if not self.structure:
            return None
        try:
            seq = []
            chain = self.structure[0][chain_id]
            for residue in chain:
                res_name = residue.get_resname()
                seq.append(THREE_TO_ONE.get(res_name, "X"))
            return "".join(seq) if seq else None
        except:
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
        except:
            pass
        return {
            "resolution": resolution,
            "mean_b_factor": np.mean(b_factors) if b_factors else np.nan
        }

# ==================================================
# 4. THREE SCORING METHODS
# ==================================================

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> Optional[float]:
    """METHOD 1: RMSD"""
    if coords1 is None or coords2 is None or coords1.shape != coords2.shape:
        return None
    try:
        distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
        return float(np.sqrt(np.mean(distances ** 2)))
    except:
        return None

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Helper: Distance matrix"""
    return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """METHOD 3: Distance Matrix Topology"""
    try:
        flat1 = dm1[np.triu_indices_from(dm1, k=1)]
        flat2 = dm2[np.triu_indices_from(dm2, k=1)]
        corr = np.corrcoef(flat1, flat2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except:
        return 0.0

def calculate_sequence_identity(seqA: str, seqB: str) -> Tuple[float, int]:
    """METHOD 2: Sequence Identity"""
    try:
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
    except:
        return 0.0, 0

def get_advanced_physchem_properties(seq: str) -> Dict[str, float]:
    """Calculate biochemical properties"""
    clean_seq = seq.replace("X", "").replace("U", "").replace("-", "")
    
    default_props = {
        "mw_kda": np.nan,
        "pi": np.nan,
        "instability": np.nan,
        "aromaticity": np.nan,
        "gravy": np.nan,
        "extinction_coeff": np.nan
    }
    
    if len(clean_seq) < 2:
        return default_props
    
    try:
        params = ProtParam.ProteinAnalysis(clean_seq)
        
        try:
            iso = IsoelectricPoint(clean_seq)
            pi = iso.pH()
        except:
            pi = params.isoelectric_point()
        
        return {
            "mw_kda": float(params.molecular_weight() / 1000),
            "pi": float(pi),
            "instability": float(params.instability_index()),
            "aromaticity": float(params.aromaticity()),
            "gravy": float(params.gravy()),
            "extinction_coeff": float(params.molar_extinction_coefficient()[0])
        }
    except:
        return default_props

def classify_confidence_level(score: float) -> Tuple[str, str]:
    """Classify confidence"""
    if score >= DIAGNOSTIC_THRESHOLDS["Identical"]:
        return "IDENTICAL MATCH", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Highly_Similar"]:
        return "HIGHLY SIMILAR", "alert-high"
    elif score >= DIAGNOSTIC_THRESHOLDS["Moderately_Similar"]:
        return "MODERATELY SIMILAR", "alert-medium"
    else:
        return "WEAKLY SIMILAR", "alert-low"

# ==================================================
# 5. DATA LOADING & DATABASE
# ==================================================
@st.cache_data
def load_reference_csv() -> pd.DataFrame:
    try:
        return pd.read_csv("data/known_capsule_proteins.csv")
    except:
        return pd.DataFrame(columns=[
            "PDB_ID", "Organism", "Gene_Name", "Function",
            "Protein_Family", "Structure_Type"
        ])

REFERENCE_DF = load_reference_csv()

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids: list):
    """Download local CSV structures"""
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
    """Execute 3-method pipeline"""
    
    qp = PDBWrapper(str(query_pdb))
    if not qp.parse():
        return pd.DataFrame(), {
            "status": "error",
            "message": "Failed to parse query PDB"
        }
    
    qcoords = qp.get_coordinates(chain_id)
    qseq = qp.get_sequence(chain_id)
    
    if qcoords is None or not qseq:
        available_chains = qp.get_available_chains()
        return pd.DataFrame(), {
            "status": "error",
            "message": f"No valid chain '{chain_id}'. Available: {', '.join(available_chains) if available_chains else 'NONE'}"
        }
    
    qdm = calculate_distance_matrix(qcoords)
    physchem_query = get_advanced_physchem_properties(qseq)
    
    results = []
    pdb_files = list(db_dir.glob("*.pdb"))
    
    if not pdb_files:
        return pd.DataFrame(), {
            "status": "error",
            "message": "No PDB files in database"
        }
    
    progress_bar = st.progress(0)
    
    for idx, pdb_file in enumerate(sorted(pdb_files)):
        target_id = pdb_file.stem.upper()
        if target_id == query_pdb.stem.upper():
            progress_bar.progress((idx + 1) / len(pdb_files))
            continue
        
        dp = PDBWrapper(str(pdb_file))
        if not dp.parse():
            progress_bar.progress((idx + 1) / len(pdb_files))
            continue
        
        dcoords = dp.get_coordinates(chain_id)
        dseq = dp.get_sequence(chain_id)
        
        if dcoords is None or not dseq:
            progress_bar.progress((idx + 1) / len(pdb_files))
            continue
        
        # METHOD 1: RMSD
        rmsd_val, r_score = np.nan, 0.0
        if len(qcoords) == len(dcoords):
            rmsd_val = calculate_rmsd(qcoords, dcoords)
            if rmsd_val is not None:
                r_score = max(0.0, 1.0 - (rmsd_val / 10.0))
        
        # METHOD 2: Sequence
        s_score, aligned_length = calculate_sequence_identity(qseq, dseq)
        
        # METHOD 3: Distance Matrix
        d_score = 0.0
        if qdm.shape == calculate_distance_matrix(dcoords).shape:
            d_score = (
                distance_matrix_correlation(qdm, calculate_distance_matrix(dcoords)) + 1.0
            ) / 2.0
        
        # Consensus
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
        
        progress_bar.progress((idx + 1) / len(pdb_files))
    
    df = pd.DataFrame(results)
    if not df.empty:
        ref_meta = REFERENCE_DF.copy()
        ref_meta["PDB_ID"] = ref_meta["PDB_ID"].str.upper()
        df = df.merge(ref_meta, left_on="Target_PDB", right_on="PDB_ID", how="left")
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
    """Predict function"""
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
# 6. VISUALIZATION ENGINES
# ==================================================
def generate_alignment_visualization(seqA: str, seqB: str, blocksize: int = 60) -> str:
    """Generate alignment"""
    try:
        seqA = str(seqA).strip()
        seqB = str(seqB).strip()
        
        if not seqA or not seqB:
            return "<p style='color:#ff3333;'>Empty sequence</p>"
        
        alns = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)
        if not alns:
            return "<p style='color:#ff3333;'>No alignment</p>"
        a = alns[0]
        
        html = '<div class="seq-align-box">\n'
        for i in range(0, len(a.seqA), blocksize):
            match_line = "".join([
                "|" if a.seqA[j] == a.seqB[j] and a.seqA[j] != "-" else
                "." if a.seqA[j] != "-" and a.seqB[j] != "-" else " "
                for j in range(i, min(i + blocksize, len(a.seqA)))
            ])
            html += f"<span style='color:#66b3ff;'>Query: </span>{a.seqA[i:i+blocksize]}\n"
            html += f"<span style='color:#00ff00;'>Match: </span>{match_line}\n"
            html += f"<span style='color:#ff3333;'>Target:</span>{a.seqB[i:i+blocksize]}\n\n"
        html += '</div>'
        return html
    except:
        return "<p style='color:#ff3333;'>Alignment error</p>"

def render_3d_structure(pdb_content: str):
    """Render 3D"""
    try:
        view = py3Dmol.view(width=800, height=450)
        view.addModel(pdb_content, 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.setBackgroundColor('#000000')
        view.zoomTo()
        return view
    except:
        return None

# ==================================================
# 7. MAIN APPLICATION UI
# ==================================================
st.title("🧬 ProCap Search Diagnostic Suite")
st.markdown("##### Production-Grade Protein Structure Analysis")
st.divider()

tmp_root = Path(tempfile.gettempdir()) / "procap_professional"
query_dir = tmp_root / "query"
query_dir.mkdir(parents=True, exist_ok=True)
db_dir = tmp_root / "db"
db_dir.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    
    chain_id = st.text_input("Chain ID", value="A", max_chars=1).strip().upper() or "A"
    
    st.divider()
    st.subheader("📥 Database Management")
    
    # Option 1: Local
    if st.button("✅ Initialize Local CSV Database", use_container_width=True):
        with st.spinner("📥 Loading local CSV structures..."):
            try:
                for f in db_dir.glob("*.pdb"):
                    f.unlink()
                pdb_list = REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist()
                if pdb_list:
                    d = build_database_from_reference_pdb_ids(db_dir, pdb_list)
                    st.success(f"✅ Loaded {len(d)} structures from CSV")
                else:
                    st.warning("⚠️ No PDB IDs in CSV")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Option 2: Web
    st.divider()
    st.subheader("🌐 Add from Web (RCSB PDB)")
    web_term = st.text_input("Search term", placeholder="e.g., 'VP4 rotavirus'")
    
    if st.button("🔍 Search & Download", use_container_width=True):
        if web_term:
            try:
                st.info(f"🔍 Searching RCSB for '{web_term}'...")
                pdb_ids = search_pdb_by_keyword(web_term, limit=5)
                
                if pdb_ids:
                    st.success(f"Found {len(pdb_ids)} structures: {', '.join(pdb_ids)}")
                    
                    with st.spinner(f"⏳ Downloading {len(pdb_ids)} structures..."):
                        downloaded = 0
                        for pid in pdb_ids:
                            out = db_dir / f"{pid}.pdb"
                            if download_pdb_by_id(pid, out):
                                downloaded += 1
                        
                        st.success(f"✅ Downloaded {downloaded}/{len(pdb_ids)} structures")
                else:
                    st.warning(f"❌ No results for '{web_term}'")
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
        else:
            st.warning("⚠️ Enter a search term")
    
    st.divider()
    db_count = len(list(db_dir.glob("*.pdb")))
    st.metric("📊 Database", f"{db_count} structures")

tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Results", "ℹ️ Help"])

# TAB 1: SEARCH
with tab1:
    col_in, col_out = st.columns([1, 2], gap="large")
    
    with col_in:
        st.subheader("Query Input")
        mode = st.radio("Source", ["PDB ID", "Upload File"])
        
        if mode == "PDB ID":
            pdb_id = st.text_input("PDB ID", value="1R3F", max_chars=4).strip().upper()
        else:
            uploaded = st.file_uploader("Upload .pdb", type=["pdb"])
            pdb_id = None
        
        execute = st.button("▶️ ANALYZE", type="primary", use_container_width=True)
    
    with col_out:
        if execute:
            db_count = len(list(db_dir.glob("*.pdb")))
            
            if db_count == 0:
                st.error("❌ Database empty. Use sidebar to initialize.")
            else:
                query_path = None
                
                if mode == "PDB ID" and len(pdb_id) == 4:
                    query_path = query_dir / f"{pdb_id}.pdb"
                    if not query_path.exists():
                        with st.spinner(f"⏳ Downloading {pdb_id}..."):
                            if not download_pdb_by_id(pdb_id, query_path):
                                query_path = None
                elif mode == "Upload File" and uploaded:
                    query_path = query_dir / uploaded.name
                    query_path.write_bytes(uploaded.getvalue())
                
                if not query_path:
                    st.error("❌ Failed to get structure")
                else:
                    st.info(f"🔬 Analyzing vs {db_count} structures...")
                    results_df, diag = run_comprehensive_diagnostic(query_path, db_dir, chain_id)
                    
                    if diag["status"] == "error":
                        st.error(f"❌ {diag['message']}")
                    elif results_df.empty:
                        st.warning("⚠️ No matches found")
                    else:
                        qp = PDBWrapper(str(query_path))
                        qp.parse()
                        exp_meta = qp.get_experimental_metadata(chain_id)
                        
                        st.session_state['query_pdb_content'] = query_path.read_text()
                        st.session_state['query_seq'] = qp.get_sequence(chain_id)
                        st.session_state['results_df'] = results_df
                        
                        top = results_df.iloc[0]
                        conf, alert = classify_confidence_level(top["Consensus_Score"])
                        
                        st.markdown(f'<div class="{alert}">✅ BEST MATCH: {conf} ({top["Consensus_Score"]:.3f})</div>', unsafe_allow_html=True)
                        
                        # 3 Methods
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🔧 RMSD", f"{top['RMSD_Score']:.3f}", f"{top['RMSD_Å']:.2f} Å")
                        c2.metric("🧬 SEQ", f"{top['Seq_Score']:.3f}", f"{top['Seq_Identity_%']:.1f}%")
                        c3.metric("📐 TOPO", f"{top['DM_Correlation']:.3f}", "Correlation")
                        
                        # PhysChem
                        st.markdown("### Query Properties")
                        p1, p2, p3, p4 = st.columns(4)
                        p1.metric("MW", f"{diag.get('mw_kda', 0):.1f} kDa")
                        p2.metric("pI", f"{diag.get('pi', 0):.2f}")
                        p3.metric("GRAVY", f"{diag.get('gravy', 0):.3f}")
                        p4.metric("Instability", f"{diag.get('instability', 0):.1f}")
                        
                        # Results Table
                        st.markdown(f"### Top Matches ({len(results_df)} total)")
                        cols_show = ["Target_PDB", "Organism", "Consensus_Score", "Seq_Identity_%", "RMSD_Å"]
                        st.dataframe(results_df[cols_show].head(10), use_container_width=True, hide_index=True)

# TAB 2: RESULTS
with tab2:
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        
        st.markdown("### Detailed Results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        if 'query_pdb_content' in st.session_state:
            st.markdown("### 3D Structure")
            view = render_3d_structure(st.session_state['query_pdb_content'])
            if view:
                components.html(view._make_html(), height=400)
    else:
        st.info("Run analysis first")

# TAB 3: HELP
with tab3:
    st.markdown("""
    ### Getting Started
    
    1. **Initialize Database**
       - Click "Initialize Local CSV Database" to load reference structures
    
    2. **Search from Web**
       - Enter search term (e.g., "VP4 rotavirus")
       - Click "Search & Download"
       - Downloads to local database
    
    3. **Analyze Query**
       - Enter PDB ID or upload file
       - Chain ID usually "A"
       - Click "ANALYZE"
    
    ### Test Examples
    - Query: **1R3F** (VP4 Rotavirus) → Expect >0.80 score
    - Query: **2FJQ** (CpsF) → Expect >0.75 score
    - Search: **"VP4 rotavirus"** → Get 5-10 results
    
    ### Three Methods
    - **RMSD**: Structural backbone alignment
    - **SEQ**: Sequence identity alignment
    - **TOPO**: Distance matrix topology
    
    All combined in **Consensus Score**
    """)
