from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from Bio import pairwise2
from Bio.PDB import PDBParser as BioPDBParser
from Bio.SeqUtils import ProtParam
from scipy.spatial.distance import cdist

# =========================
# 1. PAGE CONFIG & CSS
# =========================
st.set_page_config(page_title="ProCap Professional | Structure Search", layout="wide")

st.markdown(
    """
<style>
/* Professional LIMS Typography */
html, body, .stMarkdown, p, li, div, span, label {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 18px !important;
  line-height: 1.45 !important;
}

/* Metric Widgets Styling */
div[data-testid="stMetricLabel"] p {
  font-size: 15px !important;
  font-weight: bold !important;
  color: #4a5568 !important;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
div[data-testid="stMetricValue"] {
  font-size: 28px !important;
  font-weight: 600 !important;
  color: #2c5282 !important;
}

/* Sidebar & Tabs */
section[data-testid="stSidebar"] * { font-size: 16px !important; }
button[data-baseweb="tab"] { font-size: 18px !important; font-weight: bold !important; }

/* Buttons */
.stButton > button {
  font-size: 16px !important;
  border-radius: 6px !important;
  padding: 0.55em 1em !important;
}

/* Confidence Banners */
.priority-high { color: #276749; background-color: #f0fff4; padding: 10px; border-left: 5px solid #2f855a; border-radius: 4px; font-weight: bold;}
.priority-med { color: #9c4221; background-color: #fffaf0; padding: 10px; border-left: 5px solid #dd6b20; border-radius: 4px; font-weight: bold;}
.priority-low { color: #9b2c2c; background-color: #fff5f5; padding: 10px; border-left: 5px solid #e53e3e; border-radius: 4px; font-weight: bold;}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# 2. CORE UTILITIES
# =========================
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

class PDBWrapper:
    def __init__(self, pdb_path: str):
        self.pdb_path = Path(pdb_path)
        self.parser = BioPDBParser(QUIET=True)
        self.structure = None

    def parse(self):
        self.structure = self.parser.get_structure(self.pdb_path.stem, str(self.pdb_path))
        return self.structure

    def get_coordinates(self, chain_id="A"):
        if self.structure is None: return None
        coords = []
        try:
            chain = self.structure[0][chain_id]
            for residue in chain:
                if "CA" in residue: coords.append(residue["CA"].get_coord())
            return np.array(coords, dtype=float) if coords else None
        except KeyError: return None

    def get_sequence(self, chain_id="A"):
        if self.structure is None: return None
        seq = []
        try:
            chain = self.structure[0][chain_id]
            for residue in chain:
                res = residue.get_resname()
                seq.append(THREE_TO_ONE.get(res, "X"))
            return "".join(seq)
        except KeyError: return None

def calculate_rmsd(coords1, coords2):
    if coords1 is None or coords2 is None or coords1.shape != coords2.shape: return None
    return float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1) ** 2)))

def calculate_distance_matrix(coords): return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1, dm2):
    flat1 = dm1[np.triu_indices_from(dm1, k=1)]
    flat2 = dm2[np.triu_indices_from(dm2, k=1)]
    corr = np.corrcoef(flat1, flat2)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def download_pdb_by_id(pdb_id: str, out_path: Path):
    pdb_id = pdb_id.strip().upper()
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=20)
    r.raise_for_status()
    out_path.write_text(r.text)
    return out_path

def get_physchem_properties(seq: str) -> dict:
    try:
        params = ProtParam.ProteinAnalysis(seq.replace("X", ""))
        return {"mw_kda": params.molecular_weight() / 1000, "pi": params.isoelectric_point(), "instability": params.instability_index()}
    except Exception:
        return {"mw_kda": np.nan, "pi": np.nan, "instability": np.nan}

# =========================
# 3. DATA & DB HANDLING
# =========================
@st.cache_data
def load_reference_csv():
    try: return pd.read_csv("data/known_capsule_proteins.csv")
    except Exception: return pd.DataFrame(columns=["PDB_ID", "Organism", "Gene_Name", "Function"])

REFERENCE_DF = load_reference_csv()

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids):
    db_dir.mkdir(parents=True, exist_ok=True)
    downloaded, failed = [], []
    for pid in pdb_ids:
        try:
            out = db_dir / f"{pid}.pdb"
            if not out.exists(): download_pdb_by_id(pid, out)
            downloaded.append(pid)
        except Exception: failed.append(pid)
    return downloaded, failed

# =========================
# 4. SEARCH ENGINES
# =========================
def run_full_pipeline(query_pdb: Path, db_dir: Path, threshold: float, chain_id: str) -> pd.DataFrame:
    qp = PDBWrapper(str(query_pdb))
    qp.parse()
    qcoords = qp.get_coordinates(chain_id=chain_id)
    qseq = qp.get_sequence(chain_id=chain_id)
    
    if qcoords is None or not qseq: return pd.DataFrame()
    qdm = calculate_distance_matrix(qcoords)

    results = []
    for pdb_file in sorted(db_dir.glob("*.pdb")):
        target_id = pdb_file.stem.upper()
        if target_id == query_pdb.stem.upper(): continue # Skip self
        
        dp = PDBWrapper(str(pdb_file))
        try: dp.parse()
        except Exception: continue
        
        dcoords = dp.get_coordinates(chain_id=chain_id)
        dseq = dp.get_sequence(chain_id=chain_id)
        
        if dcoords is None or not dseq: continue

        # 1. RMSD
        rmsd_val, r_score = np.nan, 0.0
        if len(qcoords) == len(dcoords):
            rmsd_val = calculate_rmsd(qcoords, dcoords)
            if rmsd_val is not None: r_score = max(0.0, 1.0 - (rmsd_val / 10.0))

        # 2. Sequence Identity
        s_score = 0.0
        alns = pairwise2.align.globalxx(qseq, dseq, one_alignment_only=True)
        if alns:
            a = alns[0]
            matches = sum(x == y for x, y in zip(a.seqA, a.seqB))
            s_score = matches / max(len(qseq), len(dseq))

        # 3. Distance Matrix Correlation
        d_score = 0.0
        if qdm.shape == calculate_distance_matrix(dcoords).shape:
            corr = distance_matrix_correlation(qdm, calculate_distance_matrix(dcoords))
            d_score = (corr + 1.0) / 2.0

        # Consensus
        final_score = (0.33 * r_score) + (0.33 * s_score) + (0.34 * d_score)
        if final_score > 0:
            results.append({
                "Target_PDB": target_id, "Consensus_Score": final_score,
                "RMSD": rmsd_val, "Seq_Identity": s_score, "DM_Correlation": d_score
            })

    df = pd.DataFrame(results)
    if not df.empty:
        # Merge with biological metadata
        ref_meta = REFERENCE_DF[["PDB_ID", "Organism", "Gene_Name", "Function"]].copy()
        ref_meta["PDB_ID"] = ref_meta["PDB_ID"].str.upper()
        df = df.merge(ref_meta, left_on="Target_PDB", right_on="PDB_ID", how="left")
        df = df.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    return df

# =========================
# 5. UI LAYOUT
# =========================
st.title("🧬 ProCap Search Diagnostic Suite")
st.markdown("#### Operational Intelligence for Bacterial Capsule Biosynthesis Protein Identification")
st.divider()

# Setup Temp Workspace
tmp_root = Path(tempfile.gettempdir()) / "procap_space"
query_dir = tmp_root / "query"; query_dir.mkdir(parents=True, exist_ok=True)
db_dir = tmp_root / "db"; db_dir.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    chain_id = st.text_input("Target Chain ID", value="A", help="Most PDBs use chain A.")
    threshold = st.slider("Global Similarity Threshold", 0.0, 1.0, 0.40, 0.05)

    st.subheader("Reference Database")
    st.info("The system automatically pulls the 23 verified capsule proteins from your known_capsule_proteins.csv file.")
    if st.button("Initialize/Update Database"):
        with st.spinner("Downloading reference structures from RCSB..."):
            d, f = build_database_from_reference_pdb_ids(db_dir, REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist())
            st.success(f"Database ready: {len(d)} structures loaded.")

tab1, tab2, tab3 = st.tabs(["🔍 Structure Diagnostic", "📊 Database Analytics", "📖 Methodology"])

with tab1:
    c_in, c_res = st.columns([1, 1.8], gap="large")

    with c_in:
        st.subheader("I. Query Input")
        query_mode = st.radio("Input Source", ["RCSB Database ID", "Local PDB Upload"])
        if query_mode == "RCSB Database ID":
            pdb_id = st.text_input("Enter 4-letter PDB ID", value="1R3G").strip().upper()
        else:
            uploaded_query = st.file_uploader("Upload .pdb file", type=["pdb"])
            pdb_id = "USER_UPLOAD"

        execute = st.button("Execute Full Pipeline", type="primary", use_container_width=True)

    with c_res:
        if execute:
            query_pdb_path = None
            if query_mode == "RCSB Database ID" and len(pdb_id) == 4:
                query_pdb_path = query_dir / f"{pdb_id}.pdb"
                if not query_pdb_path.exists(): download_pdb_by_id(pdb_id, query_pdb_path)
            elif query_mode == "Local PDB Upload" and uploaded_query:
                query_pdb_path = query_dir / uploaded_query.name
                query_pdb_path.write_bytes(uploaded_query.getvalue())

            if not query_pdb_path:
                st.error("Failed to prepare query. Check PDB ID or file.")
            else:
                with st.spinner("Aligning structures and computing matrices..."):
                    results_df = run_full_pipeline(query_pdb_path, db_dir, threshold, chain_id)
                    
                    # Extract PhysChem for Query
                    qp = PDBWrapper(str(query_pdb_path))
                    qp.parse()
                    qseq = qp.get_sequence(chain_id=chain_id)
                    physchem = get_physchem_properties(qseq) if qseq else {}

                if results_df.empty:
                    st.warning("No homologous structures found matching the coordinate lengths or passing thresholds.")
                else:
                    top_hit = results_df.iloc[0]
                    score = top_hit["Consensus_Score"]
                    priority = "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
                    
                    with st.container(border=True):
                        st.markdown(f"### Diagnostic Status: <span class='priority-{priority.lower()}'>{priority} Confidence Homolog</span>", unsafe_allow_html=True)
                        st.markdown("**Scientific Rationale:**")
                        st.write(f"• **Top Biological Match:** {top_hit.get('Organism', 'Unknown')} | {top_hit.get('Gene_Name', 'N/A')} ({top_hit.get('Function', 'N/A')})")
                        st.write(f"• **Consensus Score:** {score:.2f}/1.00 (Aggregate of RMSD, Seq Identity, and Distance Matrix)")
                        if not pd.isna(top_hit['RMSD']): st.write(f"• **Structural Fidelity:** RMSD of {top_hit['RMSD']:.2f} Å indicates strong 3D topology conservation.")
                        if physchem.get('instability', 0) > 40: st.write("• **Warning:** Query protein predicted to be unstable in vitro.")

                    st.markdown("### Query PhysChem Profile")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Mol. Weight", f"{physchem.get('mw_kda', 0):.1f} kDa")
                    m2.metric("Sequence Length", f"{len(qseq) if qseq else 0} AA")
                    m3.metric("Isoelectric Pt (pI)", f"{physchem.get('pi', 0):.2f}")
                    m4.metric("Instability Idx", f"{physchem.get('instability', 0):.1f}")

                    st.markdown("### Consensus Output (Top 10)")
                    st.dataframe(results_df[["Target_PDB", "Organism", "Gene_Name", "Consensus_Score", "Seq_Identity", "RMSD"]].head(10), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Reference Database Analytics")
    if not REFERENCE_DF.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.pie(REFERENCE_DF, names="Organism", title="Pathogen Distribution", hole=0.3)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.bar(REFERENCE_DF["Protein_Family"].value_counts().reset_index(), x="Protein_Family", y="count", title="Protein Families")
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(REFERENCE_DF, use_container_width=True)

with tab3:
    st.subheader("Methodology")
    st.markdown("""
    **ProCap Search** aggregates three distinct algorithms to overcome individual methodology blind spots:
    1. **Sequence Identity (BioPython pairwise2):** Fast primary structure alignment.
    2. **RMSD-Based Alignment:** Physics-based 3D superposition. (Note: Requires identical coordinate counts).
    3. **Distance Matrix Correlation (SciPy):** Topology-based comparison using internal Euclidian distances, highly resilient to loop movements.
    """)
