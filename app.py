import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from Bio import pairwise2
from Bio.PDB import PDBParser as BioPDBParser
from scipy.spatial.distance import cdist

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="ProCap Search | Protein Capsule Structure Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================
# Constants
# ==========================
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# ==========================
# PDB Parser Wrapper
# ==========================
class PDBWrapper:
    def __init__(self, pdb_path: str):
        self.pdb_path = Path(pdb_path)
        self.parser = BioPDBParser(QUIET=True)
        self.structure = None

    def parse(self):
        try:
            self.structure = self.parser.get_structure(self.pdb_path.stem, str(self.pdb_path))
            return self.structure
        except Exception:
            return None

    def get_coordinates(self, chain_id="A"):
        if self.structure is None:
            return None
        try:
            coords = []
            model = self.structure[0]
            chain = model[chain_id]
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
            if not coords:
                return None
            return np.array(coords, dtype=float)
        except Exception:
            return None

    def get_sequence(self, chain_id="A"):
        if self.structure is None:
            return None
        try:
            seq = []
            model = self.structure[0]
            chain = model[chain_id]
            for residue in chain:
                res = residue.get_resname()
                seq.append(THREE_TO_ONE.get(res, "X"))
            return "".join(seq)
        except Exception:
            return None

# ==========================
# Calculation Functions
# ==========================
def calculate_rmsd(coords1, coords2):
    if coords1 is None or coords2 is None:
        return None
    if coords1.shape != coords2.shape:
        return None
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
    return float(np.sqrt(np.mean(distances ** 2)))

def calculate_distance_matrix(coords):
    return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1, dm2):
    flat1 = dm1[np.triu_indices_from(dm1, k=1)]
    flat2 = dm2[np.triu_indices_from(dm2, k=1)]
    try:
        corr = np.corrcoef(flat1, flat2)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0

def download_pdb_by_id(pdb_id: str, out_path: Path):
    pdb_id = pdb_id.strip().upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    out_path.write_text(r.text)
    return out_path

# ==========================
# Data Loading (Cached)
# ==========================
@st.cache_data
def load_reference_csv():
    csv_path = Path("data") / "known_capsule_proteins.csv"
    return pd.read_csv(csv_path)

try:
    REFERENCE_DF = load_reference_csv()
except FileNotFoundError:
    st.error("❌ Missing data/known_capsule_proteins.csv")
    st.stop()

# ==========================
# Search Methods
# ==========================
def rmsd_search(query_pdb: Path, db_dir: Path, threshold: float, chain_id: str):
    qp = PDBWrapper(str(query_pdb))
    qp.parse()
    qcoords = qp.get_coordinates(chain_id=chain_id)
    if qcoords is None:
        return pd.DataFrame()

    rows = []
    for pdb_file in sorted(db_dir.glob("*.pdb")):
        dp = PDBWrapper(str(pdb_file))
        try:
            dp.parse()
            dcoords = dp.get_coordinates(chain_id=chain_id)
        except Exception:
            continue
        if dcoords is None:
            continue
        if len(qcoords) != len(dcoords):
            continue

        rmsd = calculate_rmsd(qcoords, dcoords)
        if rmsd is None:
            continue
        score = max(0.0, 1.0 - (rmsd / 10.0))
        if score >= threshold:
            rows.append({"method": "RMSD", "target": pdb_file.stem, "score": score, "rmsd": rmsd})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

def sequence_search(query_pdb: Path, db_dir: Path, threshold: float, chain_id: str):
    qp = PDBWrapper(str(query_pdb))
    qp.parse()
    qseq = qp.get_sequence(chain_id=chain_id)
    if not qseq:
        return pd.DataFrame()

    rows = []
    for pdb_file in sorted(db_dir.glob("*.pdb")):
        dp = PDBWrapper(str(pdb_file))
        try:
            dp.parse()
            dseq = dp.get_sequence(chain_id=chain_id)
        except Exception:
            continue
        if not dseq:
            continue

        alns = pairwise2.align.globalxx(qseq, dseq, one_alignment_only=True)
        if not alns:
            continue
        a = alns[0]
        matches = sum(x == y for x, y in zip(a.seqA, a.seqB))
        identity = matches / max(len(qseq), len(dseq))
        if identity >= threshold:
            rows.append({"method": "Sequence", "target": pdb_file.stem, "score": identity, "aligned_length": len(a.seqA)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

def distance_matrix_search(query_pdb: Path, db_dir: Path, threshold: float, chain_id: str):
    qp = PDBWrapper(str(query_pdb))
    qp.parse()
    qcoords = qp.get_coordinates(chain_id=chain_id)
    if qcoords is None:
        return pd.DataFrame()

    qdm = calculate_distance_matrix(qcoords)
    rows = []

    for pdb_file in sorted(db_dir.glob("*.pdb")):
        dp = PDBWrapper(str(pdb_file))
        try:
            dp.parse()
            dcoords = dp.get_coordinates(chain_id=chain_id)
        except Exception:
            continue
        if dcoords is None:
            continue

        ddm = calculate_distance_matrix(dcoords)
        if qdm.shape != ddm.shape:
            continue

        corr = distance_matrix_correlation(qdm, ddm)
        score = (corr + 1.0) / 2.0
        if score >= threshold:
            rows.append({"method": "Distance_Matrix", "target": pdb_file.stem, "score": score, "correlation": corr})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

def consensus_from_results(rmsd_df, seq_df, dm_df):
    all_targets = set()
    for d in [rmsd_df, seq_df, dm_df]:
        if d is not None and not d.empty:
            all_targets.update(d["target"].tolist())

    rows = []
    for t in sorted(all_targets):
        r = float(rmsd_df.loc[rmsd_df["target"] == t, "score"].values[0]) if (rmsd_df is not None and not rmsd_df.empty and (rmsd_df["target"] == t).any()) else 0.0
        s = float(seq_df.loc[seq_df["target"] == t, "score"].values[0]) if (seq_df is not None and not seq_df.empty and (seq_df["target"] == t).any()) else 0.0
        d = float(dm_df.loc[dm_df["target"] == t, "score"].values[0]) if (dm_df is not None and not dm_df.empty and (dm_df["target"] == t).any()) else 0.0

        final = 0.33 * r + 0.33 * s + 0.34 * d
        agree = sum([r > 0, s > 0, d > 0])
        confidence = "High (3/3)" if agree == 3 else "Medium (2/3)" if agree == 2 else "Low (1/3)"

        rows.append({
            "method": "Consensus",
            "target": t,
            "score": final,
            "agree_count": agree,
            "confidence": confidence,
            "rmsd_score": r,
            "sequence_score": s,
            "distance_matrix_score": d,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["score", "agree_count"], ascending=[False, False]).reset_index(drop=True)
    return df

# ==========================
# UI - Header
# ==========================
st.title("🧬 ProCap Search")
st.caption("Protein Capsule Structure Search Pipeline — benchmarking structure search methods for pathogen capsule identification")

# ==========================
# Sidebar Controls
# ==========================
with st.sidebar:
    st.header("⚙️ Settings & Input")
    chain_id = st.text_input("Chain ID", value="A", help="Most PDBs use chain A")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.70, 0.01)

    st.subheader("Query Input")
    pdb_id = st.text_input("PDB ID (e.g., 1R3F)", value="", placeholder="Leave blank to upload file")
    uploaded_query = st.file_uploader("Or upload PDB file", type=["pdb"])

    st.subheader("Database Mode")
    db_mode = st.radio(
        "Choose database:",
        ["Built-in PDBs (instant)", "Upload custom ZIP", "Built-in + Custom"],
        index=0,
    )
    uploaded_db_zip = st.file_uploader("Upload .zip (if needed)", type=["zip"]) if db_mode != "Built-in PDBs (instant)" else None

    st.subheader("Run Searches")
    col1, col2 = st.columns(2)
    with col1:
        run_rmsd = st.button("1️⃣ RMSD", use_container_width=True)
        run_dm = st.button("3️⃣ Distance Matrix", use_container_width=True)
    with col2:
        run_seq = st.button("2️⃣ Sequence", use_container_width=True)
        run_consensus = st.button("4️⃣ Consensus", use_container_width=True)

# ==========================
# Workspace Setup
# ==========================
tmp_root = Path(tempfile.gettempdir()) / "procap_cloud"
tmp_root.mkdir(parents=True, exist_ok=True)
query_dir = tmp_root / "query"
db_dir = tmp_root / "db"
query_dir.mkdir(parents=True, exist_ok=True)
db_dir.mkdir(parents=True, exist_ok=True)

# ==========================
# Query Preparation
# ==========================
query_pdb_path = None
try:
    if uploaded_query is not None:
        query_pdb_path = query_dir / uploaded_query.name
        query_pdb_path.write_bytes(uploaded_query.getvalue())
    elif pdb_id.strip():
        query_pdb_path = query_dir / f"{pdb_id.strip().upper()}.pdb"
        if not query_pdb_path.exists():
            with st.spinner("Downloading from RCSB PDB..."):
                download_pdb_by_id(pdb_id, query_pdb_path)
except Exception as e:
    st.error(f"Query error: {e}")
    query_pdb_path = None

# ==========================
# Database Preparation
# ==========================
for f in db_dir.glob("*.pdb"):
    try:
        f.unlink()
    except Exception:
        pass

if db_mode in ["Built-in PDBs (instant)", "Built-in + Custom"]:
    built_in_ids = REFERENCE_DF["PDB_ID"].dropna().astype(str).str.upper().tolist()
    with st.spinner("Loading built-in database..."):
        for pid in built_in_ids:
            try:
                out = db_dir / f"{pid}.pdb"
                if not out.exists():
                    download_pdb_by_id(pid, out)
            except Exception:
                pass

if db_mode in ["Upload custom ZIP", "Built-in + Custom"]:
    if uploaded_db_zip is not None:
        with st.spinner("Extracting custom database..."):
            with zipfile.ZipFile(io.BytesIO(uploaded_db_zip.getvalue()), "r") as z:
                z.extractall(db_dir)

# ==========================
# Session State
# ==========================
for key in ("rmsd_df", "seq_df", "dm_df", "cons_df"):
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame()

# ==========================
# Main Content
# ==========================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📋 About")
    st.markdown("""
    **Three complementary search methods:**
    1. **RMSD** — structural superposition (3D backbone alignment)
    2. **Sequence** — amino acid alignment (evolutionary conservation)
    3. **Distance Matrix** — pairwise distance correlation (fold topology)
    
    **Output:** Ranked results + consensus scoring across all methods
    """)

with col2:
    st.subheader("📊 Reference Data")
    st.dataframe(REFERENCE_DF.head(8), use_container_width=True, height=250)

st.divider()

# Database Status
pdb_count = len(list(db_dir.glob("*.pdb")))
st.metric("Database PDBs Ready", pdb_count)

if query_pdb_path is None:
    st.warning("⚠️ Provide a query: enter PDB ID or upload file (sidebar)")
    st.stop()

st.success(f"✅ Query: {query_pdb_path.name} | DB: {pdb_count} PDBs")

# ==========================
# Results
# ==========================
st.subheader("🔍 Results")

if run_rmsd:
    with st.spinner("RMSD search running..."):
        st.session_state.rmsd_df = rmsd_search(query_pdb_path, db_dir, threshold, chain_id)
    st.toast("✅ RMSD complete")

if not st.session_state.rmsd_df.empty:
    st.markdown("### 1️⃣ RMSD Results")
    st.dataframe(st.session_state.rmsd_df, use_container_width=True)
    st.download_button("📥 RMSD CSV", data=st.session_state.rmsd_df.to_csv(index=False).encode("utf-8"), file_name="rmsd.csv", mime="text/csv")

if run_seq:
    with st.spinner("Sequence search running..."):
        st.session_state.seq_df = sequence_search(query_pdb_path, db_dir, threshold, chain_id)
    st.toast("✅ Sequence complete")

if not st.session_state.seq_df.empty:
    st.markdown("### 2️⃣ Sequence Results")
    st.dataframe(st.session_state.seq_df, use_container_width=True)
    st.download_button("📥 Sequence CSV", data=st.session_state.seq_df.to_csv(index=False).encode("utf-8"), file_name="sequence.csv", mime="text/csv")

if run_dm:
    with st.spinner("Distance matrix search running..."):
        st.session_state.dm_df = distance_matrix_search(query_pdb_path, db_dir, threshold, chain_id)
    st.toast("✅ Distance matrix complete")

if not st.session_state.dm_df.empty:
    st.markdown("### 3️⃣ Distance Matrix Results")
    st.dataframe(st.session_state.dm_df, use_container_width=True)
    st.download_button("📥 Distance Matrix CSV", data=st.session_state.dm_df.to_csv(index=False).encode("utf-8"), file_name="distance_matrix.csv", mime="text/csv")

if run_consensus:
    if st.session_state.rmsd_df.empty and st.session_state.seq_df.empty and st.session_state.dm_df.empty:
        st.error("Run at least one method first")
    else:
        with st.spinner("Computing consensus..."):
            st.session_state.cons_df = consensus_from_results(st.session_state.rmsd_df, st.session_state.seq_df, st.session_state.dm_df)
        st.toast("✅ Consensus ready")

if not st.session_state.cons_df.empty:
    st.markdown("### 4️⃣ Consensus (BEST)")
    st.dataframe(st.session_state.cons_df, use_container_width=True)
    st.download_button("📥 Consensus CSV", data=st.session_state.cons_df.to_csv(index=False).encode("utf-8"), file_name="consensus.csv", mime="text/csv")

st.divider()
st.markdown("**Author:** Utkarsh Patel | **GitHub:** [procap-search](https://github.com/utkarshcpatel24052001-cell/procap-search)")
