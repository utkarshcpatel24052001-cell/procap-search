import io
import os
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

# ---------------------------
# UI Config
# ---------------------------
st.set_page_config(
    page_title="ProCap Search | Protein Capsule Structure Search Demo",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------
# Core Utilities (embedded to keep Space self-contained)
# ---------------------------
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
        if self.structure is None:
            return None
        coords = []
        model = self.structure[0]
        chain = model[chain_id]
        for residue in chain:
            if "CA" in residue:
                coords.append(residue["CA"].get_coord())
        if not coords:
            return None
        return np.array(coords, dtype=float)

    def get_sequence(self, chain_id="A"):
        if self.structure is None:
            return None
        seq = []
        model = self.structure[0]
        chain = model[chain_id]
        for residue in chain:
            res = residue.get_resname()
            seq.append(THREE_TO_ONE.get(res, "X"))
        return "".join(seq)

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
    corr = np.corrcoef(flat1, flat2)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)

def download_pdb_by_id(pdb_id: str, out_path: Path):
    pdb_id = pdb_id.strip().upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    out_path.write_text(r.text)
    return out_path

# ---------------------------
# Reference dataset (from your repo)
# ---------------------------
@st.cache_data
def load_reference_csv():
    # Keep the CSV inside the Space repo under data/known_capsule_proteins.csv
    csv_path = Path("data") / "known_capsule_proteins.csv"
    return pd.read_csv(csv_path)

REFERENCE_DF = load_reference_csv()

# ---------------------------
# Database Handling
# ---------------------------
def ensure_database_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids):
    db_dir = ensure_database_dir(db_dir)
    downloaded = []
    failed = []

    for pid in pdb_ids:
        try:
            out = db_dir / f"{pid}.pdb"
            if not out.exists():
                download_pdb_by_id(pid, out)
            downloaded.append(pid)
        except Exception:
            failed.append(pid)

    return downloaded, failed

def extract_zip_to_db(zip_bytes: bytes, db_dir: Path):
    db_dir = ensure_database_dir(db_dir)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        z.extractall(db_dir)

# ---------------------------
# Search Methods (one-by-one)
# ---------------------------
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
        score = max(0.0, 1.0 - (rmsd / 10.0))  # normalize to 0-1
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
    # unify scores by target
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

# ---------------------------
# UI
# ---------------------------
st.title("🧬 ProCap Search")
st.caption("Protein Capsule Structure Search Pipeline — benchmarking protein structure search tools for pathogen capsule identification.")

with st.sidebar:
    st.header("Inputs")
    chain_id = st.text_input("Chain ID", value="A", help="Most PDBs use chain A. If not, change it.")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.70, 0.01)

    st.subheader("Query (choose one)")
    pdb_id = st.text_input("PDB ID (optional)", value="", placeholder="e.g., 1R3F")
    uploaded_query = st.file_uploader("Or upload a PDB file", type=["pdb"])

    st.divider()
    st.subheader("Database")
    db_mode = st.radio(
        "Database source",
        ["A) Built-in (known capsule proteins from CSV)", "B) Upload your own PDB database (.zip of .pdb files)", "A + B) Combine both"],
        index=2,
    )
    uploaded_db_zip = st.file_uploader("Upload .zip (only for option B or A+B)", type=["zip"])

    st.divider()
    st.subheader("Run")
    run_rmsd = st.button("1) Run RMSD Search")
    run_seq = st.button("2) Run Sequence Search")
    run_dm = st.button("3) Run Distance Matrix Search")
    run_consensus = st.button("4) Run Consensus (requires previous results)")

# Prepare temp workspace
tmp_root = Path(tempfile.gettempdir()) / "procap_space"
tmp_root.mkdir(parents=True, exist_ok=True)
query_dir = tmp_root / "query"
db_dir = tmp_root / "db"
query_dir.mkdir(parents=True, exist_ok=True)
db_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Build Query File
# ---------------------------
query_pdb_path = None
try:
    if uploaded_query is not None:
        query_pdb_path = query_dir / uploaded_query.name
        query_pdb_path.write_bytes(uploaded_query.getvalue())
    elif pdb_id.strip():
        query_pdb_path = query_dir / f"{pdb_id.strip().upper()}.pdb"
        if not query_pdb_path.exists():
            download_pdb_by_id(pdb_id, query_pdb_path)
except Exception as e:
    st.error(f"Failed to prepare query PDB: {e}")

# ---------------------------
# Build Database
# ---------------------------
db_info = st.container()
with db_info:
    st.subheader("Database status")

    # reset db directory each run for clarity
    for f in db_dir.glob("*.pdb"):
        try:
            f.unlink()
        except Exception:
            pass

    built_in_ids = REFERENCE_DF["PDB_ID"].dropna().astype(str).str.upper().tolist()

    use_a = db_mode.startswith("A)") or db_mode.startswith("A +")
    use_b = db_mode.startswith("B)") or db_mode.startswith("A +")

    downloaded, failed = ([], [])
    if use_a:
        with st.spinner("Preparing built-in database (downloading PDBs from RCSB if needed)..."):
            downloaded, failed = build_database_from_reference_pdb_ids(db_dir, built_in_ids)

    if use_b:
        if uploaded_db_zip is None:
            st.warning("Upload a .zip of .pdb files to use database option B.")
        else:
            with st.spinner("Extracting uploaded database zip..."):
                extract_zip_to_db(uploaded_db_zip.getvalue(), db_dir)

    pdb_count = len(list(db_dir.glob("*.pdb")))
    st.write(f"Database PDB files available: **{pdb_count}**")

    if use_a:
        st.write(f"Built-in PDB IDs downloaded/available: **{len(downloaded)}**")
        if failed:
            st.warning(f"Failed to download {len(failed)} PDB IDs from RCSB: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

# ---------------------------
# Results State
# ---------------------------
if "rmsd_df" not in st.session_state:
    st.session_state.rmsd_df = pd.DataFrame()
if "seq_df" not in st.session_state:
    st.session_state.seq_df = pd.DataFrame()
if "dm_df" not in st.session_state:
    st.session_state.dm_df = pd.DataFrame()
if "cons_df" not in st.session_state:
    st.session_state.cons_df = pd.DataFrame()

# ---------------------------
# Main Panels
# ---------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Project summary")
    st.markdown(
        """
**What this demo does**
- Accepts a query protein structure (upload PDB file or fetch by PDB ID)
- Searches for similar proteins using **RMSD**, **Sequence Identity**, and **Distance Matrix Correlation**
- Builds a **Consensus score** combining all three methods

**Reference dataset**
- 23 known capsule biosynthesis proteins across 5 pathogenic bacteria (from `data/known_capsule_proteins.csv`)

**Note**
- RMSD and distance-matrix searches require consistent chain selection and comparable coordinate lengths, so some structures may be skipped.
        """
    )

with col2:
    st.subheader("Reference proteins (preview)")
    st.dataframe(REFERENCE_DF.head(10), use_container_width=True)

st.divider()
st.subheader("Run results")

if query_pdb_path is None:
    st.info("Provide a query by uploading a PDB file or entering a PDB ID in the sidebar.")
else:
    st.success(f"Query ready: {query_pdb_path.name}")

# RMSD
if run_rmsd:
    if query_pdb_path is None:
        st.error("No query PDB provided.")
    else:
        with st.spinner("Running RMSD search..."):
            st.session_state.rmsd_df = rmsd_search(query_pdb_path, db_dir, threshold, chain_id)
        st.toast("RMSD search complete.", icon="✅")

if not st.session_state.rmsd_df.empty:
    st.markdown("### 1) RMSD results")
    st.dataframe(st.session_state.rmsd_df, use_container_width=True)
    st.download_button(
        "Download RMSD CSV",
        data=st.session_state.rmsd_df.to_csv(index=False).encode("utf-8"),
        file_name="rmsd_results.csv",
        mime="text/csv",
    )

# Sequence
if run_seq:
    if query_pdb_path is None:
        st.error("No query PDB provided.")
    else:
        with st.spinner("Running sequence search..."):
            st.session_state.seq_df = sequence_search(query_pdb_path, db_dir, threshold, chain_id)
        st.toast("Sequence search complete.", icon="✅")

if not st.session_state.seq_df.empty:
    st.markdown("### 2) Sequence results")
    st.dataframe(st.session_state.seq_df, use_container_width=True)
    st.download_button(
        "Download Sequence CSV",
        data=st.session_state.seq_df.to_csv(index=False).encode("utf-8"),
        file_name="sequence_results.csv",
        mime="text/csv",
    )

# Distance Matrix
if run_dm:
    if query_pdb_path is None:
        st.error("No query PDB provided.")
    else:
        with st.spinner("Running distance-matrix search..."):
            st.session_state.dm_df = distance_matrix_search(query_pdb_path, db_dir, threshold, chain_id)
        st.toast("Distance matrix search complete.", icon="✅")

if not st.session_state.dm_df.empty:
    st.markdown("### 3) Distance matrix results")
    st.dataframe(st.session_state.dm_df, use_container_width=True)
    st.download_button(
        "Download Distance Matrix CSV",
        data=st.session_state.dm_df.to_csv(index=False).encode("utf-8"),
        file_name="distance_matrix_results.csv",
        mime="text/csv",
    )

# Consensus
if run_consensus:
    if st.session_state.rmsd_df.empty and st.session_state.seq_df.empty and st.session_state.dm_df.empty:
        st.error("Run at least one method first (preferably all three).")
    else:
        with st.spinner("Computing consensus..."):
            st.session_state.cons_df = consensus_from_results(
                st.session_state.rmsd_df,
                st.session_state.seq_df,
                st.session_state.dm_df,
            )
        st.toast("Consensus ready.", icon="✅")

if not st.session_state.cons_df.empty:
    st.markdown("### 4) Consensus results")
    st.dataframe(st.session_state.cons_df, use_container_width=True)
    st.download_button(
        "Download Consensus CSV",
        data=st.session_state.cons_df.to_csv(index=False).encode("utf-8"),
        file_name="consensus_results.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("About")
st.markdown(
    """
- **Author:** Utkarsh Patel  
- **GitHub:** github.com/utkarshcpatel24052001-cell/procap-search  
- **Goal:** Benchmark protein structure search strategies for pathogen capsule identification  
"""
)
