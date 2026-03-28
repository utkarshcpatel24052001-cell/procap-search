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
    html, body, [class*="css"], .stApp { background-color: #000000 !important; color: #ffffff !important; }
    html, body, .stMarkdown, p, li, span, label { font-family: "Segoe UI", "Inter", sans-serif !important; font-size: 16px !important; color: #ffffff !important; }
    h1, h2, h3, h4 { color: #ff3333 !important; font-weight: 700 !important; }
    h3 { font-size: 20px !important; border-bottom: 1px solid #333333; padding-bottom: 8px;}
    div[data-testid="stMetricLabel"] p { font-size: 13px !important; font-weight: 700 !important; color: #cccccc !important; text-transform: uppercase; }
    div[data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 800 !important; color: #ff3333 !important; }
    .alert-high { background-color: #002200; border-left: 6px solid #00ff00; padding: 16px; border-radius: 6px; color: #00ff00; font-weight: 600; margin-bottom: 15px;}
    .alert-medium { background-color: #331a00; border-left: 6px solid #ffa500; padding: 16px; border-radius: 6px; color: #ffa500; font-weight: 600; margin-bottom: 15px;}
    .alert-low { background-color: #330000; border-left: 6px solid #ff0000; padding: 16px; border-radius: 6px; color: #ff3333; font-weight: 600; margin-bottom: 15px;}
    .info-box { background-color: #001133; border-left: 6px solid #3399ff; padding: 16px; border-radius: 6px; color: #3399ff; font-weight: 600; margin-bottom: 15px;}
    .success-box { background-color: #001a00; border-left: 6px solid #00ff00; padding: 16px; border-radius: 6px; color: #00ff00; font-weight: 600; margin-bottom: 15px;}
    .chain-box { background-color: #0a1a2e; border-left: 6px solid #16c784; padding: 12px; border-radius: 6px; color: #16c784; font-weight: 600; margin-bottom: 10px;}
    .method-box { background-color: #111111; border: 1px solid #333333; border-left: 6px solid #ff3333; padding: 16px; border-radius: 6px; color: #ffffff; font-weight: 500; margin-bottom: 15px;}
    .seq-align-box { font-family: 'Fira Code', monospace !important; background-color: #111111; padding: 16px; border-radius: 6px; border: 1px solid #333333; overflow-x: auto; font-size: 14px; white-space: pre !important; color: #ffffff; }
    .stButton > button { font-weight: 600 !important; border-radius: 6px !important; background-color: #222222 !important; color: #ffffff !important; border: 1px solid #ff3333 !important; }
    .stButton > button:hover { background-color: #ff3333 !important; color: #000000 !important; }
    .stContainer { border: 1px solid #333333 !important; border-radius: 8px !important; padding: 16px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 2. WEB DATABASE FUNCTIONS
# ==================================================

def search_pdb_rcsb(keyword: str, limit: int = 50) -> List[Dict]:
    try:
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        query = {"query": {"type": "terminal", "service": "text", "parameters": {"value": keyword}}, "return_type": "entry", "num_result_set": limit}
        response = requests.post(url, json=query, headers={"Content-Type": "application/json"}, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return [{"PDB_ID": res["identifier"], "Source": "RCSB PDB"} for res in data.get("result_set", [])[:limit]]
        return []
    except Exception: return []

def search_uniprot_for_pdb(protein_name: str, organism: str = "") -> List[Dict]:
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
        return []
    except Exception: return []

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
        except Exception: return False

    def get_available_chains(self) -> List[Dict]:
        if not self.structure: return []
        chains_info = []
        for chain in self.structure[0]:
            residue_count = len([r for r in chain if r.id[0] == " "])
            chains_info.append({'id': chain.id, 'residues': residue_count})
        return sorted(chains_info, key=lambda x: x['id'])

    def get_coordinates(self, chain_id="A") -> Optional[np.ndarray]:
        if not self.structure: return None
        try:
            coords = [res["CA"].get_coord() for res in self.structure[0][chain_id] if "CA" in res]
            return np.array(coords, dtype=float) if coords else None
        except Exception: return None

    def get_sequence(self, chain_id="A") -> Optional[str]:
        if not self.structure: return None
        try:
            seq = []
            for residue in self.structure[0][chain_id]:
                if residue.id[0] != " ": continue # Skip heteroatoms (ligands/water)
                seq.append(THREE_TO_ONE.get(residue.get_resname(), "X"))
            return "".join(seq) if seq else None
        except Exception: return None
        
    def get_experimental_metadata(self, chain_id="A"):
        res = self.header.get("resolution", np.nan)
        b_factors = []
        try:
            if self.structure:
                b_factors = [res["CA"].get_bfactor() for res in self.structure[0][chain_id] if "CA" in res]
        except Exception: pass
        return {"resolution": res, "mean_b_factor": np.mean(b_factors) if b_factors else np.nan}

# ==================================================
# 4. SCORING ALGORITHMS
# ==================================================

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> Optional[float]:
    if coords1 is None or coords2 is None or coords1.shape != coords2.shape: return None
    try: return float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1))))
    except Exception: return None

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1: np.ndarray, dm2: np.ndarray) -> float:
    try:
        flat1, flat2 = dm1[np.triu_indices_from(dm1, k=1)], dm2[np.triu_indices_from(dm2, k=1)]
        corr = np.corrcoef(flat1, flat2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception: return 0.0

def calculate_sequence_identity(seqA: str, seqB: str) -> Tuple[float, int]:
    try:
        if not seqA or not seqB: return 0.0, 0
        alns = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)
        if not alns: return 0.0, 0
        a = alns[0]
        matches = sum(x == y for x, y in zip(a.seqA, a.seqB))
        return float(matches / max(len(seqA), len(seqB))), len(a.seqA)
    except Exception: return 0.0, 0

def download_pdb_by_id(pdb_id: str, out_path: Path) -> bool:
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id.strip().upper()}.pdb", timeout=15)
        r.raise_for_status()
        out_path.write_text(r.text)
        return True
    except Exception: return False

def get_advanced_physchem_properties(seq: str) -> Dict[str, float]:
    clean_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', str(seq).upper())
    default_props = {"mw_kda": np.nan, "pi": np.nan, "instability": np.nan, "aromaticity": np.nan, "gravy": np.nan, "extinction_coeff": np.nan}
    if len(clean_seq) < 2: return default_props
    try:
        params = ProtParam.ProteinAnalysis(clean_seq)
        pi = IsoelectricPoint(clean_seq).pH()
        return {
            "mw_kda": params.molecular_weight() / 1000, "pi": pi, "instability": params.instability_index(),
            "aromaticity": params.aromaticity(), "gravy": params.gravy(), "extinction_coeff": params.molar_extinction_coefficient()[0]
        }
    except Exception: return default_props

# ==================================================
# 5. DATABASE PIPELINE
# ==================================================
@st.cache_data
def load_reference_csv() -> pd.DataFrame:
    try: return pd.read_csv("data/known_capsule_proteins.csv")
    except Exception: return pd.DataFrame(columns=["PDB_ID", "Organism", "Gene_Name", "Function", "Protein_Family", "Structure_Type"])

REFERENCE_DF = load_reference_csv()

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids: list):
    db_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    progress_bar = st.progress(0)
    for i, pid in enumerate(pdb_ids):
        out = db_dir / f"{pid}.pdb"
        if out.exists() or download_pdb_by_id(pid, out): downloaded.append(pid)
        progress_bar.progress((i + 1) / len(pdb_ids))
    return downloaded

def run_comprehensive_diagnostic(query_pdb: Path, db_dir: Path, chain_id: str) -> Tuple[pd.DataFrame, Dict]:
    qp = PDBWrapper(str(query_pdb)); qp.parse()
    qcoords, qseq = qp.get_coordinates(chain_id), qp.get_sequence(chain_id)
    if qcoords is None or not qseq: return pd.DataFrame(), {"status": "error", "message": "Chain data missing."}
    
    qdm = calculate_distance_matrix(qcoords)
    results, total_files = [], len(list(db_dir.glob("*.pdb")))
    progress_bar = st.progress(0)
    
    for idx, pdb_file in enumerate(sorted(db_dir.glob("*.pdb"))):
        progress_bar.progress((idx + 1) / total_files)
        target_id = pdb_file.stem.upper()
        if target_id == query_pdb.stem.upper(): continue
        
        dp = PDBWrapper(str(pdb_file)); dp.parse()
        d_chain = chain_id if chain_id in [c['id'] for c in dp.get_available_chains()] else (dp.get_available_chains()[0]['id'] if dp.get_available_chains() else None)
        if not d_chain: continue
        
        dcoords, dseq = dp.get_coordinates(d_chain), dp.get_sequence(d_chain)
        if dcoords is None or not dseq: continue
        
        r_score = max(0.0, 1.0 - (calculate_rmsd(qcoords, dcoords) / 10.0)) if len(qcoords) == len(dcoords) else 0.0
        s_score, aln_len = calculate_sequence_identity(qseq, dseq)
        d_score = (distance_matrix_correlation(qdm, calculate_distance_matrix(dcoords)) + 1.0) / 2.0 if qdm.shape == calculate_distance_matrix(dcoords).shape else 0.0
        
        final_score = (0.33 * r_score) + (0.33 * s_score) + (0.34 * d_score)
        if final_score > 0:
            results.append({
                "Target_PDB": target_id, "Target_Chain": d_chain, "Consensus_Score": final_score,
                "RMSD_Å": calculate_rmsd(qcoords, dcoords), "RMSD_Score": r_score,
                "Seq_Identity_%": s_score * 100, "Seq_Score": s_score, "DM_Correlation": d_score,
                "Query_Coverage_%": (aln_len / len(qseq) * 100), "Target_Seq": dseq,
            })
    
    df = pd.DataFrame(results)
    if not df.empty:
        ref_meta = REFERENCE_DF.copy(); ref_meta["PDB_ID"] = ref_meta["PDB_ID"].str.upper()
        df = df.merge(ref_meta, left_on="Target_PDB", right_on="PDB_ID", how="left").drop("PDB_ID", axis=1)
        df["Organism"] = df["Organism"].fillna("Web Database Search")
        df = df.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    
    return df, {"status": "success", "query_seq_length": len(qseq), **get_advanced_physchem_properties(qseq)}

# ==================================================
# 6. UI & VISUALIZATION
# ==================================================

def render_3d_structure(pdb_content: str):
    view = py3Dmol.view(width=800, height=450)
    view.addModel(pdb_content, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.setBackgroundColor('#000000'); view.zoomTo()
    return view

st.title("🧬 ProCap Search Diagnostic Suite")
st.markdown("##### Structural Homology & Functional Annotation Pipeline v3.0")
st.divider()

tmp_root = Path(tempfile.gettempdir()) / "procap_professional"
db_dir = tmp_root / "db"; db_dir.mkdir(parents=True, exist_ok=True)
query_dir = tmp_root / "query"; query_dir.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    chain_id = st.text_input("Target Chain ID", value="A", max_chars=1).strip().upper() or "A"
    
    st.subheader("🗄️ Database")
    if st.button("📥 Sync Local CSV Database", use_container_width=True):
        for f in db_dir.glob("*.pdb"): f.unlink()
        d = build_database_from_reference_pdb_ids(db_dir, REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist())
        st.success(f"Loaded {len(d)} structures.")
    
    st.divider()
    st.subheader("🌐 Web Expansion")
    w_type = st.selectbox("Search Type", ["Keyword Search", "UniProt Name"])
    w_term = st.text_input("Term")
    if st.button("🔍 Search Web"):
        res = search_pdb_rcsb(w_term, 20) if w_type == "Keyword Search" else search_uniprot_for_pdb(w_term)
        st.session_state['web_res'] = res
    
    if 'web_res' in st.session_state and st.session_state['web_res']:
        st.dataframe(pd.DataFrame(st.session_state['web_res']), hide_index=True)
        if st.button("⬇️ Download All"):
            added = sum(1 for r in st.session_state['web_res'] if download_pdb_by_id(r["PDB_ID"], db_dir / f"{r['PDB_ID']}.pdb"))
            st.success(f"Added {added} structures.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Diagnostic", "🧪 3D & Alignment", "📊 Analytics", "📖 Algorithm", "🧬 Breakdown"])

with tab1:
    c1, c2 = st.columns([1, 1.8], gap="large")
    with c1:
        st.subheader("Input")
        q_mode = st.radio("Source", ["RCSB ID", "Upload"])
        if q_mode == "RCSB ID": q_id = st.text_input("PDB ID", value="1R3F").strip().upper()
        else: u_file = st.file_uploader("PDB File", type=["pdb"])
        
        if st.button("▶ EXECUTE PIPELINE", type="primary", use_container_width=True):
            q_path = (query_dir / f"{q_id}.pdb") if q_mode == "RCSB ID" else (query_dir / "user.pdb")
            if q_mode == "RCSB ID": download_pdb_by_id(q_id, q_path)
            elif u_file: q_path.write_bytes(u_file.getvalue())
            
            if q_path.exists():
                results_df, diags = run_comprehensive_diagnostic(q_path, db_dir, chain_id)
                st.session_state['results_df'] = results_df
                st.session_state['diags'] = diags
                st.session_state['q_pdb'] = q_path.read_text()
                st.session_state['q_path'] = q_path

    with c2:
        if 'results_df' in st.session_state:
            res_df, diags = st.session_state['results_df'], st.session_state['diags']
            if not res_df.empty:
                top = res_df.iloc[0]
                conf, alert = classify_confidence_level(top["Consensus_Score"])
                st.markdown(f'<div class="{alert}">🧬 CONSENSUS: {top["Consensus_Score"]:.3f} | {conf}</div>', unsafe_allow_html=True)
                
                m_c1, m_c2, m_c3 = st.columns(3)
                m_c1.markdown(f'<div class="method-box"><b>RMSD</b><br>{top["RMSD_Score"]:.3f}</div>', unsafe_allow_html=True)
                m_c2.markdown(f'<div class="method-box"><b>SEQ</b><br>{top["Seq_Score"]:.3f}</div>', unsafe_allow_html=True)
                m_c3.markdown(f'<div class="method-box"><b>TOPOL</b><br>{top["DM_Correlation"]:.3f}</div>', unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown("### Advanced PhysChem Profile")
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("MW", f"{diags['mw_kda']:.1f} kDa")
                    p2.metric("pI", f"{diags['pi']:.2f}")
                    p3.metric("Instability", f"{diags['instability']:.1f}")
                    p4.metric("GRAVY", f"{diags['gravy']:.2f}")

                st.markdown(f"**Found {len(res_df)} Structural Homologs**")
                st.dataframe(res_df[["Target_PDB", "Consensus_Score", "Organism", "Protein_Family", "RMSD_Å", "Seq_Identity_%"]].head(15), use_container_width=True, hide_index=True)

with tab2:
    if 'q_pdb' in st.session_state:
        v_c1, v_c2 = st.columns([1.3, 1], gap="large")
        with v_c1:
            st.markdown("### 3D Query Topology")
            view = render_3d_structure(st.session_state['q_pdb'])
            components.html(view._make_html(), height=480)
        with v_c2:
            top = st.session_state['results_df'].iloc[0]
            st.markdown(f"### Alignment vs {top['Target_PDB']}")
            qp = PDBWrapper(str(st.session_state['q_path'])); qp.parse()
            qseq = qp.get_sequence(chain_id)
            
            # Robust Visual Alignment
            alns = pairwise2.align.globalxx(qseq, top['Target_Seq'], one_alignment_only=True)
            if alns:
                a = alns[0]
                h = '<div class="seq-align-box">'
                for i in range(0, len(a.seqA), 40):
                    match = "".join(["|" if a.seqA[j] == a.seqB[j] and a.seqA[j] != "-" else " " for j in range(i, min(i+40, len(a.seqA)))])
                    h += f"<span style='color:#66b3ff;'>Q: </span>{a.seqA[i:i+40]}\n<span style='color:#00ff00;'>M: </span>{match}\n<span style='color:#ff3333;'>T: </span>{a.seqB[i:i+40]}\n\n"
                st.markdown(h + "</div>", unsafe_allow_html=True)
    else: st.info("Run diagnostic first.")

with tab3:
    if not REFERENCE_DF.empty:
        ch1, ch2 = st.columns(2)
        with ch1: st.plotly_chart(px.pie(REFERENCE_DF, names="Organism", hole=0.3, template="plotly_dark"), use_container_width=True)
        with ch2: st.plotly_chart(px.bar(REFERENCE_DF["Protein_Family"].value_counts().reset_index(), x="Protein_Family", y="count", template="plotly_dark"), use_container_width=True)

with tab4:
    st.markdown("### Methodology\n1. **RMSD**: 3D Coordinate Superposition\n2. **Sequence**: Needleman-Wunsch Global Alignment\n3. **Distance Matrix**: Topology Pearson Correlation")

with tab5:
    if 'results_df' in st.session_state:
        st.plotly_chart(px.bar(st.session_state['results_df'].head(10), x="Target_PDB", y=["RMSD_Score", "Seq_Score", "DM_Correlation"], barmode="group", template="plotly_dark"), use_container_width=True)
