from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

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

# =========================
# 1. PAGE CONFIG & CSS
# =========================
st.set_page_config(page_title="ProCap Professional | Structure Search", layout="wide")

st.markdown(
    """
<style>
/* LIMS Typography */
html, body, .stMarkdown, p, li, div, span, label {
  font-family: "Times New Roman", Times, serif !important;
  font-size: 18px !important;
  line-height: 1.45 !important;
}

/* Metric Widgets */
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

/* Sequence Alignment Box */
.seq-align-box {
    font-family: 'Courier New', Courier, monospace !important;
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    overflow-x: auto;
    font-size: 15px !important;
    line-height: 1.2 !important;
    white-space: pre;
}

/* Priority Banners */
.priority-high { color: #276749; background-color: #f0fff4; padding: 12px; border-left: 6px solid #2f855a; border-radius: 4px; font-weight: bold;}
.priority-med { color: #9c4221; background-color: #fffaf0; padding: 12px; border-left: 6px solid #dd6b20; border-radius: 4px; font-weight: bold;}
.priority-low { color: #9b2c2c; background-color: #fff5f5; padding: 12px; border-left: 6px solid #e53e3e; border-radius: 4px; font-weight: bold;}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# 2. CORE BIOPYTHON UTILITIES
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
        self.header = {}

    def parse(self):
        self.structure = self.parser.get_structure(self.pdb_path.stem, str(self.pdb_path))
        self.header = self.parser.get_header()
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
                seq.append(THREE_TO_ONE.get(residue.get_resname(), "X"))
            return "".join(seq)
        except KeyError: return None
        
    def get_experimental_metadata(self, chain_id="A"):
        """Extracts real-world crystallographic parameters."""
        resolution = self.header.get("resolution", np.nan)
        b_factors = []
        try:
            if self.structure:
                for residue in self.structure[0][chain_id]:
                    if "CA" in residue: b_factors.append(residue["CA"].get_bfactor())
        except KeyError: pass
        mean_b = np.mean(b_factors) if b_factors else np.nan
        return {"resolution": resolution, "mean_b_factor": mean_b}

def calculate_rmsd(coords1, coords2):
    if coords1 is None or coords2 is None or coords1.shape != coords2.shape: return None
    return float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1))))

def calculate_distance_matrix(coords): return cdist(coords, coords, metric="euclidean")

def distance_matrix_correlation(dm1, dm2):
    flat1 = dm1[np.triu_indices_from(dm1, k=1)]
    flat2 = dm2[np.triu_indices_from(dm2, k=1)]
    corr = np.corrcoef(flat1, flat2)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def download_pdb_by_id(pdb_id: str, out_path: Path) -> Optional[Path]:
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=15)
        r.raise_for_status()
        out_path.write_text(r.text)
        return out_path
    except requests.exceptions.RequestException:
        return None

def get_physchem_properties(seq: str) -> dict:
    """Calculates advanced real-world biochemical properties."""
    try:
        params = ProtParam.ProteinAnalysis(seq.replace("X", ""))
        return {
            "mw_kda": params.molecular_weight() / 1000, 
            "pi": params.isoelectric_point(), 
            "instability": params.instability_index(),
            "gravy": params.gravy(), # Grand Average of Hydropathy
            "extinction_coeff": params.molar_extinction_coefficient()[0] # Assuming reduced Cys
        }
    except Exception: 
        return {"mw_kda": np.nan, "pi": np.nan, "instability": np.nan, "gravy": np.nan, "extinction_coeff": np.nan}

# =========================
# 3. INTERACTIVE 3D & VISUALIZERS
# =========================
def render_3d_viewer(pdb_string: str):
    view = py3Dmol.view(width=800, height=400)
    view.addModel(pdb_string, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.setBackgroundColor('#ffffff')
    view.zoomTo()
    return view

def generate_alignment_html(seqA: str, seqB: str) -> str:
    alns = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True)
    if not alns: return "<p>Alignment failed.</p>"
    a = alns[0]
    
    match_line = ""
    for i in range(len(a.seqA)):
        if a.seqA[i] == a.seqB[i] and a.seqA[i] != "-": match_line += "|"
        elif a.seqA[i] != "-" and a.seqB[i] != "-": match_line += "."
        else: match_line += " "
        
    html = f"""
    <div class="seq-align-box">
<span style="color:#2c5282; font-weight:bold;">Query:</span>  {a.seqA}
<span style="color:#2f855a; font-weight:bold;">Match:</span>  {match_line}
<span style="color:#c53030; font-weight:bold;">Target:</span> {a.seqB}
    </div>
    """
    return html

def generate_fasta_export(query_seq: str, target_seq: str, target_id: str) -> str:
    return f">Query_Protein\n{query_seq}\n>Target_Homolog_{target_id}\n{target_seq}\n"

# =========================
# 4. DATABASE & PREDICTION LOGIC
# =========================
@st.cache_data
def load_reference_csv():
    try: return pd.read_csv("data/known_capsule_proteins.csv")
    except Exception: return pd.DataFrame(columns=["PDB_ID", "Organism", "Gene_Name", "Function", "Protein_Family", "Structure_Type"])

REFERENCE_DF = load_reference_csv()

def predict_biological_function(results_df: pd.DataFrame) -> dict:
    top_3 = results_df.head(3).dropna(subset=['Protein_Family', 'Structure_Type', 'Function'])
    if top_3.empty: return {"family": "Unknown", "structure": "Unknown", "function": "Unknown", "confidence": "Low"}
    
    return {
        "family": top_3['Protein_Family'].mode()[0],
        "structure": top_3['Structure_Type'].mode()[0],
        "function": top_3['Function'].mode()[0],
        "confidence": "High" if top_3['Protein_Family'].nunique() == 1 else "Moderate"
    }

def build_database_from_reference_pdb_ids(db_dir: Path, pdb_ids):
    db_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for pid in pdb_ids:
        out = db_dir / f"{pid}.pdb"
        if not out.exists():
            if download_pdb_by_id(pid, out): downloaded.append(pid)
        else: downloaded.append(pid)
    return downloaded

def run_full_pipeline(query_pdb: Path, db_dir: Path, chain_id: str) -> pd.DataFrame:
    qp = PDBWrapper(str(query_pdb))
    qp.parse()
    qcoords = qp.get_coordinates(chain_id=chain_id)
    qseq = qp.get_sequence(chain_id=chain_id)
    
    if qcoords is None or not qseq: return pd.DataFrame()
    qdm = calculate_distance_matrix(qcoords)

    results = []
    for pdb_file in sorted(db_dir.glob("*.pdb")):
        target_id = pdb_file.stem.upper()
        if target_id == query_pdb.stem.upper(): continue 
        
        dp = PDBWrapper(str(pdb_file))
        try: dp.parse()
        except Exception: continue
        
        dcoords = dp.get_coordinates(chain_id=chain_id)
        dseq = dp.get_sequence(chain_id=chain_id)
        
        if dcoords is None or not dseq: continue

        # RMSD
        r_score = max(0.0, 1.0 - (calculate_rmsd(qcoords, dcoords) / 10.0)) if len(qcoords) == len(dcoords) else 0.0
        
        # Sequence Alignment & Coverage
        alns = pairwise2.align.globalxx(qseq, dseq, one_alignment_only=True)
        if alns:
            matches = sum(x == y for x, y in zip(alns[0].seqA, alns[0].seqB))
            s_score = matches / max(len(qseq), len(dseq))
            alignment_coverage = (len(qseq) - alns[0].seqA.count('-')) / len(qseq) * 100
        else:
            s_score, alignment_coverage = 0.0, 0.0
        
        # Distance Matrix
        d_score = 0.0
        if qdm.shape == calculate_distance_matrix(dcoords).shape:
            d_score = (distance_matrix_correlation(qdm, calculate_distance_matrix(dcoords)) + 1.0) / 2.0

        final_score = (0.33 * r_score) + (0.33 * s_score) + (0.34 * d_score)
        if final_score > 0:
            results.append({
                "Target_PDB": target_id, "Consensus_Score": final_score,
                "RMSD_Score": r_score, "Seq_Identity": s_score, "Coverage_%": alignment_coverage, 
                "DM_Correlation": d_score, "Target_Seq": dseq
            })

    df = pd.DataFrame(results)
    if not df.empty:
        ref_meta = REFERENCE_DF.copy()
        ref_meta["PDB_ID"] = ref_meta["PDB_ID"].str.upper()
        df = df.merge(ref_meta, left_on="Target_PDB", right_on="PDB_ID", how="left")
        df = df.sort_values("Consensus_Score", ascending=False).reset_index(drop=True)
    return df

# =========================
# 5. APP START & UI
# =========================
st.title("🧬 ProCap Search Diagnostic Suite")
st.markdown("#### High-Throughput Structural Homology & Functional Annotation Pipeline")
st.divider()

tmp_root = Path(tempfile.gettempdir()) / "procap_space"
query_dir = tmp_root / "query"; query_dir.mkdir(parents=True, exist_ok=True)
db_dir = tmp_root / "db"; db_dir.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    chain_id = st.text_input("Target Chain ID", value="A", help="Default chain for biological assemblies.")
    
    st.subheader("Reference Database")
    st.info("The system automatically pulls the verified capsule proteins from your known_capsule_proteins.csv file.")
    if st.button("Initialize/Update Local Database"):
        with st.spinner("Downloading reference structures from RCSB..."):
            d = build_database_from_reference_pdb_ids(db_dir, REFERENCE_DF["PDB_ID"].dropna().astype(str).tolist())
            st.success(f"Database synchronized: {len(d)} PDBs ready.")

tab1, tab2, tab3 = st.tabs(["🔍 Structural Diagnostic", "🧪 3D Viewer & Alignments", "📊 Benchmark Analytics"])

# =========================
# TAB 1: Structural Diagnostic
# =========================
with tab1:
    c_in, c_res = st.columns([1, 1.8], gap="large")

    with c_in:
        st.subheader("I. Sequence/Structure Query")
        query_mode = st.radio("Input Source", ["RCSB Database ID", "Local PDB Upload"])
        
        if query_mode == "RCSB Database ID":
            pdb_id = st.text_input("Enter 4-letter PDB ID", value="1R3F").strip().upper()
        else:
            uploaded_query = st.file_uploader("Upload .pdb file", type=["pdb"])
            pdb_id = "USER_UPLOAD"

        execute = st.button("Execute Bioinformatics Pipeline", type="primary", use_container_width=True)

    with c_res:
        if execute:
            query_pdb_path = None
            if query_mode == "RCSB Database ID" and len(pdb_id) == 4:
                query_pdb_path = query_dir / f"{pdb_id}.pdb"
                if not query_pdb_path.exists():
                    if not download_pdb_by_id(pdb_id, query_pdb_path):
                        query_pdb_path = None
            elif query_mode == "Local PDB Upload" and uploaded_query:
                query_pdb_path = query_dir / uploaded_query.name
                query_pdb_path.write_bytes(uploaded_query.getvalue())

            if not query_pdb_path: 
                st.error("Failed to prepare query. Check if the PDB ID exists on the RCSB database.")
            else:
                with st.spinner("Executing structural superposition and sequence alignments..."):
                    results_df = run_full_pipeline(query_pdb_path, db_dir, chain_id)
                    
                    qp = PDBWrapper(str(query_pdb_path))
                    qp.parse()
                    qseq = qp.get_sequence(chain_id=chain_id)
                    physchem = get_physchem_properties(qseq) if qseq else {}
                    exp_meta = qp.get_experimental_metadata(chain_id=chain_id)

                if results_df.empty:
                    st.warning("No structurally viable homologs detected in the database. Check if chain ID is correct.")
                else:
                    top_hit = results_df.iloc[0]
                    score = top_hit["Consensus_Score"]
                    prediction = predict_biological_function(results_df)
                    
                    st.session_state['query_pdb'] = query_pdb_path.read_text()
                    st.session_state['qseq'] = qseq
                    st.session_state['tseq'] = top_hit['Target_Seq']
                    st.session_state['top_hit_id'] = top_hit['Target_PDB']

                    priority = "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
                    
                    with st.container(border=True):
                        st.markdown(f"### Diagnostic Status: <span class='priority-{priority.lower()}'>{priority} Confidence Homolog</span>", unsafe_allow_html=True)
                        st.markdown("**🔬 Automated Functional Annotation:**")
                        st.write(f"Based on a **{prediction['confidence'].lower()}-confidence** cluster consensus, the query sequence is annotated as a **{prediction['function']}**.")
                        st.write(f"Assigned to the **{prediction['family']}** protein family, adopting an **{prediction['structure']}** topology.")
                        
                        st.markdown("**🧬 Primary Biological Match:**")
                        st.write(f"• **Organism:** {top_hit.get('Organism', 'Unknown')} | **Gene:** {top_hit.get('Gene_Name', 'Unknown')}")
                        st.write(f"• **Alignment Coverage:** {top_hit.get('Coverage_%', 0):.1f}% of query residues mapped.")
                        if physchem.get('instability', 0) > 40: st.write("• **Stability Alert:** Primary sequence predicted to be unstable in vitro.")

                    st.markdown("### Crystallographic & PhysChem Profile")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Resolution", f"{exp_meta['resolution']:.2f} Å" if not np.isnan(exp_meta['resolution']) else "N/A")
                    m2.metric("Mean B-Factor", f"{exp_meta['mean_b_factor']:.1f}" if not np.isnan(exp_meta['mean_b_factor']) else "N/A")
                    m3.metric("GRAVY Index", f"{physchem.get('gravy', 0):.2f}", help="Grand Average of Hydropathy. Positive = Hydrophobic (membrane). Negative = Hydrophilic (soluble).")
                    m4.metric("Extinction Coeff.", f"{physchem.get('extinction_coeff', 0):.0f}", help="Molar Extinction Coefficient (M-1 cm-1).")

                    n1, n2, n3, n4 = st.columns(4)
                    n1.metric("Mol. Weight", f"{physchem.get('mw_kda', 0):.1f} kDa")
                    n2.metric("Sequence Length", f"{len(qseq) if qseq else 0} AA")
                    n3.metric("Isoelectric Pt (pI)", f"{physchem.get('pi', 0):.2f}")
                    n4.metric("Instability Idx", f"{physchem.get('instability', 0):.1f}")

                    st.markdown("### Ortholog Candidate Scoring Matrix")
                    st.dataframe(results_df[["Target_PDB", "Organism", "Gene_Name", "Consensus_Score", "Coverage_%", "Seq_Identity"]].head(5), use_container_width=True, hide_index=True)

# =========================
# TAB 2: 3D Viewer & Alignments
# =========================
with tab2:
    st.subheader("Structural Visualization & Sequence Interrogation")
    if 'query_pdb' in st.session_state:
        colA, colB = st.columns([1.5, 1], gap="large")
        
        with colA:
            st.markdown("#### Interactive 3D Query Topology")
            st.caption("Colored by Secondary Structure (Spectrum). Use mouse to rotate/zoom.")
            with st.container(border=True):
                view = render_3d_viewer(st.session_state['query_pdb'])
                components.html(view._make_html(), height=400)
                
            st.download_button("Download Query Structure (.pdb)", data=st.session_state['query_pdb'], file_name="query_structure.pdb", mime="chemical/x-pdb")

        with colB:
            st.markdown(f"#### Alignment vs Top Hit ({st.session_state['top_hit_id']})")
            st.caption("BioPython Global Sequence Alignment. (| = Exact Match)")
            
            align_html = generate_alignment_html(st.session_state['qseq'], st.session_state['tseq'])
            st.markdown(align_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Export Bioinformatics Assets")
            fasta_data = generate_fasta_export(st.session_state['qseq'], st.session_state['tseq'], st.session_state['top_hit_id'])
            st.download_button(
                label="Download Aligned FASTA (.fasta)",
                data=fasta_data,
                file_name=f"alignment_{st.session_state['top_hit_id']}.fasta",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.info("Execute a structural diagnostic in Tab 1 to generate 3D models and sequence alignments.")

# =========================
# TAB 3: Analytics
# =========================
with tab3:
    st.subheader("Reference Database Analytics")
    if not REFERENCE_DF.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.pie(REFERENCE_DF, names="Organism", title="Pathogen Distribution", hole=0.3)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.bar(REFERENCE_DF["Protein_Family"].value_counts().reset_index(), x="Protein_Family", y="count", title="Protein Families")
            st.plotly_chart(fig2, use_container_width=True)
