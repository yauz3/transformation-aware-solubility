#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import rdFingerprintGenerator

from sklearn.random_projection import SparseRandomProjection

RDLogger.DisableLog("rdApp.error")

# =========================
# Config
# =========================
INPUT_CSV  = "test_group-based_maccs_map4_padel.csv"
OUTPUT_CSV = "test_group-based_maccs_map4_padel_sen_features.csv"

SOLUTE_SMILES_COL  = "SMILES_Solute"
SOLVENT_SMILES_COL = "SMILES_Solvent"
TEMP_COL           = "Temperature_K"

# Similarity fingerprint
MORGAN_BITS   = 1024
MORGAN_RADIUS = 2

# Random projection dimension (compact!)
RP_DIM = 64
RP_RANDOM_STATE = 42

# =========================
# Helpers
# =========================
def safe_mol(smi: str):
    if not isinstance(smi, str) or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi)

def fp_to_np(fp, n):
    arr = np.zeros((n,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr

def tanimoto_from_bit_arrays(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.sum((a == 1) & (b == 1))
    union = np.sum((a == 1) | (b == 1))
    return float(inter / union) if union > 0 else 0.0

def numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def split_padel_blocks(df: pd.DataFrame, exclude_cols):
    """
    Try to split PaDEL (numeric) columns into solute vs solvent blocks.
    Priority: solute_*, solvent_* prefixes.
    Fallback heuristics included.
    """
    num = [c for c in numeric_cols(df) if c not in exclude_cols]

    # 1) Preferred prefixes
    sol_pref = ["solute_", "sol_", "s_"]
    sv_pref  = ["solvent_", "sv_", "v_"]

    sol_cols = [c for c in num if any(c.startswith(p) for p in sol_pref)]
    sv_cols  = [c for c in num if any(c.startswith(p) for p in sv_pref)]

    # 2) If empty, try contains-based heuristics
    if len(sol_cols) == 0 or len(sv_cols) == 0:
        sol_cols2 = [c for c in num if ("solute" in c.lower()) or (c.lower().endswith("_s"))]
        sv_cols2  = [c for c in num if ("solvent" in c.lower()) or (c.lower().endswith("_v"))]
        if len(sol_cols) == 0: sol_cols = sol_cols2
        if len(sv_cols) == 0:  sv_cols = sv_cols2

    # 3) If still problematic, last resort: split by suffix conventions _1/_2 etc not reliable -> raise guidance
    if len(sol_cols) == 0 or len(sv_cols) == 0:
        raise ValueError(
            "PaDEL feature bloklarını (solute vs solvent) ayıramadım.\n"
            "Lütfen PaDEL kolonlarını 'solute_' ve 'solvent_' prefixleriyle adlandırın "
            "veya scriptte split_padel_blocks() içindeki prefix listesini güncelleyin."
        )

    # Keep intersection removed (avoid duplicates)
    overlap = set(sol_cols).intersection(set(sv_cols))
    if overlap:
        sol_cols = [c for c in sol_cols if c not in overlap]
        sv_cols  = [c for c in sv_cols  if c not in overlap]

    return sol_cols, sv_cols

def block_stats(X: np.ndarray, prefix: str):
    """
    Low-dim summary stats for a block (per-row).
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    eps = 1e-12
    mean = X.mean(axis=1)
    std  = X.std(axis=1)
    l1   = np.sum(np.abs(X), axis=1)
    l2   = np.sqrt(np.sum(X * X, axis=1) + eps)
    mx   = X.max(axis=1)
    mn   = X.min(axis=1)

    return pd.DataFrame({
        f"{prefix}mean": mean,
        f"{prefix}std": std,
        f"{prefix}l1": l1,
        f"{prefix}l2": l2,
        f"{prefix}max": mx,
        f"{prefix}min": mn,
    })

def pick_core_descriptor_columns(sol_cols, sv_cols):
    """
    Try to find core physchem-like descriptors in PaDEL columns.
    We match by suffix after prefix.
    """
    core_keys = [
        "MolWt", "MolLogP", "XlogP", "ALogP", "LogP",
        "TPSA", "HBD", "HBA", "RotB", "NumRotatableBonds",
        "Rings", "AromRings", "HeavyAtoms", "HeavyAtomCount", "FractionCSP3"
    ]

    def strip_prefix(c):
        for p in ["solute_", "solvent_", "sol_", "sv_", "s_", "v_"]:
            if c.startswith(p):
                return c[len(p):]
        return c

    # Build lookup by stripped name
    sol_map = {strip_prefix(c): c for c in sol_cols}
    sv_map  = {strip_prefix(c): c for c in sv_cols}

    picked = []
    for k in core_keys:
        if k in sol_map and k in sv_map:
            picked.append((k, sol_map[k], sv_map[k]))

    return picked  # list of (key, sol_col, sv_col)

# =========================
# Main
# =========================
df = pd.read_csv(INPUT_CSV)

# checks
for c in [SOLUTE_SMILES_COL, SOLVENT_SMILES_COL, TEMP_COL]:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors="coerce").astype(np.float32)
T = df[TEMP_COL].to_numpy(dtype=np.float32)
T = np.nan_to_num(T, nan=np.nanmedian(T))  # simple robust fill

solute_smiles  = df[SOLUTE_SMILES_COL].fillna("").astype(str).tolist()
solvent_smiles = df[SOLVENT_SMILES_COL].fillna("").astype(str).tolist()

# --- Split PaDEL blocks ---
exclude = {TEMP_COL}  # keep all other numeric cols as PaDEL candidates
sol_cols, sv_cols = split_padel_blocks(df, exclude_cols=exclude)

X_sol = df[sol_cols].to_numpy(dtype=np.float32)
X_sv  = df[sv_cols].to_numpy(dtype=np.float32)

X_sol = np.nan_to_num(X_sol, nan=0.0, posinf=0.0, neginf=0.0)
X_sv  = np.nan_to_num(X_sv,  nan=0.0, posinf=0.0, neginf=0.0)

# --- Morgan similarity ---
uniq_smiles = sorted(set(solute_smiles + solvent_smiles))
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

morgan_cache = {}
for s in uniq_smiles:
    mol = safe_mol(s)
    if mol is None:
        morgan_cache[s] = np.zeros(MORGAN_BITS, dtype=np.int8)
    else:
        fp = morgan_gen.GetFingerprint(mol)
        morgan_cache[s] = fp_to_np(fp, MORGAN_BITS)

sol_fp = np.vstack([morgan_cache[s] for s in solute_smiles])
sv_fp  = np.vstack([morgan_cache[s] for s in solvent_smiles])

sim = np.zeros(len(df), dtype=np.float32)
and_cnt = np.zeros(len(df), dtype=np.int32)
xor_cnt = np.zeros(len(df), dtype=np.int32)

for i in range(len(df)):
    a = sol_fp[i]; b = sv_fp[i]
    inter = np.sum((a == 1) & (b == 1))
    union = np.sum((a == 1) | (b == 1))
    sim[i] = (inter / union) if union > 0 else 0.0
    and_cnt[i] = int(inter)
    xor_cnt[i] = int(np.sum((a == 1) ^ (b == 1)))

# --- Temperature kernels (Approach 4) ---
T0 = 298.15
dT = (T - T0).astype(np.float32)
k_T   = T
k_iT  = (1.0 / np.maximum(T, 1e-6)).astype(np.float32)
k_log = np.log(np.maximum(T, 1e-6)).astype(np.float32)
k_dT  = dT
k_dT2 = (dT * dT).astype(np.float32)

alpha1, alpha2 = 0.01, 0.03
k_exp1 = np.exp(-alpha1 * np.abs(dT)).astype(np.float32)
k_exp2 = np.exp(-alpha2 * np.abs(dT)).astype(np.float32)

# --- Regime gates (Approach 5) ---
g_like   = (T * sim).astype(np.float32)
g_unlike = (T * (1.0 - sim)).astype(np.float32)

# =========================
# Build synthetic blocks
# =========================
blocks = []

# --- Base similarity + counts (small but useful) ---
base_sim = pd.DataFrame({
    "sim_morgan": sim,
    "T_x_sim_morgan": T * sim,
    "T_x_1mSim_morgan": T * (1.0 - sim),
    "morgan_and_cnt": and_cnt,
    "morgan_xor_cnt": xor_cnt,
    "T_x_morgan_and_cnt": T * and_cnt.astype(np.float32),
    "T_x_morgan_xor_cnt": T * xor_cnt.astype(np.float32),
})
blocks.append(base_sim)

# --- Approach 1: sim*T gating + PaDEL block summaries ---
sol_stats = block_stats(X_sol, prefix="padel_solute_")
sv_stats  = block_stats(X_sv,  prefix="padel_solvent_")
blocks += [sol_stats, sv_stats]

g = (sim * T).astype(np.float32)
sum_gate = pd.DataFrame({
    "g_simT": g,
    "g_simT_x_sol_mean": g * sol_stats["padel_solute_mean"].to_numpy(np.float32),
    "g_simT_x_sol_l2":   g * sol_stats["padel_solute_l2"].to_numpy(np.float32),
    "g_simT_x_sv_mean":  g * sv_stats["padel_solvent_mean"].to_numpy(np.float32),
    "g_simT_x_sv_l2":    g * sv_stats["padel_solvent_l2"].to_numpy(np.float32),
    "g_simT_x_mean_diff": g * (sol_stats["padel_solute_mean"].to_numpy(np.float32) -
                               sv_stats["padel_solvent_mean"].to_numpy(np.float32)),
    "g_simT_x_l2_diff":   g * (sol_stats["padel_solute_l2"].to_numpy(np.float32) -
                               sv_stats["padel_solvent_l2"].to_numpy(np.float32)),
})
blocks.append(sum_gate)

# --- Approach 2: core descriptor crosses (only if found) ---
picked = pick_core_descriptor_columns(sol_cols, sv_cols)
if len(picked) > 0:
    eps = 1e-6
    core_df = pd.DataFrame(index=df.index)
    for key, sc, vc in picked:
        s = df[sc].to_numpy(np.float32)
        v = df[vc].to_numpy(np.float32)

        d = s - v
        ad = np.abs(d)
        prod = s * v
        ratio = s / (v + eps)

        core_df[f"d_{key}"] = d
        core_df[f"absd_{key}"] = ad
        core_df[f"prod_{key}"] = prod
        core_df[f"ratio_{key}"] = ratio

        # temp + sim interactions (kept compact)
        core_df[f"T_x_absd_{key}"] = T * ad
        core_df[f"T_x_prod_{key}"] = T * prod
        core_df[f"sim_x_prod_{key}"] = sim * prod
        core_df[f"T_x_sim_x_prod_{key}"] = (T * sim) * prod

        # regime split
        core_df[f"g_like_x_prod_{key}"] = g_like * prod
        core_df[f"g_unlike_x_absd_{key}"] = g_unlike * ad

        # kernelized temperature modulation (few)
        core_df[f"exp1_x_absd_{key}"] = k_exp1 * ad
        core_df[f"exp2_x_prod_{key}"] = k_exp2 * prod

    blocks.append(core_df)

# --- Approach 3: Random Projection compact interactions ---
rp = SparseRandomProjection(n_components=RP_DIM, random_state=RP_RANDOM_STATE)
Z_sol = rp.fit_transform(X_sol).astype(np.float32)
Z_sv  = rp.transform(X_sv).astype(np.float32)

# latent interactions
Z_prod = Z_sol * Z_sv
Z_absd = np.abs(Z_sol - Z_sv)

# cosine in latent space
eps = 1e-12
num = np.sum(Z_sol * Z_sv, axis=1)
den = (np.linalg.norm(Z_sol, axis=1) * np.linalg.norm(Z_sv, axis=1)) + eps
Z_cos = (num / den).astype(np.float32)

rp_cols_prod = [f"rp_prod_{i}" for i in range(RP_DIM)]
rp_cols_absd = [f"rp_absd_{i}" for i in range(RP_DIM)]

rp_df = pd.DataFrame(Z_prod, columns=rp_cols_prod)
rp_absd_df = pd.DataFrame(Z_absd, columns=rp_cols_absd)

# gated versions (still compact: only prod)
rp_gate_df = pd.DataFrame((Z_prod * g[:, None]).astype(np.float32),
                          columns=[f"g_simT_x_rp_prod_{i}" for i in range(RP_DIM)])

rp_scalar = pd.DataFrame({
    "rp_cos": Z_cos,
    "T_x_rp_cos": T * Z_cos,
    "sim_x_rp_cos": sim * Z_cos,
    "T_x_sim_x_rp_cos": (T * sim) * Z_cos,
    "g_like_x_rp_cos": g_like * Z_cos,
    "g_unlike_x_rp_cos": g_unlike * Z_cos,
})

blocks += [rp_df, rp_absd_df, rp_gate_df, rp_scalar]

# --- Approach 4: temperature kernels (standalone small set) ---
temp_kernels = pd.DataFrame({
    "T": k_T,
    "invT": k_iT,
    "logT": k_log,
    "dT": k_dT,
    "dT2": k_dT2,
    "exp_absdT_a001": k_exp1,
    "exp_absdT_a003": k_exp2,
})
blocks.append(temp_kernels)

# --- Approach 5: regime gates (standalone) ---
regime_df = pd.DataFrame({
    "g_like": g_like,
    "g_unlike": g_unlike,
    "sim": sim,
    "1m_sim": (1.0 - sim).astype(np.float32),
})
blocks.append(regime_df)

# =========================
# Concatenate + Save
# =========================
synthetic = pd.concat(blocks, axis=1)

# Keep original columns + new synthetic features (do not drop labels/SMILES)
out = pd.concat([df.reset_index(drop=True), synthetic.reset_index(drop=True)], axis=1)

out.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved: {OUTPUT_CSV}")
print(f"Rows: {out.shape[0]}  Cols: {out.shape[1]}")
print(f"PaDEL blocks: solute={len(sol_cols)} solvent={len(sv_cols)}")
print(f"Core physchem pairs found: {len(picked)}")
print(f"RP_DIM={RP_DIM}, MorganBits={MORGAN_BITS}")
