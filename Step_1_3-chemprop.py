#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13-Chemprop.py
Chemprop feature generator ile SMILES -> feature üretir ve CSV yazar.

Not:
- Chemprop sürümüne göre generator isimleri değişebilir.
- Script şu sırayla dener:
    rdkit_2d_normalized -> rdkit_2d -> morgan
- NumPy / Chemprop uyumsuzluğu için
    np.VisibleDeprecationWarning
  monkey-patch uygulanır.
"""

import re
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

# =========================
# Config
# =========================
INPUT_CSV  = "test_group-based_maccs_map4.csv"
OUTPUT_CSV = "test_group-based_maccs_map4_chemprop.csv"

SOLUTE_COL  = "SMILES_Solute"
SOLVENT_COL = "SMILES_Solvent"

# None ise otomatik fallback:
#   rdkit_2d_normalized -> rdkit_2d -> morgan
GENERATOR_NAME = None

KEEP_INPUT_COLS = True
NAN_ON_FAIL = True

# =========================
# Compatibility patch
# =========================
# Bazı eski chemprop sürümleri import sırasında
# np.VisibleDeprecationWarning bekliyor.
if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = DeprecationWarning

# =========================
# Helpers
# =========================
def safe_colname(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def resolve_chemprop_generator(requested: Optional[str] = None) -> Tuple[str, Callable]:
    """
    Chemprop feature generator'ı alır.
    requested verilmişse onu dener, yoksa fallback sırası uygular.
    """
    try:
        from chemprop.features import get_features_generator, get_available_features_generators
    except Exception as e:
        raise ImportError(
            "Chemprop import edilemedi.\n"
            "Muhtemel neden: chemprop sürümü ile numpy sürümü uyumsuz.\n"
            "Script içinde np.VisibleDeprecationWarning monkey-patch uygulandı, "
            "ama import yine de başarısız oldu.\n\n"
            "Deneyebileceğin kurulumlar:\n"
            "  pip install chemprop\n"
            "veya eski chemprop için numpy düşürmek gerekebilir.\n\n"
            f"Original import error: {e}"
        ) from e

    available = set(get_available_features_generators())

    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"Requested generator '{requested}' yok.\n"
                f"Available: {sorted(available)}"
            )
        return requested, get_features_generator(requested)

    for name in ["rdkit_2d_normalized", "rdkit_2d", "morgan"]:
        if name in available:
            return name, get_features_generator(name)

    raise ValueError(
        "Uygun generator bulunamadı.\n"
        f"Available: {sorted(available)}"
    )

def featurize_unique_smiles(
    uniq_smiles: List[str],
    generator_fn: Callable,
    nan_on_fail: bool = True
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    uniq_smiles için feature üretir, cache dict döner.
    Dimension'ı ilk başarılı SMILES'ten öğrenir.
    """
    cache: Dict[str, np.ndarray] = {}
    dim: Optional[int] = None

    # ilk pass: dim bul
    for s in uniq_smiles:
        s2 = "" if s is None else str(s).strip()
        if not s2:
            continue
        try:
            v = generator_fn(s2)
            if v is None:
                continue
            v = np.asarray(v, dtype=np.float32).ravel()
            if v.size > 0:
                dim = int(v.size)
                break
        except Exception:
            continue

    if dim is None:
        raise RuntimeError(
            "Hiçbir SMILES'ten feature üretilemedi (dim bulunamadı). "
            "SMILES / generator / chemprop kurulumunu kontrol et."
        )

    bad_vec = np.full((dim,), np.nan, dtype=np.float32) if nan_on_fail else np.zeros((dim,), dtype=np.float32)

    for s in uniq_smiles:
        s2 = "" if s is None else str(s).strip()

        if not s2:
            cache[s] = bad_vec.copy()
            continue

        try:
            v = generator_fn(s2)
            v = np.asarray(v, dtype=np.float32).ravel()
            if v.size != dim:
                cache[s] = bad_vec.copy()
            else:
                cache[s] = v
        except Exception:
            cache[s] = bad_vec.copy()

    return cache, dim

def build_side_df(smiles_series: pd.Series, prefix: str, generator_fn: Callable, gen_name: str) -> pd.DataFrame:
    smiles_series = smiles_series.fillna("").astype(str)
    uniq = smiles_series.unique().tolist()

    print(f"✅ Unique SMILES ({prefix}): {len(uniq)}")

    cache, dim = featurize_unique_smiles(uniq, generator_fn, nan_on_fail=NAN_ON_FAIL)

    X = np.vstack([cache[s] for s in smiles_series.tolist()]).astype(np.float32)

    cols = [safe_colname(f"{prefix}_chemprop_{gen_name}_{i}") for i in range(dim)]
    return pd.DataFrame(X, columns=cols)

# =========================
# Main
# =========================
def main():
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    for c in [SOLUTE_COL, SOLVENT_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    gen_name, gen_fn = resolve_chemprop_generator(GENERATOR_NAME)

    print(f"✅ Input: {INPUT_CSV} | rows={len(df)}")
    print(f"✅ Chemprop generator: {gen_name}")

    print("✅ Computing SOLUTE Chemprop features...")
    sol_df = build_side_df(df[SOLUTE_COL], prefix="solute", generator_fn=gen_fn, gen_name=gen_name)

    print("✅ Computing SOLVENT Chemprop features...")
    sv_df = build_side_df(df[SOLVENT_COL], prefix="solvent", generator_fn=gen_fn, gen_name=gen_name)

    if KEEP_INPUT_COLS:
        out = pd.concat(
            [df.reset_index(drop=True), sol_df.reset_index(drop=True), sv_df.reset_index(drop=True)],
            axis=1
        )
    else:
        out = pd.concat(
            [df[[SOLUTE_COL, SOLVENT_COL]].reset_index(drop=True), sol_df.reset_index(drop=True), sv_df.reset_index(drop=True)],
            axis=1
        )

    out.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Saved: {OUTPUT_CSV}")
    print(f"Rows: {out.shape[0]}  Cols: {out.shape[1]}")
    print("Included:", f"generator={gen_name}")

if __name__ == "__main__":
    main()
