#!/usr/bin/env bash
# ============================================================
# Full RGB evaluation
# writer -> gpvm_k_lambda_u0 -> heatmap -> bottom-metal profile
# - Uses --best-json everywhere
# - Writer outputs fixed-name files next to best-json, so put best-json in per-color dirs
# ============================================================

LAMBDA_R="${LAMBDA_R:-620}"
LAMBDA_G="${LAMBDA_G:-530}"
LAMBDA_B="${LAMBDA_B:-460}"

LAM_SPAN="${LAM_SPAN:-300}"
N_LAM="${N_LAM:-201}"

WRITER="run_phase3_opt_then_gpvm_eml_profile.py"
U0_MAIN="run_gpvm_K_lambda_u0.py"
U0_AVSB="run_gpvm_K_lambda_u0_A_vs_B.py"
HEATMAP="run_gpvm_K_lambda_kpar_map_strict.py"
BOTTOM_METAL="run_bottom_metal_PEC_proxy.py"

# ------------------------------------------------------------
# (FIX) Put best-json into per-color directories to avoid overwrite
# ------------------------------------------------------------
OUT_STEP2="out_gpvm_step2"
GEOM_R="${OUT_STEP2}/R/best_geometry.json"
GEOM_G="${OUT_STEP2}/G/best_geometry.json"
GEOM_B="${OUT_STEP2}/B/best_geometry.json"
mkdir -p "${OUT_STEP2}/R" "${OUT_STEP2}/G" "${OUT_STEP2}/B"

# output dirs (separate to avoid cache collisions)
OUT_U0_R="gpvm_k_lambda_u0/out_R"
OUT_U0_G="gpvm_k_lambda_u0/out_G"
OUT_U0_B="gpvm_k_lambda_u0/out_B"

OUT_HM_R="gpvm_K_lambda_kpar_map_strict/out_R"
OUT_HM_G="gpvm_K_lambda_kpar_map_strict/out_G"
OUT_HM_B="gpvm_K_lambda_kpar_map_strict/out_B"

# (FIX) remove accidental double slash
OUT_BM_R="out_bottom_metal/out_bottom_metal_R"
OUT_BM_G="out_bottom_metal/out_bottom_metal_G"
OUT_BM_B="out_bottom_metal/out_bottom_metal_B"

mkdir -p "${OUT_U0_R}" "${OUT_U0_G}" "${OUT_U0_B}"
mkdir -p "${OUT_HM_R}" "${OUT_HM_G}" "${OUT_HM_B}"
mkdir -p "${OUT_BM_R}" "${OUT_BM_G}" "${OUT_BM_B}"

echo "=== (1) Writer: generate best_geometry.json in per-color dirs ==="
python "${WRITER}" --lambda-nm "${LAMBDA_R}" --best-json "${GEOM_R}"
python "${WRITER}" --lambda-nm "${LAMBDA_G}" --best-json "${GEOM_G}"
python "${WRITER}" --lambda-nm "${LAMBDA_B}" --best-json "${GEOM_B}"

echo "=== (2) gpvm_k_lambda_u0 (RGB) ==="
python "${U0_MAIN}" --best-json "${GEOM_R}" --outdir "${OUT_U0_R}" --tag "_R" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"
python "${U0_AVSB}" --best-json "${GEOM_R}" --outdir "${OUT_U0_R}" --tag "_R" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"

python "${U0_MAIN}" --best-json "${GEOM_G}" --outdir "${OUT_U0_G}" --tag "_G" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"
python "${U0_AVSB}" --best-json "${GEOM_G}" --outdir "${OUT_U0_G}" --tag "_G" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"

python "${U0_MAIN}" --best-json "${GEOM_B}" --outdir "${OUT_U0_B}" --tag "_B" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"
python "${U0_AVSB}" --best-json "${GEOM_B}" --outdir "${OUT_U0_B}" --tag "_B" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"

echo "=== (3) Heatmap K(lambda, k_parallel) (RGB) ==="
python "${HEATMAP}" --best-json "${GEOM_R}" --outdir "${OUT_HM_R}" --tag "_R" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"
python "${HEATMAP}" --best-json "${GEOM_G}" --outdir "${OUT_HM_G}" --tag "_G" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"
python "${HEATMAP}" --best-json "${GEOM_B}" --outdir "${OUT_HM_B}" --tag "_B" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"

echo "=== (4) Compute xmax from R geometry (sum of cavity stack thickness) ==="
XMAX_R_NM="$(
python - "${GEOM_R}" <<'PY'
import json, sys
p = sys.argv[1]
j = json.load(open(p, "r", encoding="utf-8"))
d = j["d_best_nm"]
order = ["pHTL", "Rprime", "HTL", "EBL", "EML", "ETL"]
print(sum(float(d[k]) for k in order))
PY
)"
echo "R-based xmax_nm = ${XMAX_R_NM}"

echo "=== (5) Bottom-metal |E|^2 profiles with common xmax (R) ==="
python "${BOTTOM_METAL}" --best-json "${GEOM_R}" --outdir "${OUT_BM_R}" --tag "_R" --xmax-nm "${XMAX_R_NM}"
python "${BOTTOM_METAL}" --best-json "${GEOM_G}" --outdir "${OUT_BM_G}" --tag "_G" --xmax-nm "${XMAX_R_NM}"
python "${BOTTOM_METAL}" --best-json "${GEOM_B}" --outdir "${OUT_BM_B}" --tag "_B" --xmax-nm "${XMAX_R_NM}"

echo "=== Done ==="
echo "Writer outputs are under:"
echo "  $(dirname "${GEOM_R}")"
echo "  $(dirname "${GEOM_G}")"
echo "  $(dirname "${GEOM_B}")"
