tar -xzf phase3p1_gpvm_k_lambda_full_with_figs_v1.tar.gz
cd phase3p1_pyMoosh_Splane_TMMrewrap

# (A) proxy vs (B) strict-EML の u=0 比較図
python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py

# strict-EML の K_iso(λ,k//) heatmap + k//=0 の peak/FWHM 一致チェック
python gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py
