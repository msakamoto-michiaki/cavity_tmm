# ver8 strict-EML full polarization decomposition (K_iso / TE_only / TM_h_only / TM_v_only)

この ver8 ベースでは、strict-EML GPVM の K を以下の成分に分解して出力します。

- K_TE_h(λ,k//)   : TE, horizontal dipole  (Eq.26+Eq.30)
- K_TM_h(λ,k//)   : TM, horizontal dipole  (Eq.27+Eq.30)
- K_TM_v(λ,k//)   : TM, vertical dipole    (Eq.28+Eq.30)

等方（isotropic dipole）への重み付け:

- K_iso       = (2/3)K_TE_h + (2/3)K_TM_h + (1/3)K_TM_v
- K_TE_only   = (2/3)K_TE_h
- K_TM_h_only = (2/3)K_TM_h
- K_TM_v_only = (1/3)K_TM_v

## 実行

repo root（`out_gpvm_step2/` があるディレクトリ）で:

```bash
# まず geometry を作る（必要なら）
python run_phase3_opt_then_gpvm_eml_profile.py

# u=0 の 1D スペクトル（成分分解）
python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0.py

# K(λ,k//) heatmap（成分分解）
python gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py
```

## 出力

### u=0
`gpvm_k_lambda_u0/`
- `gpvm_K_lambda_u0_components_strict.png`
- `K_iso.npy, K_TE_only.npy, K_TM_h_only.npy, K_TM_v_only.npy` など

### heatmap
`gpvm_K_lambda_kpar_map_strict/`
- `heatmap_lambda_x_kpar_y__K_iso__linear.png` など（K_iso/TE_only/TM_h_only/TM_v_only それぞれ linear/log, 2レイアウト）
- `check_k0_matches_u0_peak_fwhm__K_iso.png`
- `K_iso_map.npy`, `K_TE_only_map.npy`, `K_TM_h_only_map.npy`, `K_TM_v_only_map.npy` など

## 注意
- この実装は **bottom ITO を PEC-like metal (0.14 + 2000 i)** に置換した条件です（ver8 の既存設定に合わせています）。
- k// 境界線は air/substrate/EML/WGP と、SPP proxy（ETL/cathode1）を重ね描きします。
