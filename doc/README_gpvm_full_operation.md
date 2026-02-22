# phase3p1_gpvm_k_lambda_full_with_figs — 実行フローと出力の説明

このドキュメントは、同梱 tar（`phase3p1_gpvm_k_lambda_full_with_figs_v1.tar.gz`）を展開した後の
`phase3p1_pyMoosh_Splane_TMMrewrap/` 以下で、**geometry の生成（phase3 最適化）を含め**、
各プログラムが **どのように動作し、何を出力するか**をまとめたものです。

---

## 最短の実行手順（推奨）

```bash
tar -xf ver11.tar
cd ver10

# 1) geometry（phase3 最適化）と EML 内部プロファイル（phase3-BT と strict-EML）の確認
python run_phase3_opt_then_gpvm_eml_profile.py --lambda-nm 650

# 2) K(λ, u=0) の比較図（A: phase3-BT proxy vs B: strict-EML GPVM）
python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py

# 3) strict-EML の K_iso(λ,k//) heatmap + k//=0 でピーク/FWHM 一致チェック
python gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py
```

> **注意**: tar には、(2)(3) の図と `.npy` キャッシュが同梱されています。  
> 同じ条件なら、(3) は通常「キャッシュ読込→図再生成」だけで終わります。条件を変えたら
> `gpvm_K_lambda_kpar_map_strict/*.npy` を削除すると再計算します。

---

## RGB など複数 geometry を扱う（ver11 で追加）

ver11 では、**Writer / Reader すべてに `--best-json` を追加**し、`best_geometry*.json` を任意名で扱えます。

- Writer: `run_phase3_opt_then_gpvm_eml_profile.py`
  - `--best-json` : 出力先 JSON（例: `out_gpvm_step2/best_geometry_R.json`）
  - その **JSON と同じディレクトリ**に、EML プロファイルの `.npy/.png` も書き出します（上書き事故を避けたい場合は色ごとに別ディレクトリ推奨）

- Readers（例）:
  - `gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py --best-json ...`
  - `gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py --best-json ... --outdir ...`
  - `run_bottom_metal_PEC_proxy.py --best-json ... --outdir ...`

### 例：R/G/B を別々に作って別出力へ

```bash
# 1) geometry を色ごとに保存（必要なら波長も色ごとに変える）
python run_phase3_opt_then_gpvm_eml_profile.py --lambda-nm 620 --best-json out_gpvm_step2_R/best_geometry.json
python run_phase3_opt_then_gpvm_eml_profile.py --lambda-nm 530 --best-json out_gpvm_step2_G/best_geometry.json
python run_phase3_opt_then_gpvm_eml_profile.py --lambda-nm 460 --best-json out_gpvm_step2_B/best_geometry.json

# 2) 以降の計算は同じ best-json を指定
python gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py --best-json out_gpvm_step2_R/best_geometry.json --outdir gpvm_k_lambda_u0/out_R
python gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py --best-json out_gpvm_step2_R/best_geometry.json --outdir gpvm_K_lambda_kpar_map_strict/out_R

# bottom metal |E|^2 profile（x軸を R に揃える例は前の回答の bash 参照）
python run_bottom_metal_PEC_proxy.py --best-json out_gpvm_step2_R/best_geometry.json --outdir out_bottom_metal_R --tag _R
```

> **注意（キャッシュ）**: `gpvm_K_lambda_kpar_map_strict` は `--outdir` 内の `*.npy` をキャッシュとして使います。  
> geometry/波長を変えたら、同じ outdir を使い回さず、色ごとに outdir を分けるのが安全です。
---

## 0) このパッケージでの “geometry” の定義

geometry は以下の 3 点セットとして扱います。

### (i) 層構造と厚み（nm）
内部層順序（左→右）：
**substrate | anode | ITO | pHTL | Rprime | HTL | EBL | EML | ETL | cathode1 | cathode | CPL | air**

- 厚み `d_list` は内部層（anode〜CPL）のみが対象
- 両端（substrate, air）は **半無限媒質**

最適厚みは `out_gpvm_step2/best_geometry.json` の `d_best_nm` に保存されます。

### (ii) 屈折率（光学定数）
`oled_cavity_phase3p1_policyB.build_current_base()` の `n0`（分散なしの定数モデル）を使用。

今回の再現条件では **ITO だけ**を **n = 0.14 + 2000 i** に置換（PEC-like metal proxy）します。

### (iii) phase3 “光学座標”と cavity 光学長（proxy用）
phase3 の BT-proxy 用に
- `LBT_opt_nm`（光学長）
- `z_ex_opt_nm`（光学座標上の励起位置）

が `best_geometry.json` に入ります。

strict-EML 計算では、proxy の `zopt/LBT` は使わず、
**EML物理厚 `d_EML` と EML中心 `z_ex=d_EML/2`** を用います。

---

## 1) geometry生成 + EML 内部プロファイル確認

### スクリプト
`run_phase3_opt_then_gpvm_eml_profile.py`

### 目的
1. 既存 phase3p1 の最適化（主に ETL とスケール s）を実行して **best geometry** を決める
2. 同一 geometry で
   - (A) phase3-BT互換 “cavity GPVM profile”
   - (B) strict-EML GPVM profile
   の両方の **EML内部 |E|^2 プロファイル**を出力し、整合確認できるようにする

### 入力（コード内固定）
- `lambda_target_nm = args --lambda-nm (default 650)`
- `u_norm = 0`
- `pol_for_opt = TE`
- `cathode1_nm = 30`

### geometry最適化（中身）
`oled_cavity_phase3p1_policyB.optimize_etl_then_s()` を呼びます。

- ETL を 10–80 nm（1 nm刻み）で走査
- 各 ETL に対して `estimate_s_for_resonance()` で **(HTL, Rprime) スケール係数 s** を推定
- その geometry で phase3 互換の standing-wave |E|^2 プロファイルを作り、
  **ピークが EML 中心に近いもの**を選ぶ（`peak_err`最小）

### 主な出力（`out_gpvm_step2/`）
- `best_geometry.json`
  - `d_best_nm`（最適厚み）
  - `etl_best_nm, s_best`
  - `LBT_opt_nm, z_ex_opt_nm`（proxy用）
  - ほかメタ情報
- phase3-BT互換（光学座標）の EMLプロファイル（正規化）
  - `z_eml_phys_nm.npy`
  - `I_TE_h_phase3BT.npy`, `I_TM_h_phase3BT.npy`, `I_TM_v_phase3BT.npy`, `I_iso_phase3BT.npy`
  - `gpvm_eml_profiles_phase3BT.png`
- strict-EML（物理 EML 境界）の EMLプロファイル（正規化）
  - `z_eml_nm.npy`
  - `I_TE_h_strictEML.npy`, `I_TM_h_strictEML.npy`, `I_TM_v_strictEML.npy`, `I_iso_strictEML.npy`
  - `gpvm_eml_profiles_strictEML.png`

---

## 2) K(λ,u=0)（strict-EML のみ）

### スクリプト
`gpvm_k_lambda_u0/run_gpvm_K_lambda_u0.py`

### 目的
最適 geometry を読み込み、strict-EML 定義で  
**K(λ,u=0)**（TE,h → iso=2/3）を計算して図にする。

### 動作（概略）
1. `best_geometry.json` から `d_best_nm` を読む
2. `build_current_base()` で屈折率 `n0` を作り、**ITO を 0.14+2000i に置換**
3. λ=550–750 nm（1 nm刻み）、u=0 でループ
4. strict-EML の rA,rB を **EML 界面**で定義して TMM で求める（TE）
   - rA: EML から左スタックを見た反射（EML側入射）
   - rB: EML から右スタックを見た反射（EML側入射）
5. GPVMの Eq.(26) + Eq.(30) を評価して K を作る
6. `K_iso = (2/3) K_TE,h` をプロット

### 出力（`gpvm_k_lambda_u0/`）
- `gpvm_K_lambda_u0_PEC_like.png`
- `lam_nm.npy`, `K_te_h.npy`, `K_iso.npy`

---

## 3) K(λ,u=0) の (A)proxy vs (B)strict 比較

### スクリプト
`gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py`

### 目的
(A) **phase3-BT proxy** と (B) **strict-EML** の K(λ,u=0) を同一条件で比較し、
ピーク位置や形状差を見える化する。

### (A) phase3-BT proxy の作り方
- 反射係数は EML 界面ではなく phase3 定義の端：
  - B端：pHTL|ITO（pHTL側から見た端面反射）
  - T端：ETL|cathode1（ETL側から見た端面反射）
- ver15+ では `terminal_reflections_BT()`（内部実装：安定な scattering-matrix）で
  `rb_B(λ), rt_T(λ)` を計算（位相ポリシー込み）。
  PyMoosh による同等計算は `terminal_reflections_BT_from_pymoosh()` として残しており、
  テスト/比較用途にのみ使用する。
- 伝搬は phase3 の光学座標でまとめた
  `LBT_opt_nm`, `z_ex_opt_nm` を用いて Eq.(26)&(30) を評価（「cavity に見立て」）

### (B) strict-EML
(2) と同様に、EML 界面 rA/rB を TMM で求めて Eq.(26)&(30) を評価。

### 出力（`gpvm_k_lambda_u0/`）
- `gpvm_K_lambda_u0_A_vs_B.png`
- `lam_nm.npy`, `K_A_iso.npy`, `K_B_iso.npy`

---

## 4) K(λ,k//) heatmap（strict-EML）

### スクリプト
`gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py`

### 目的
strict-EML 定義で **K_iso(λ,k//)** を計算し、
- heatmap（linear / log）
- **k//=0 の列が strict u=0 と一致**
- k//=0 の **ピーク & FWHM の一致**を図で確認
まで自動生成する。

### グリッド定義（デフォルト）
- λ：550–750 nm（201点）
- u_max：0.99
- k//：λ に依存しない一定グリッド（0〜kpar_max を 240点）
  - `kpar_max = u_max * n_EML * k0(λ_min)`（k0=2π/λ）
- 各 (λ,k//) で `u = k// / (n_EML*k0(λ))` に変換して計算

### 計算の中身（各格子点）
- 左/右スタックから rA,rB を TMM で求める（TE）
- Eq.(26)&(30) → `K_TE,h`
- `K_iso = (2/3)K_TE,h`
- `K_iso_map[λ, k//]` に格納

### 一致チェック（要求仕様）
- `K_iso_map[:,0]`（k//=0 → u=0）を取り出す
- 別途計算した strict `K_iso_u0(λ)` と比較
- 差分、ピーク、FWHM を図に書き込む

### キャッシュ仕様
以下が存在すると再計算せずロードします：
- `K_iso_map_lambda_kpar.npy`
- `lam_nm.npy`
- `kpar_grid_um.npy`
- `K_iso_u0.npy`

### 出力（`gpvm_K_lambda_kpar_map_strict/`）
- heatmap（λ-x / k//-y）
  - `heatmap_lambda_x_kpar_y_linear.png`
  - `heatmap_lambda_x_kpar_y_log.png`
- heatmap（k//-x / λ-y）
  - `heatmap_gpvm_Kiso_linear.png`
  - `heatmap_gpvm_Kiso_log.png`
- k//=0 一致チェック
  - `check_k0_matches_strictB_peak_fwhm.png`
  - `check_u0_vs_k0.png`
- 数値配列
  - `lam_nm.npy`, `kpar_grid_um.npy`
  - `K_iso_map_lambda_kpar.npy`, `K_iso_u0.npy`

---

## 5) どこを変えると何が変わるか（実務メモ）

- **geometry（厚み）を変える**
  - `run_phase3_opt_then_gpvm_eml_profile.py` を再実行 → `best_geometry.json` 更新
  - heatmap 側は `gpvm_K_lambda_kpar_map_strict/*.npy` を削除して再計算
- **ITO→metal をやめる／変更する**
  - 各スクリプトで `n0["ITO"]=...` の置換値を変更
- **strict の source 位置**
  - `z_ex = 0.5*d_EML` を変更（現状は EML 中央）
- **k// の範囲や分解能**
  - heatmap スクリプト内の `u_max`, `kpar_grid` 点数を変更

---

## 付録：ファイル/フォルダの場所

- geometry とプロファイル出力：
  - `out_gpvm_step2/`
- u=0 スペクトル系：
  - `gpvm_k_lambda_u0/`
- heatmap 系：
  - `gpvm_K_lambda_kpar_map_strict/`
