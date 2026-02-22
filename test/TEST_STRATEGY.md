# ver17 テスト戦略（PyMoosh 参照を最大限利用）

本ドキュメントは **ver17** をベースに、`run_rgb_full_eval.sh` で使用している主要スクリプト群および共通ライブラリ `./gpvm/*.py` を対象に、  
(1) TMM、(2) BT境界条件、(3) 境界での |E|^2 の k 依存性、(4) Fresnel、(5) rewrap を **できる限り PyMoosh を参照(ref)** にして検証する方法を、**省略せず**に整理したものです。

対象（run_rgb_full_eval.sh の構成）
- WRITER: `run_phase3_opt_then_gpvm_eml_profile.py`
- U0_MAIN: `gpvm_k_lambda_u0/run_gpvm_K_lambda_u0.py`
- U0_AVSB: `gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py`
- HEATMAP: `gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py`
- BOTTOM_METAL: `run_bottom_metal_PEC_proxy.py`
- 共通ライブラリ: `./gpvm/*.py`
- rewrap/BT: `tmm_rewrap_utils_policyB.py`, `oled_cavity_phase3p1_policyB.py`

---

## 全体方針

### テストを3層に分ける（破局点の局所化）
1. **Unit（式・局所）**  
   Fresnel、界面行列、伝搬行列、sqrt枝（Im(kz)≥0）、TM符号規約など「局所」な仕様を固定する。
2. **Component（サブシステム）**  
   単一スタックの反射係数 r(λ,u,pol)（TMM vs 内部S行列 vs PyMoosh）、
   BT端反射（rb/rt）、
   rewrap の ru/rd、proxy E2 など「部品」単位で一致を確認する。
3. **Integration（最終出力）**  
   `run_rgb_full_eval.sh` 相当のスクリプト（WRITER/u0/avsb/bottom-metal/heatmap）を
   小さい設定で “動くこと＋最小限の整合” を確認する（いわゆる smoke/integration）。

---

## 比較の型（判定基準・誤差）

### 複素量の比較（r, t, ru/rd など）
- `max_abs = max(|x - x_ref|)`
- `rel = max(|x - x_ref| / max(1, |x_ref|))`
- 角度比較が必要な場合は `unwrap(angle(x))` の差も見る。

### 実数スペクトルの比較（K(λ), proxy E2(z) など）
- `max_abs_diff`
- `rel_l2 = ||a-b||_2 / max( ||b||_2, eps )`
- `argmax` の差（ピーク位置差）や FWHM の差（ただし端点ピークは FWHM が NaN になり得る）

### 許容誤差（目安）
- Fresnel（同一規約での r）：`1e-12`（機械精度）
- terminal stack reflection r（PyMoosh ref）：`1e-10`〜`1e-12`（枝固定や正規化の微差を見込む）
- proxy E2（同一式の2表現比較）：`1e-12`〜`1e-10`
- 2D map など重いもの：`1e-6`〜`1e-4`（補間や正規化、描画の影響を見込む）

---

## 参照(ref)としての PyMoosh の使い方（基本ルール）

### PyMoosh に渡す材料は ε=n² として扱う
ver17 の方針と同じく、
PyMoosh `Structure(mats=...)` の scalar は **屈折率 n ではなく permittivity ε と解釈**されるため、
常に `eps = n**2` を渡す。

### u → θ 変換は枝を固定したものを使う
- `theta_from_u_branch_fixed(n_inc, u)` を必ず使い、evanescent でも規約が揃うようにする。

### “参照面” を混ぜない
- terminal reflection（端面反射）  
  `pHTL|ITO` seen from pHTL（B側）, `ETL|cathode1` seen from ETL（T側）
- source-plane 参照（GPVMのSA/SBで要求される rA/rB）  
  terminal を source plane に移すときは位相因子が必要

これを混ぜると正しくてもズレるので、テストは必ず **どの参照面の量か**を明記して比較する。

---

# (4) Fresnel の検証（最優先）

Fresnel がズレると、TMM/GPVM/rewrap が全てズレるため最優先で固める。

## 4-1. “rewrap規約” の Fresnel を PyMoosh 参照で一致させる（Unit）
`./tmm_rewrap_utils_policyB.py` 内の `_interface_r()` は PyMoosh の規約に合わせている：

- TE: `b = kz`
- TM: `b = kz/eps`（eps=n²）
- `r = (b1 - b2)/(b1 + b2)`

この `_interface_r()` の r が、PyMoosh `coefficient_S` で “界面だけ” を計算した r と一致することを確認する。

**チェック**：
- n1,n2 を実数・複素（吸収あり）で複数
- u = 0, 0.5, 0.9
- λ = 数点（例 450, 550, 650 nm）
- TE/TM それぞれで `max|Δr| < 1e-10` 程度

## 4-2. “GPVM論文規約” Fresnel を自己整合で固める（Unit）
`./gpvm/fresnel.py` は GPVM論文 Eq.(3) を実装し、
TM反射に **追加マイナス**が入る規約を採用している。

- TE：教科書式と同じ
- TM：`r_TM = - (kz_j/n_j^2 - kz_k/n_k^2) / (kz_j/n_j^2 + kz_k/n_k^2)`

ここは PyMoosh 規約と一致させる必要はない（規約が違う）ので、
以下をテストする：
- TE：rewrap規約と一致
- TM：rewrap規約と “符号反転” で一致（`r_gpvm ≈ -r_rewrap`）

---

# (1) TMM（transfer matrix）の検証

## 1-1. TMMスタック反射 r を PyMoosh 参照で一致（Component）
対象：`gpvm/system_matrix.stack_transfer_matrix()` と `rt_from_transfer_matrix()` による r。

ただし gpvm は u の定義が `k_parallel = n_e k0 u` なので、PyMoosh と比較する際は
- 可能なら `n_e = 1` を採用して `u_gpvm = u_pymoosh` とする（単体テスト）
- あるいは `u_pymoosh = n_e * u_gpvm` に変換して比較する

**チェック**：
- 2層程度の単純スタック（incident/layer/substrate）
- TE：`r_gpvm ≈ r_pymoosh`
- TM：GPVM規約により `r_gpvm ≈ -r_pymoosh`（“全体”として符号反転になることを確認）

※ ver17では transfer matrix の overflow 対策として
- `gpvm/matrices.layer_matrix` のスカラー正規化
- `gpvm/system_matrix._renorm` の段階正規化
が入っているため、TE/TM双方で NaN/Inf が出ないことも合わせて確認する。

---

# (2) BT境界条件（rb_used, proxy E2 の整合）

## 2-1. terminal reflections（rb_raw, rt）の一致（Component）
対象：`tmm_rewrap_utils_policyB.terminal_reflections_BT()`（本番：内部S行列）と  
`terminal_reflections_BT_from_pymoosh()`（参照：PyMoosh）

比較量：
- `rb_raw(λ,u,pol)`（pHTL|ITO seen from pHTL）
- `rt(λ,u,pol)`（ETL|cathode1 seen from ETL）

**チェック**：
- λ: 数点（例 450, 550, 650）
- u: 0.0, 0.7
- pol: TE/TM
- `max|Δ| < 1e-10` 程度

## 2-2. rb_used の位相ポリシー（Unit）
`phi_b_mode="pi"` のとき、
- `rb_used = -|rb_raw|` が **常に実数負**になること
- `arg(rb_used) ≈ π` になること（数値誤差を許容）

## 2-3. proxy E2(z) の2表現一致（Integration寄り）
phase3p1 の最適化で使っている “未正規化 proxy” は
- `green_profile_eq66_from_BT(...)` が返す `E2_raw`
- `ru_rd_from_BT_at_z(...)` → `green_terms_from_ru_rd(...)` が返す `E2_raw`

の2通りで評価でき、ver17コードでは両者は一致するはず。

**チェック**：
- 同一の (rb_used, rt, zopt, LBT, λ) を与えたとき
- `max|ΔE2_raw(z)|` が十分小さい（例 `1e-10`）

さらに、最適化の目的関数が
- `peak_err = |z_peak - z_eml_center|`
であることを担保するため、best候補で
- `z_peak` が EML領域内（境界〜境界）にあることも確認する。

---

# (3) 境界での |E|^2 の k 依存性（bottom metal PEC proxy）

対象：`run_bottom_metal_PEC_proxy.py`

ここは PyMoosh と “電場プロファイル” を直接一致させるより、
- `r(k)` の挙動（PyMooshで裏取り可能）
- `|E0|/max|E_eml|` の単調改善（物理妥当性）
を確認するのが現実的。

## 3-1. terminal r(k) の挙動を PyMoosh 参照で裏取り（Component）
bottom ambient を `n = 0.14 + i k` としたとき、
- `|1 + r|` が k 増加で小さくなる傾向があること（完全単調でなくても大域的改善）

## 3-2. スクリプトの主要指標 ratio(k) の妥当性（Integration）
`ratio(k) = |E(z=0)| / max|E(EML)|` を、kを増やすと改善することを確認。

最低限の判定例：
- `ratio(k_high) < ratio(k_low)`（例えば k=2000 と k=50 の比較）
- NaN/Inf が発生しない
- 閾値 `1e-3` を満たす k が存在する場合は、その k が妥当な範囲（数百〜数千）に入る

---

# (5) rewrap の検証（PyMoosh参照を最大限）

rewrap の混乱源は
- 参照面（terminal / EML境界 / source-plane）
- TM符号規約
- sqrt枝の取り方
なので、段階的に一致を取る。

## 5-1. terminal r の一致（Component）
すでに (2-1) の terminal reflections で担保。

## 5-2. rewrap 計算 ru/rd の整合（Component）
terminal の rB, rT を given として
- `wrap_reflection_from_terminal(...)` により EML側へ rewrap した反射
- `effective_ru_rd_at_S_from_BT(...)` により source plane S 参照の ru/rd
を計算し、同じ式を別経路で計算した際に矛盾しないことを確認する。

## 5-3. proxy E2 の end-to-end 一致（Integration）
(2-3) と同じ：
- ru/rd → E2
- Eq66形 → E2
が一致すれば、rewrap + BT境界 + proxy の end-to-end が担保される。

## 5-4. source-plane 参照変換の再発防止（Unit）
terminal reflection を source plane に移す場合、位相因子が必要：

- `rA_src = rB_term * exp(i 2 k0 z_ex_opt)`
- `rB_src = rT_term * exp(i 2 k0 (LBT - z_ex_opt))`

この変換を “つい忘れる” のが典型バグなので、**小さなケースで式通りであること**をテストに入れて再発防止する。

---

## 実装方針（testディレクトリに置く構成）

ver17 の下に `test/` を置き、その中で
- `python run_all_tests.py`
で全テストが走るようにする。

### テストカテゴリ
- `test_fresnel_rewrap_vs_pymoosh.py`（(4) rewrap Fresnel）
- `test_gpvm_fresnel_sign_convention.py`（(4) gpvm Fresnel TM符号）
- `test_terminal_reflections_BT_vs_pymoosh.py`（(2) terminal reflections）
- `test_gpvm_transfer_matrix_vs_pymoosh.py`（(1) TMM）
- `test_phase3_proxy_E2_consistency.py`（(2)(5) proxy）
- `test_bottom_metal_k_dependence.py`（(3) k依存）
- `test_scripts_smoke.py`（run_rgb_full_eval.sh の主要スクリプトを小さめ設定で smoke 実行）

### PyMoosh 参照の使い方
- ver17 同梱の `PyMoosh-stable/` を import パスに追加して使う
- 参照比較では `coefficient_S` を優先（安定）

---

## 優先順位（どこから固めるべきか）
1. Fresnel（rewrap規約） vs PyMoosh interface
2. terminal reflections（BT） vs PyMoosh
3. proxy E2（最適化で使う量）の2表現一致
4. TMM stack r vs PyMoosh（TE、TMは符号規約も含めて）
5. bottom metal の ratio(k) の妥当性
6. スクリプト smoke（最小設定で動作確認）

以上。
