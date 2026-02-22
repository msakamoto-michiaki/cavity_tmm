# OLED Cavity 仕様書（verifyTM記法に統一・コード再監査版）

## 0. 目的と今回の修正点

本書は、`phase2p7.tar` と `verify.tar` の**実コード**を照合し、  
記法を `verifyTM.tex` の流儀（B/T面定義式 + rewrap写像式）に統一した仕様書です。

今回の主修正:
- Bottom / Top の「巻き戻し層」と「終端反射の定義面」を厳密に分離して記述
- `verifyTM.tex` と同じ比較式
  \[
  r_{\mathrm{rewrap,bot/top}}^{(\sigma)} \stackrel{?}{=} r_{\mathrm{full,bot/top}}^{(\sigma)}
  \]
  で統一
- 全モジュール再点検の結果を反映（記述ミスの有無、実装上の注意点を明記）

---

## 1. 記法（verifyTM.texと同一流儀）

偏光 \(\sigma\in\{s,p\}\)、  
\(u=k_\parallel/k_0,\;k_0=2\pi/\lambda\)。

PyMoosh整合規約（\(\mu=1,\;\varepsilon=n^2\)）:
\[
k_{z,j}=k_0\sqrt{\varepsilon_j-u^2},\qquad \Im(k_{z,j})\ge 0
\]
\[
b_j^{(s)}=k_{z,j},\qquad b_j^{(p)}=\frac{k_{z,j}}{\varepsilon_j}
\]
\[
r_{j,k}^{(\sigma)}=\frac{b_j^{(\sigma)}-b_k^{(\sigma)}}{b_j^{(\sigma)}+b_k^{(\sigma)}}
\]

再帰演算子 \(\mathcal W\)（端面反射を上側へ写像）:
\[
\mathcal W\!\left(n_{\mathrm{inc}};\{(n_1,d_1),\dots,(n_N,d_N)\};r_{\mathrm{term}}^{(\sigma)}\right)
\]
を
\[
r^{(N)}=r_{\mathrm{term}}^{(\sigma)},
\]
\[
r^{(j-1)}
=\frac{r_{j-1,j}^{(\sigma)}+r^{(j)}e^{2ik_{z,j}d_j}}
{1+r_{j-1,j}^{(\sigma)}r^{(j)}e^{2ik_{z,j}d_j}}
\quad (j=N,\dots,1)
\]
で定義し、最終値 \(r^{(0)}\) を返す。

---

## 2. B/T/S面の厳密定義

- **B面（Bottom終端反射の定義面）**: \(pHTL|ITO\)（pHTL側から見た反射）
- **T面（Top終端反射の定義面）**: \(ETL|cathode1\)（ETL側から見た反射）
- **S面**: EML中心

> 注: 「B=ITO/pHTL」「T=ETL/cathode1」は境界名の略記であり、  
> 反射係数の入射側は上記のとおり **pHTL側 / ETL側** で定義される。

---

## 3. Bottom / Top の rewrap と full-stack 比較式（verifyTM統一）

## 3.1 Bottom

終端反射:
\[
r_B^{(\sigma)}(u,\lambda)
=
r^{(\sigma)}\!\left(
\mathrm{incident}=pHTL,\;
\mathrm{layers}=[ITO,anode],\;
\mathrm{substrate}=substrate
\right).
\]

rewrap（EML|EBL面へ）:
\[
r_{\mathrm{rewrap,bot}}^{(\sigma)}(u,\lambda)
=
\mathcal W\!\left(
EML;\;
[(EBL,d_{EBL}),(HTL,d_{HTL}),(Rprime,d_{Rprime}),(pHTL,d_{pHTL})];\;
r_B^{(\sigma)}
\right).
\]

full-stack:
\[
r_{\mathrm{full,bot}}^{(\sigma)}(u,\lambda)
=
r^{(\sigma)}\!\left(
\mathrm{incident}=EML,\;
\mathrm{layers}=[EBL,HTL,Rprime,pHTL,ITO,anode],\;
\mathrm{substrate}=substrate
\right).
\]

検証条件:
\[
r_{\mathrm{rewrap,bot}}^{(\sigma)}
\stackrel{?}{=}
r_{\mathrm{full,bot}}^{(\sigma)}.
\]

## 3.2 Top

終端反射:
\[
r_T^{(\sigma)}(u,\lambda)
=
r^{(\sigma)}\!\left(
\mathrm{incident}=ETL,\;
\mathrm{layers}=[cathode1,cathode,CPL],\;
\mathrm{substrate}=air
\right).
\]

rewrap（EML|ETL面へ）:
\[
r_{\mathrm{rewrap,top}}^{(\sigma)}(u,\lambda)
=
\mathcal W\!\left(
EML;\;
[(ETL,d_{ETL})];\;
r_T^{(\sigma)}
\right).
\]

full-stack:
\[
r_{\mathrm{full,top}}^{(\sigma)}(u,\lambda)
=
r^{(\sigma)}\!\left(
\mathrm{incident}=EML,\;
\mathrm{layers}=[ETL,cathode1,cathode,CPL],\;
\mathrm{substrate}=air
\right).
\]

検証条件:
\[
r_{\mathrm{rewrap,top}}^{(\sigma)}
\stackrel{?}{=}
r_{\mathrm{full,top}}^{(\sigma)}.
\]

---

## 4. S面（EML中心）への写像とFP proxy

EML境界反射を \(r_{EML/EBL}^{(\sigma)},\, r_{EML/ETL}^{(\sigma)}\) とすると  
（上の bottom/top rewrap の出力）、EML中心 \(S\) への位相移送は
\[
\Delta z = d_{EML}/2,\qquad
k_{z,EML}=k_0\sqrt{\varepsilon_{EML}-u^2},
\]
\[
r_u^{(S,\sigma)}=r_{EML/EBL}^{(\sigma)}e^{2ik_{z,EML}\Delta z},\qquad
r_d^{(S,\sigma)}=r_{EML/ETL}^{(\sigma)}e^{2ik_{z,EML}\Delta z}.
\]

FP代替関数:
\[
F^{(\sigma)}(\lambda,u)=
\frac{|1+r_u^{(S,\sigma)}|^2}{|1-r_u^{(S,\sigma)}r_d^{(S,\sigma)}|^2}.
\]

---

## 5. モジュール別仕様（再監査反映）

## 5.1 (1) `tmm_rewrap_utils.py`

### 物理・数学モデル
- 上記 \(\mathcal W\) をそのまま実装
- TE/TM は \(b=k_z,\;k_z/\varepsilon\) 規約
- \(k_z\) の枝は \(\Im(k_z)\ge 0\) で固定

### 入出力
- 入力: `n0, d, lam_nm, u, rB_pol, rT_pol, pol`
- 出力: `(ru_S, rd_S)` と `F`

### Bottom/Topの実装
- Bottom再帰層: `[EBL, HTL, Rprime, pHTL]`
- Top再帰層: `[ETL]`
- ここで `rB_pol` / `rT_pol` は**終端面で既に定義された反射**を受け取る

---

## 5.2 (2) `oled_cavity**`

対象:
- `oled_cavity_pymoosh_sourceplane_FP_finetune.py`
- `oled_cavity_pymoosh_sourceplane_FP_finetune_compat_TMMrewrap.py`

### 数学モデル
- \(\lambda_0=650\) nm で位相条件から HTL+Rprime 比率 \(s\) を算出
\[
L_{\mathrm{target}}=m\frac{\lambda_0}{2}
-\frac{\lambda_0}{4\pi}\bigl(\phi_b+\phi_e\bigr),\qquad \phi_b=\pi.
\]
\[
s=\frac{L_{\mathrm{target}}-L_{\mathrm{fixed}}}{L_{\mathrm{unit}}}.
\]
- \(\phi_b=\pi\) は `r_B\leftarrow-|r_B|` で強制
- ETL粗探索（1 nm）→細探索（0.1 nm）
- 指標: EML中心と \(|E|^2\) ピーク位置の差

### 実装方式の違い
- `finetune.py`:
  - B/T参照の位相シフト式（`F_BT` と理論等価な `F_S`）
- `compat_TMMrewrap.py`:
  - `effective_ru_rd_at_S_from_BT` で再帰TMM写像後に \(F\) を評価

### 入出力
- CSV: coarse scan, best summary
- PNG: \(F(\lambda)\), \(|E|^2(z)\)
- NPY: `Splane_FP_compat_*` など

### 監査メモ
- `compat_TMMrewrap.py` 細探索ループに early return（互換挙動）が残存  
  （最初の有効候補で `return`）

---

## 5.3 (3) `reproduce**`

対象:
- `reproduce_phase2_step2_u_dependence_pymoosh_sourceplane_FP_TMMrewrap.py`

### 数学モデル
- \(F_{TE/TM}(\lambda,u)\) をS面再帰TMMで評価
- 正規化後に `K_h, K_v, K_iso=(2/3)K_h+(1/3)K_v` を生成
- Fig.4形式で \(k_\parallel\)-\(\lambda\) マップ化
\[
u_{\mathrm{req}}=
\frac{\lambda[\mathrm{nm}]\times 10^{-9}}{2\pi}k_\parallel.
\]

### 実装
- `u -> θ` は枝固定関数 `theta_from_u_branch_fixed` を使用
- B/T面反射はPyMooshで計算し、`effective_ru_rd_at_S_from_BT` でSへ写像
- 既定では `K_h=F_{TE}`, `K_v=10^{-10}`（簡略版）

### 入出力
- NPY: `F_TE.npy`, `F_TM.npy`, `K_h.npy`, `K_v.npy`
- CSV: `lambda_grid.csv`, `u_grid.csv`, `thickness_used.csv`
- PNG: heatmap群, `fig4_style_Kiso_map.png`
- 整合チェックON時: `consistency_check_*`

---

## 5.4 (4) `verify`（bottom/top）

対象:
- `tmm_rewrap_utils_verify.py`
- `test_verify_bottom_rewrap_vs_fullstack_v3.py`
- `test_verify_top_rewrap_vs_fullstack_v3.py`

### 検証内容
- 第3章の比較式を \((\lambda,u)\) 2Dで実行
\[
\Delta r=|r_{\mathrm{rewrap}}-r_{\mathrm{full}}|.
\]
- `log10(|Δr|)` ヒートマップと最大誤差を出力

### 実行結果（実測）
- Bottom: `max|Δr| TE = 3.333e-15`, `TM = 1.355e-15`
- Top: `max|Δr| TE = 1.746e-14`, `TM = 4.220e-14`

---

## 6. 「他モジュールも含めた」再監査結論

1. **Bottom/Top の巻き戻し構造は verify と整合**  
   - Bottom: 終端 `r_B` を作ってから `[EBL,HTL,Rprime,pHTL]` で写像  
   - Top: 終端 `r_T` を作ってから `[ETL]` で写像

2. **前回問題は“実装ミス”ではなく“仕様書の省略表現”**  
   - 「EML→ETL→cathode1 を再帰」と書くと、  
     `cathode1`まで再帰しているように読める  
   - 正確には「`r_T` は ETL|cathode1 で定義済み、再帰層は ETL のみ」

3. **実装上の注意（仕様書に残すべき点）**
   - `compat_TMMrewrap.py` の fine-scan early return
   - `reproduce**` の `K_h/K_v` が簡略化実装

---

## 7. 実行コマンド

```bash
# verify
python test_verify_bottom_rewrap_vs_fullstack_v3.py
python test_verify_top_rewrap_vs_fullstack_v3.py

# phase2p7
python oled_cavity_pymoosh_sourceplane_FP_finetune.py
python oled_cavity_pymoosh_sourceplane_FP_finetune_compat_TMMrewrap.py
python reproduce_phase2_step2_u_dependence_pymoosh_sourceplane_FP_TMMrewrap.py
```
