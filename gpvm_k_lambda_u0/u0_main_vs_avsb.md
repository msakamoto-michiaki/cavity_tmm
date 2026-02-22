# run_gpvm_K_lambda_u0.py（U0_MAIN）と run_gpvm_K_lambda_u0_A_vs_B.py（U0_AVSB）の違い（出力・モデル・目的）

本書は `run_rgb_full_eval.sh` で同じ `--best-json` を与えているにもかかわらず、

```bash
python "${U0_MAIN}" --best-json "${GEOM_R}" --outdir "${OUT_U0_R}" --tag "_R" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"
python "${U0_AVSB}" --best-json "${GEOM_R}" --outdir "${OUT_U0_R}" --tag "_R" --lam-span-nm "${LAM_SPAN}" --n-lam "${N_LAM}"
```

と **2本のスクリプトを両方実行している理由**を、**計算モデル／目的／出力**の観点で省略なく説明するものです。

---

## 1. まず結論（役割分担）

- **U0_MAIN（`run_gpvm_K_lambda_u0.py`）**  
  strict-EML（物理スタック由来の反射係数）で **u=0 の K(λ) を“成分分解つき”で定量計算**し、解析に使える `*.npy` と図を出す「本命の定量出力」。

- **U0_AVSB（`run_gpvm_K_lambda_u0_A_vs_B.py`）**  
  **A=phase3 BT-proxy** と **B=strict-EML** の2モデルで u=0 の K(λ) を計算し、**AとBのズレを1枚の図で比較**する「監視・検証」用の出力。

同じ geometry (`--best-json`) でも、**評価するモデル（境界条件）**と**出力粒度（成分の扱い）**が違うため、両方を回す意味があります。

---

## 2. U0_MAIN：run_gpvm_K_lambda_u0.py の中身

### 2.1 目的
strict-EML（物理 stack）に基づく **u=0 の K(λ) を GPVM式（Eq.26–30）で計算**し、偏光・配向成分まで含めて保存・可視化する。

「この geometry に対して、物理スタックモデルでは K がどういうスペクトルを持つか」を定量的に得るのが目的です。

### 2.2 計算モデル（strict-EML）
- best geometry（`d_best_nm`）から **left / right stack** を構成する（EML界面で分割）
- 各 λ で、u=0 固定のまま
  - TE/TM 別に transfer（または安定化した行列）から **rA(λ), rB(λ)** を抽出
  - `kz_e(λ,u=0)` を計算
  - GPVM 論文の式
    - `Pe_TE_h_eq26`
    - `Pe_TM_h_eq27`
    - `Pe_TM_v_eq28`
    - `K_from_Pe_eq30`
    を用いて K を得る

### 2.3 出力（典型）
U0_MAIN は「解析に使えるように」**成分別の配列をすべて保存**し、まとめ図を作ります。

- `lam_nm*.npy`
- 成分：
  - `K_TE_h*.npy`
  - `K_TM_h*.npy`
  - `K_TM_v*.npy`
- “only” と isotropic（コード規約）：
  - `K_TE_only*.npy`（通常 `(2/3)K_TE_h`）
  - `K_TM_h_only*.npy`（通常 `(2/3)K_TM_h`）
  - `K_TM_v_only*.npy`（通常 `(1/3)K_TM_v`）
  - `K_iso*.npy`（上記の和）
- 図：
  - `gpvm_K_lambda_u0_components_strict*.png`  
    成分ごとの K(λ) が同一図に載る（正規化/未正規化は実装に依るが、基本は“成分全体が見える”図）。

**要するに**：strictモデルの u=0 スペクトルを **全部の成分で**揃えて出す「本体の定量出力」です。

---

## 3. U0_AVSB：run_gpvm_K_lambda_u0_A_vs_B.py の中身

### 3.1 目的
同じ geometry に対して

- **A（phase3 BT-proxy）**
- **B（strict-EML）**

の2モデルで u=0 の K(λ) を計算し、**2本の曲線を同じ図で比較**して「ズレ」を可視化する。

U0_MAIN は strict の単独評価ですが、U0_AVSB は **phase3最適化（BT-proxy）と strict（物理）でどれくらい差があるか**を毎回見張るためのスクリプトです。

### 3.2 計算モデル A：phase3 BT-proxy
- `terminal_reflections_BT(...)` により BT端反射
  - `rb_used(λ)`：bottom側（位相ポリシー適用後）
  - `rt(λ)`：top側
  を得る（u=0、TE/TMの選択あり）
- `best-json` に入っている phase3由来の光学座標パラメータ
  - `z_ex_opt`
  - `LBT_opt`
  を用い、BT-proxy の式に投入する

この A モデルは「phase3p1 の proxy と同じ境界条件・同じ参照面」を再現するためのものです。

### 3.3 計算モデル B：strict-EML
- U0_MAIN と同様に、EML界面で left/right stack を組み、rA/rB を抽出して GPVM式へ投入。

### 3.4 成分の扱い（簡略化：TE×h）
U0_AVSB は比較が目的なので、多くの実装では

- **TE×h（水平双極子）だけ**を計算し
- isotropic は簡略近似として  
  `K_iso ≈ (2/3) K_TE_h`
として扱います。

理由：
- A vs B のズレ確認には、まず最も代表的な成分（TE×h）で十分なことが多い
- TE/TM×(h,v) を全部載せると図や出力が増え、比較の主目的がぼける

（必要なら拡張可能ですが、出力増・計算増になります。）

### 3.5 出力（典型）
- 比較図：
  - `gpvm_K_lambda_u0_A_vs_B*.png`  
    A（BT-proxy）と B（strict）の曲線を同一図に描画。
    多くの場合、`best-json` の `lambda_nm` に縦線を引きます。
- 保存する配列は必要最小：
  - `lam_nm*.npy`
  - `K_A_iso*.npy` / `K_B_iso*.npy` など（実装により命名は多少違う）

**要するに**：strict の全成分出力が欲しいわけではなく、**BT-proxyとstrictの差を監視する比較プロット**です。

---

## 4. run_rgb_full_eval.sh が両方回す意味

`run_rgb_full_eval.sh` の観点では、

- `U0_MAIN`：strict-EML の **成分付きスペクトル（保存・解析の本体）**
- `U0_AVSB`：phase3 BT-proxy と strict-EML の **ズレ監視（検証）**

という役割分担になっています。

### 実務的な使い分け
- **解析・後段処理に使う数値**が欲しい → U0_MAIN が必要
- **最適化モデル（phase3 proxy）と strict（物理）が整合しているか**を毎回確認したい → U0_AVSB が必要

---

## 5. どちらか片方だけにする場合の判断

- 「strictだけでよい／BT-proxy比較はいらない」  
  → `U0_AVSB` を外す

- 「BT-proxyとstrictの一致だけ見たい／成分分解は不要」  
  → `U0_MAIN` を外して `U0_AVSB` だけ  
  （ただし TM成分等は見えず、ズレの原因分析には弱くなる）

---

## 6. 補足：なぜ U0_AVSB は Eq.(26) と Eq.(30) を合成していたのか

（ver19以降では `gpvm/eqs_gpvm.py` に共通化して wrapper 化しているが、意図の説明）

- U0_AVSB は比較目的で、途中量 `P_e` を保存する必要が薄い
- u=0固定なら Eq.(30) は `K = π P_e` の定数倍なので、関数を分ける利点が小さい

このため、過去実装では `K_TE_h_eq26()` として **Eq.(26)→Eq.(30) を一体化**していました。  
ただし共通化の観点では分離した方が良いため、現在は

- `Pe_TE_h_eq26(...)`
- `K_from_Pe_eq30(...)`

を呼ぶ薄い wrapper に寄せるのが安全です。

---

## 7. まとめ（短い要点）

- U0_MAIN：strict-EML の **成分別 K(λ)** を保存・可視化する「定量の本体」
- U0_AVSB：BT-proxy と strict-EML の **差を比較**して可視化する「検証・監視」

同じ geometry を渡しても、「モデル・目的・出力」が違うため、両方を実行する意味があります。
