# ver10: geometry JSON の生成・参照マップ（__file__ 基準 / cwd 非依存）

対象: `ver10.tar` を展開したディレクトリ（以下ではそれを **REPO_ROOT** と呼びます）
例: `/workspace/cavity_tmm/phase4/ver8/`

このドキュメントは、

- **どのプログラムが geometry JSON (`best_geometry.json`) を作るか**
- **どのプログラムがそれを読むか**
- **JSON の探索基準が「作業ディレクトリ(cwd)」なのか「スクリプト位置(__file__)」なのか**

を、**ver10 の実コードに基づいて**1本ずつ明記します。

---

## 結論（重要）

ver10 の主要スクリプトは **全て `__file__`（=スクリプト自身の場所）基準**で
`out_gpvm_step2/best_geometry.json` を読み書きします。

したがって、

- `cd` でどこに移動してから実行しても **参照先 JSON は変わりません**
- ただし **同名スクリプトが別フォルダに重複配置**されている場合、  
  それぞれが **自分のフォルダ基準の out_gpvm_step2** を見に行くので、  
  **別の JSON を読んだり、JSON が存在せず落ちたり**します（後述）

---

## 1) geometry JSON を作る（Writer）

### 1.1 `run_phase3_opt_then_gpvm_eml_profile.py`  ✅ Writer（唯一）
**役割**: phase3 の最適化（ETL とスケール s）を行い、best geometry を JSON に保存する。

**出力する JSON**
- `REPO_ROOT/out_gpvm_step2/best_geometry.json`

**パスの決まり方（cwd 非依存）**
- `base = Path(__file__).resolve().parent`
- `out_dir = base / "out_gpvm_step2"`
- そこへ `best_geometry.json` を書き込み

**要点**
- ver10 で `best_geometry.json` を **生成/上書きするのはこの1本だけ**です。

---

## 2) geometry JSON を読む（Readers）

以下は **全て `__file__` 基準**で
`out_gpvm_step2/best_geometry.json` を参照します（cwd を見ません）。

### 2.1 `run_bottom_metal_PEC_proxy.py` ✅ Reader
**役割**: bottom を半無限 metal に置いた診断（LCAV |E|² プロファイル等）を出す。

**読む JSON**
- `REPO_ROOT/out_gpvm_step2/best_geometry.json`

**パスの決まり方（cwd 非依存）**
- `root = os.path.dirname(os.path.abspath(__file__))`
- `root/out_gpvm_step2/best_geometry.json` を読む

**主な出力**
- `REPO_ROOT/out_bottom_metal/gpvm_lcav_profile_bottomMetal_k2000.png`
- `REPO_ROOT/out_bottom_metal/need_k_for_E0_small.png`
- `REPO_ROOT/out_bottom_metal/need_k_for_E0_small.csv`

---

### 2.2 `gpvm_k_lambda_u0/run_gpvm_K_lambda_u0.py` ✅ Reader
**役割**: strict-EML の `K(λ, u=0)` を計算。

**読む JSON**
- `REPO_ROOT/out_gpvm_step2/best_geometry.json`

**パスの決まり方（cwd 非依存）**
- `HERE = Path(__file__).resolve().parent`
- `BASE = HERE.parent`  → ここで `BASE` は **REPO_ROOT**
- `BEST_JSON = BASE / "out_gpvm_step2" / "best_geometry.json"`

**主な出力**
- `REPO_ROOT/gpvm_k_lambda_u0/` 配下に `*.png`, `*.npy`

---

### 2.3 `gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py` ✅ Reader
**役割**: (A) phase3-BT proxy と (B) strict-EML の `K(λ,u=0)` を比較。

**読む JSON**
- `REPO_ROOT/out_gpvm_step2/best_geometry.json`

**パスの決まり方（cwd 非依存）**
- `HERE = Path(__file__).resolve().parent`
- `BASE = HERE.parent` → **REPO_ROOT**
- `BEST_JSON = BASE / "out_gpvm_step2" / "best_geometry.json"`

**主な出力**
- `REPO_ROOT/gpvm_k_lambda_u0/gpvm_K_lambda_u0_A_vs_B.png` ほか

---

### 2.4 `gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py` ✅ Reader
**役割**: strict-EML の `K(λ, k//)` heatmap と k//=0 一致チェック。

**読む JSON**
- `REPO_ROOT/out_gpvm_step2/best_geometry.json`

**パスの決まり方（cwd 非依存）**
- `HERE = Path(__file__).resolve().parent`
- `BASE = HERE.parent` → **REPO_ROOT**
- `BEST_JSON = BASE / "out_gpvm_step2" / "best_geometry.json"`

**主な出力**
- `REPO_ROOT/gpvm_K_lambda_kpar_map_strict/` 配下に heatmap `*.png` と `*.npy`

---

## 3) 重複配置による事故ポイント（ver10 で “混乱の温床” になっている箇所）

ver10 には **同名の heatmap スクリプトが2箇所**に存在します：

1. **ルート側（正）**  
   `REPO_ROOT/gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py`  
   → `REPO_ROOT/out_gpvm_step2/best_geometry.json` を読む（ver10 の実体と整合）

2. **ネスト側（要注意）**  
   `REPO_ROOT/phase3p1_pyMoosh_Splane_TMMrewrap/gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py`  
   → `BASE = HERE.parent = REPO_ROOT/phase3p1_pyMoosh_Splane_TMMrewrap`  
   → **`REPO_ROOT/phase3p1_pyMoosh_Splane_TMMrewrap/out_gpvm_step2/best_geometry.json`** を読もうとする

ところが ver10 には通常、
- `REPO_ROOT/out_gpvm_step2/` は存在する
- `REPO_ROOT/phase3p1_pyMoosh_Splane_TMMrewrap/out_gpvm_step2/` は **存在しない**
ため、ネスト側を実行すると
- JSON が見つからない（FileNotFound）
- あるいは別バージョンのファイルがそこに残っていれば **別の geometry を読んでしまう**
などが起きます。

**対策**
- ver10 では基本的に **ルート側のスクリプト群**を使う（`REPO_ROOT/gpvm_*` や `REPO_ROOT/run_*`）
- もしくは、ネスト側を使うなら、その配下に `out_gpvm_step2/best_geometry.json` を置く（=構造を揃える）

---

## 4) 一覧（Writer/Reader と JSON パス）

| 種別 | スクリプト | JSON の読み書き | JSON パスの基準 | 実際の JSON パス |
|---|---|---|---|---|
| Writer | `run_phase3_opt_then_gpvm_eml_profile.py` | **write** | `Path(__file__).parent` | `REPO_ROOT/out_gpvm_step2/best_geometry.json` |
| Reader | `run_bottom_metal_PEC_proxy.py` | read | `dirname(abspath(__file__))` | `REPO_ROOT/out_gpvm_step2/best_geometry.json` |
| Reader | `gpvm_k_lambda_u0/run_gpvm_K_lambda_u0.py` | read | `BASE = HERE.parent` | `REPO_ROOT/out_gpvm_step2/best_geometry.json` |
| Reader | `gpvm_k_lambda_u0/run_gpvm_K_lambda_u0_A_vs_B.py` | read | `BASE = HERE.parent` | `REPO_ROOT/out_gpvm_step2/best_geometry.json` |
| Reader | `gpvm_K_lambda_kpar_map_strict/run_gpvm_K_lambda_kpar_map_strict.py` | read | `BASE = HERE.parent` | `REPO_ROOT/out_gpvm_step2/best_geometry.json` |
| Reader(注意) | `phase3p1_pyMoosh_Splane_TMMrewrap/.../run_gpvm_K_lambda_kpar_map_strict.py` | read | `BASE = HERE.parent` | `REPO_ROOT/phase3p1_pyMoosh_Splane_TMMrewrap/out_gpvm_step2/best_geometry.json` |

---

## 5) 「どこで実行したか(cwd)」が効かない理由（再掲）
上記の通り、全スクリプトが `__file__` を使ってパスを決めているため、
`cd` して作業ディレクトリを変えても、参照される JSON は変わりません。

**変わるのは**「そのスクリプトが置かれているディレクトリ（= __file__）」だけです。

---

必要なら、次の段階として  
「重複配置を排除し、`--best_json` で明示指定できるようにして、取り違えを物理的に潰す」  
という設計に整理できます（ver10 以降の混乱もここが根本原因になりがちです）。
