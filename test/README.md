# ver17 test/

## 目的
- `run_rgb_full_eval.sh` で使う主要スクリプト（WRITER/u0/avsb/heatmap/bottom-metal）と
- 共通ライブラリ `./gpvm/*.py` と
- rewrap/BT (`tmm_rewrap_utils_policyB.py`, `oled_cavity_phase3p1_policyB.py`)
を、できる限り **PyMoosh-stable** を参照(ref)にして検証する。

## 実行方法
ver17 ルートで：

```bash
cd test
python run_all_tests.py
```

### 出力の詳細度
デフォルトで、各テストは以下を**必ず表示**します：
- 参照(ref: PyMoosh 等)の値
- プログラム側の値
- 差分指標（max|Δ|, max_rel など）
- OK 判定基準（atol/rtol/閾値）

出力を減らしたい場合は、環境変数を指定してください：

```bash
TEST_VERBOSE=0 python run_all_tests.py
```

## 注意
- PyMoosh は ver17 同梱の `../PyMoosh-stable/` を参照します。
- 重い 2D heatmap の full 計算は時間がかかるため、このテストでは主に
  - (a) u=0 計算の健全性
  - (b) λグリッド生成/キャッシュの整合
  を検証します（map全点の再計算はしません）。
