# QAOA Newsvendor 可視化ガイド

## 概要

このガイドでは、newsvendor.py に追加された詳細な可視化機能の使用方法を説明します。

## 追加された可視化機能

### 1. QAOAの収束過程の可視化
### 2. 量子回路構造の分析
### 3. FSLエンコーディングの忠実度評価
### 4. 包括的な分析ダッシュボード

---

## 使用方法

### 基本的な実行と可視化

```python
from scipy.stats import norm
from newsvendor import solve_newsvendor_qaoa, comprehensive_analysis

# 1. 問題を定義
mu, sigma = 3, 1
demand_pdf = lambda x: norm.pdf(x, mu, sigma)

# 2. QAOAで解く
result = solve_newsvendor_qaoa(
    demand_dist=demand_pdf,
    c=1.0, lam=5.0,
    Q_max=5, D_max=5,
    p=1, M=1,
    n_shots=1000,
    verbose=True
)

# 3. 包括的な分析を実行（すべての可視化を一度に）
comprehensive_analysis(result)
```

これにより以下が自動的に生成されます：
- ✅ 最適化収束プロット (`qaoa_convergence.png`)
- ✅ 回路構造の分析
- ✅ FSL忠実度評価 (`fsl_fidelity.png`)
- ✅ 最終結果の比較 (`newsvendor_results.png`)

---

## 個別の可視化機能

### 1. 最適化収束プロット

```python
from newsvendor import plot_optimization_convergence

# QAOAの収束過程を詳細に可視化
plot_optimization_convergence(result['optimization_history'])
```

**表示内容：**
- **左上**: コスト関数の収束
  - 各イテレーションでのコスト値
  - 最良コストの位置（赤い星印）

- **右上**: γパラメータの進化
  - 各QAOA層のγ値の変化

- **左下**: βパラメータの進化
  - 各QAOA層のβ値の変化

- **右下**: Q分布の進化
  - 初期、中間、最終イテレーションでの測定分布の比較

### 2. 回路構造の分析

```python
from newsvendor import visualize_circuit_structure

# 回路の詳細情報を表示
visualize_circuit_structure(result, show_full=False)
```

**表示内容：**
- 総量子ビット数
- 総ゲート数
- 推定回路深さ
- QAOAの層数 (p)
- レジスタレイアウト（R_q, R_d, R_f, Ancilla）
- ゲートの種類と数

**回路図の表示：**
```python
# qulacsvisがインストールされている場合、回路図も表示可能
visualize_circuit_structure(result, show_full=True)
```

> **注意**: 大きな回路の場合、最初の100ゲートのみ表示されます

### 3. FSLエンコーディングの忠実度評価

```python
from newsvendor import analyze_fsl_encoding

# FSLの精度を詳細にチェック
fsl_metrics = analyze_fsl_encoding(
    result['ks'],
    result['cs'],
    result['demand_dist'],
    D_max=5,
    n_samples=10000
)

print(f"Classical Fidelity: {fsl_metrics['fidelity']:.6f}")
print(f"Total Variation Distance: {fsl_metrics['tvd']:.6f}")
```

**表示内容：**
- **左**: ターゲット分布 vs FSLエンコード分布（棒グラフ比較）
- **右**: エンコーディング誤差（差分のプロット）

**評価指標：**
- **Classical Fidelity**: 1.0に近いほど良い（理想値 = 1.0）
- **Total Variation Distance**: 0.0に近いほど良い（理想値 = 0.0）

### 4. 基本的な結果の可視化

```python
from newsvendor import visualize_qaoa_results

# 最終結果のみを可視化
visualize_qaoa_results(
    result['distribution'],
    result['demand_dist'],
    c=1.0, lam=5.0, Q_max=5,
    classical_q=result['classical_solution'],
    quantum_q=result['quantum_solution']
)
```

**表示内容（2x2ダッシュボード）：**
- **左上**: QAOA測定分布
- **右上**: コスト関数ランドスケープ
- **左下**: 需要分布
- **右下**: 比較メトリクス（量子 vs 古典）

---

## Jupyter Notebookでの使用例

### セル 1: インポートと設定

```python
import numpy as np
from scipy.stats import norm
from newsvendor import (
    solve_newsvendor_qaoa,
    comprehensive_analysis,
    plot_optimization_convergence,
    visualize_circuit_structure,
    analyze_fsl_encoding
)

# 日本語フォント設定（オプション）
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
```

### セル 2: 問題の定義と実行

```python
# ガウス分布による需要
mu, sigma = 3, 1
demand_pdf = lambda x: norm.pdf(x, mu, sigma)

# QAOAで解く
result = solve_newsvendor_qaoa(
    demand_dist=demand_pdf,
    c=1.0,          # 発注コスト
    lam=5.0,        # 欠品ペナルティ
    Q_max=5,        # 最大発注量
    D_max=5,        # 最大需要
    p=1,            # QAOA深さ
    M=1,            # フーリエ打ち切り
    n_shots=1000,   # 測定回数
    verbose=True
)
```

### セル 3: 包括的な分析（推奨）

```python
# すべての可視化を一度に実行
comprehensive_analysis(result)
```

### セル 4: 個別の可視化（詳細な分析用）

```python
# 1. 収束過程の詳細分析
plot_optimization_convergence(result['optimization_history'])
```

```python
# 2. 回路構造の確認
visualize_circuit_structure(result, show_full=False)
```

```python
# 3. FSL精度の検証
fsl_metrics = analyze_fsl_encoding(
    result['ks'], result['cs'],
    result['demand_dist'], D_max=5
)
```

---

## トラブルシューティング

### 問題 1: グラフが表示されない

**原因**: Jupyter Notebookでmatplotlibのバックエンドが正しく設定されていない

**解決策**:
```python
# ノートブックの最初のセルで実行
%matplotlib inline
import matplotlib.pyplot as plt
```

### 問題 2: 日本語が文字化けする

**原因**: 日本語フォントが設定されていない

**解決策**:
```python
import matplotlib.pyplot as plt
# 利用可能なフォントを確認
from matplotlib import font_manager
fonts = [f.name for f in font_manager.fontManager.ttflist]
print([f for f in fonts if 'Japanese' in f or 'Noto' in f])

# 日本語フォントを設定
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # またはIPAexGothic等
plt.rcParams['axes.unicode_minus'] = False
```

### 問題 3: 回路図が表示されない

**原因**: qulacsvisがインストールされていない

**解決策**:
```bash
pip install qulacsvis
```

または、回路図なしで分析を続行:
```python
visualize_circuit_structure(result, show_full=False)  # テキスト情報のみ
```

---

## 出力ファイル

実行すると以下のPNGファイルが自動生成されます：

| ファイル名 | 内容 | 生成関数 |
|-----------|------|---------|
| `qaoa_convergence.png` | 最適化収束プロット（4パネル） | `plot_optimization_convergence()` |
| `fsl_fidelity.png` | FSLエンコーディング精度（2パネル） | `analyze_fsl_encoding()` |
| `newsvendor_results.png` | 最終結果比較（4パネル） | `visualize_qaoa_results()` |
| `circuit_diagram.png` | 量子回路図（オプション） | `visualize_circuit_structure(show_full=True)` |

---

## 解釈のポイント

### 1. 収束プロットから読み取れること

✅ **良好な収束の兆候:**
- コストが単調に減少している
- パラメータの変化が徐々に小さくなる
- Q分布が明確なピークに収束

⚠️ **問題がある場合:**
- コストが振動している → `n_shots`を増やす
- パラメータが大きく変動 → 初期値を変える
- Q分布が一様のまま → `p`を増やす、または問題設定を見直す

### 2. FSL忠実度の評価

- **Fidelity > 0.95**: 十分良好
- **Fidelity 0.90-0.95**: 許容範囲、`M`を増やすと改善
- **Fidelity < 0.90**: 要改善、`M`を大幅に増やす必要あり

### 3. 回路構造の評価

- **ゲート数**: 少ないほど良い（ノイズの影響が小さい）
- **回路深さ**: 小さいほど良い（実機での実行可能性が高い）
- **QAOA層数 (p)**:
  - `p=1-2`: 小規模問題に適切
  - `p=3-5`: 中規模問題
  - `p>5`: 通常不要、収束が遅い

---

## 高度な使用例

### 複数の実行結果を比較

```python
# 異なるパラメータで複数回実行
results = []
for p in [1, 2, 3]:
    result = solve_newsvendor_qaoa(
        demand_dist=demand_pdf,
        c=1.0, lam=5.0, Q_max=5, D_max=5,
        p=p, M=1, n_shots=1000, verbose=False
    )
    results.append(result)

# 収束を比較
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, result in enumerate(results):
    axes[i].plot(result['optimization_history']['costs'])
    axes[i].set_title(f"p={i+1}")
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel('Cost')
plt.tight_layout()
plt.show()
```

### カスタム分析

```python
# 最適化履歴から詳細なメトリクスを抽出
history = result['optimization_history']

# パラメータの変動量を計算
params_std = np.std(history['params'], axis=0)
print(f"Parameter variation: γ={params_std[0]:.4f}, β={params_std[1]:.4f}")

# 最良コストの安定性を評価
costs = np.array(history['costs'])
best_cost_idx = np.argmin(costs)
costs_after_best = costs[best_cost_idx:]
stability = np.std(costs_after_best) / np.mean(costs_after_best)
print(f"Cost stability after convergence: {stability:.4f}")
```

---

## 参考文献

- QAOA: Farhi et al. (2014) "A Quantum Approximate Optimization Algorithm"
- FSL: Fourier Series Loading for quantum state preparation
- Newsvendor Problem: Classical operations research

---

**更新日**: 2025-12-26
