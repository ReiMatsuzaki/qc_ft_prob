# Newsvendor QAOA - 使用例

## Jupyter Notebookでの基本的な使い方

### セル 1: 必要なモジュールのインポート

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# newsvendor関数をインポート
from newsvendor import (
    solve_newsvendor_qaoa,           # メイン関数
    comprehensive_analysis,           # 包括的な分析
    plot_optimization_convergence,   # 収束プロット
    visualize_circuit_structure,     # 回路構造の可視化
    analyze_fsl_encoding,            # FSL精度評価
    visualize_qaoa_results           # 基本的な結果プロット
)

# Jupyter用の設定
%matplotlib inline
```

### セル 2: 問題の定義と実行

```python
# ガウス分布による需要の定義
mu, sigma = 3, 1
demand_pdf = lambda x: norm.pdf(x, mu, sigma)

# QAOAで解く
result = solve_newsvendor_qaoa(
    demand_dist=demand_pdf,  # 需要分布（連続PDF）
    c=1.0,                   # 単位発注コスト
    lam=5.0,                 # 欠品ペナルティ（λ）
    Q_max=5,                 # 最大発注量
    D_max=5,                 # 最大需要
    p=1,                     # QAOA深さ
    M=1,                     # フーリエ打ち切り次数
    n_shots=1000,            # 測定回数
    verbose=True             # 詳細表示
)
```

**実行結果の例：**
```
============================================================
Newsvendor Problem with FSL + QAOA
============================================================
Problem size: Q_max=5, D_max=5
Qubits: n_q=3, n_d=3, total=10
QAOA depth: p=1, Fourier truncation: M=1
Demand distribution encoded with 3 Fourier modes

Optimizing QAOA parameters...
  Iteration 10: cost = 5.2340
  Iteration 20: cost = 4.8120
Optimization completed: 28 function evaluations

============================================================
RESULTS
============================================================
Quantum solution: q = 4
Quantum cost: 5.0000
Classical solution: q = 4
Classical cost: 5.0000
Approximation ratio: 1.0000
Measurement confidence: 45.20%
Circuit depth (estimated): 20
```

### セル 3: 包括的な分析（推奨）

```python
# すべての可視化を一度に実行
comprehensive_analysis(result)
```

**生成される可視化：**

1. **最適化収束プロット** (4パネル)
   - コスト関数の収束
   - γパラメータの進化
   - βパラメータの進化
   - Q分布の進化

2. **回路構造の分析**
   - 量子ビット数とゲート数
   - レジスタレイアウト
   - ゲートの種類と数

3. **FSL忠実度評価** (2パネル)
   - ターゲット vs エンコード分布
   - エンコーディング誤差

4. **最終結果の比較** (4パネル)
   - QAOA測定分布
   - コスト関数ランドスケープ
   - 需要分布
   - 量子 vs 古典の比較

### セル 4: 個別の可視化（オプション）

```python
# 1. 最適化収束の詳細分析
plot_optimization_convergence(result['optimization_history'])
```

```python
# 2. 回路構造の確認
visualize_circuit_structure(result, show_full=False)
```

**出力例：**
```
============================================================
CIRCUIT STRUCTURE ANALYSIS
============================================================

Circuit Statistics:
  Total qubits: 10
  Total gates: 41
  Estimated depth: 20
  QAOA layers (p): 1

Register Layout:
  R_q (order quantity): qubits 0-2 (3 qubits)
  R_d (demand): qubits 3-5 (3 qubits)
  R_f (stockout flag): qubit 6
  Ancilla: qubits 7+

Gate Composition:
  ClsOneQubitGate: 15
  QuantumGateMatrix: 13
  ClsOneQubitRotationGate: 7
  ClsOneControlOneTargetGate: 6
```

```python
# 3. FSL精度の詳細評価
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

```python
# 4. 基本的な結果のみ表示
Q_max = max(result['distribution'].keys())
visualize_qaoa_results(
    result['distribution'],
    result['demand_dist'],
    c=1.0, lam=5.0, Q_max=Q_max,
    classical_q=result['classical_solution'],
    quantum_q=result['quantum_solution']
)
```

---

## 異なる問題サイズでの実験

### 小問題（Q_max=3）

```python
mu, sigma = 2, 0.5
demand_pdf = lambda x: norm.pdf(x, mu, sigma)

result_small = solve_newsvendor_qaoa(
    demand_dist=demand_pdf,
    c=1.0, lam=5.0,
    Q_max=3, D_max=3,
    p=1, M=1,
    n_shots=500,
    verbose=True
)

comprehensive_analysis(result_small)
```

### 中問題（Q_max=10）

```python
mu, sigma = 5, 2
demand_pdf = lambda x: norm.pdf(x, mu, sigma)

result_medium = solve_newsvendor_qaoa(
    demand_dist=demand_pdf,
    c=2.0, lam=10.0,
    Q_max=10, D_max=10,
    p=2, M=2,  # より多くの層とフーリエモード
    n_shots=2000,
    verbose=True
)

comprehensive_analysis(result_medium)
```

---

## 離散分布での実行

```python
# 離散的な需要分布の定義
demand_dist = {
    0: 0.05,
    1: 0.15,
    2: 0.30,
    3: 0.30,
    4: 0.15,
    5: 0.05
}

result_discrete = solve_newsvendor_qaoa(
    demand_dist=demand_dist,  # 辞書形式で直接指定
    c=1.0, lam=5.0,
    Q_max=5, D_max=5,
    p=1, M=1,
    n_shots=1000,
    verbose=True
)

comprehensive_analysis(result_discrete)
```

---

## 結果の詳細な分析

### 最適化履歴の取得

```python
# 最適化の各イテレーションのデータ
history = result['optimization_history']

print(f"総イテレーション数: {len(history['costs'])}")
print(f"初期コスト: {history['costs'][0]:.4f}")
print(f"最終コスト: {history['costs'][-1]:.4f}")
print(f"最良コスト: {min(history['costs']):.4f}")

# コストの推移をプロット
plt.figure(figsize=(10, 4))
plt.plot(history['costs'], 'o-')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('QAOA Optimization Progress')
plt.grid(alpha=0.3)
plt.show()
```

### パラメータの分析

```python
# 最適パラメータの取得
optimal_params = result['optimal_params']
p = result['p']
gammas = optimal_params[:p]
betas = optimal_params[p:]

print("最適パラメータ:")
for i in range(p):
    print(f"  Layer {i+1}: γ = {gammas[i]:.4f}, β = {betas[i]:.4f}")
```

### 測定分布の詳細

```python
# Q分布の詳細表示
dist = result['distribution']
total = sum(dist.values())

print("測定分布:")
for q in sorted(dist.keys()):
    prob = dist[q] / total
    print(f"  q={q}: {dist[q]:4d} counts ({prob:.2%})")
```

---

## カスタムコスト関数での実験

```python
# 異なるコストパラメータで実行
cost_params = [
    (0.5, 5.0),   # 低い発注コスト
    (1.0, 5.0),   # 標準
    (2.0, 5.0),   # 高い発注コスト
]

results_comparison = []

for c, lam in cost_params:
    result = solve_newsvendor_qaoa(
        demand_dist=demand_pdf,
        c=c, lam=lam,
        Q_max=5, D_max=5,
        p=1, M=1,
        n_shots=1000,
        verbose=False
    )
    results_comparison.append((c, lam, result))

# 結果の比較
for c, lam, result in results_comparison:
    print(f"c={c}, λ={lam}: q*={result['quantum_solution']}, "
          f"cost={result['quantum_cost']:.4f}")
```

---

## よくある質問（FAQ）

### Q1: M (フーリエ打ち切り次数) はどのように選ぶべきですか？

**A:** 問題サイズに応じて：
- 小問題 (D_max ≤ 7): `M=1-2`
- 中問題 (D_max ≤ 31): `M=4-8`
- 大問題 (D_max ≤ 127): `M=16-32`

FSL忠実度が低い場合は、Mを増やしてください。

### Q2: p (QAOA深さ) はいくつにすべきですか？

**A:** 通常：
- 小問題: `p=1-2`
- 中問題: `p=2-3`
- 大問題: `p=3-5`

p>5はあまり改善がなく、計算時間が長くなります。

### Q3: n_shots (測定回数) の適切な値は？

**A:**
- テスト: `n_shots=100-500`
- 通常使用: `n_shots=1000-2000`
- 高精度: `n_shots=5000-10000`

最適化中は少なめ、最終測定は多めが効率的です。

### Q4: 量子解と古典解が異なる場合は？

**A:** 以下を確認：
1. 近似比が1に近いか（コストが同等なら問題なし）
2. FSL忠実度が高いか（低い場合はMを増やす）
3. 測定信頼度が十分か（低い場合はn_shotsを増やす）
4. QAOAが収束しているか（収束プロットを確認）

---

## 生成されるファイル

実行後、以下のPNGファイルが生成されます：

| ファイル | サイズ | 内容 |
|---------|-------|------|
| `qaoa_convergence.png` | ~200KB | 最適化収束プロット（4パネル） |
| `fsl_fidelity.png` | ~150KB | FSL精度評価（2パネル） |
| `newsvendor_results.png` | ~200KB | 最終結果比較（4パネル） |

---

**作成日**: 2025-12-26
**対応バージョン**: newsvendor.py v1.1+
