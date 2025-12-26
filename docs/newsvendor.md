# Design Document

## Newsvendor (Stockout-Probability Formulation) with FSL + QAOA

---

## 1. Problem Definition

### 1.1 Decision Variable

* 発注量
  [
  q \in \mathcal Q = {0,1,\dots,Q_{\max}}
  ]

### 1.2 Random Variable

* 需要
  [
  D \in \mathcal D = {0,1,\dots,D_{\max}},\quad \Pr(D=d)=p_d
  ]

### 1.3 Objective Function（簡約版ニュースベンダー）

[
\boxed{
C(q)= c,q + \lambda,\Pr(D>q)
}
]

* (c>0)：単位発注コスト
* (\lambda>0)：欠品が起きた場合の期待的ペナルティ（「欠品が起きること自体」への罰）

**解釈**

* 「どれだけ欠品するか」ではなく
* 「欠品が起きる確率」を抑えたい意思決定
* 実務上の *サービスレベル重視型KPI* と整合的

---

## 2. Quantum–Classical Role Split（重要）

| 要素         | 実装                    |
| ---------- | --------------------- |
| 需要分布 (p_d) | **FSL** により量子状態としてロード |
| 欠品判定 (D>q) | **量子比較回路（1bit）**      |
| (\Pr(D>q)) | 量子期待値として自然に評価         |
| (q) の探索    | **QAOA**              |
| パラメータ最適化   | 古典最適化（外側ループ）          |

---

## 3. Quantum Register Layout

### 3.1 Registers

| Register | 内容      | Qubits                                |
| -------- | ------- | ------------------------------------- |
| (R_q)    | 発注量 (q) | (n_q=\lceil \log_2(Q_{\max}+1)\rceil) |
| (R_d)    | 需要 (d)  | (n_d=\lceil \log_2(D_{\max}+1)\rceil) |
| (R_f)    | 欠品フラグ   | 1                                     |
| Ancilla  | 補助      | 必要最小限                                 |

---

## 4. State Preparation (FSL)

### 4.1 Demand State via FSL

FSL回路 (U_{\mathrm{FSL}}) を用いて

[
U_{\mathrm{FSL}} |0\rangle^{\otimes n_d}
;\approx;
|\psi_D\rangle
==============

\sum_{d\in\mathcal D} \sqrt{p_d},|d\rangle
]

### 4.2 Initial State

[
|\Psi_0\rangle
==============

\Big(\frac{1}{\sqrt{|\mathcal Q|}}\sum_{q\in\mathcal Q}|q\rangle\Big)
\otimes
|\psi_D\rangle
\otimes
|0\rangle_f
]

* (R_q)：一様重ね合わせ（(|+\rangle^{\otimes n_q})）
* (R_d)：FSL
* (R_f)：欠品判定用

---

## 5. Cost Oracle Design（最重要）

### 5.1 Stockout Indicator

欠品フラグを次で定義：
[
f(q,d) = \mathbf 1[d>q]
]

#### Reversible Comparator

[
|q\rangle|d\rangle|0\rangle_f
;\mapsto;
|q\rangle|d\rangle|f(q,d)\rangle_f
]

* **比較のみ**
* 減算・絶対値・乗算なし
* 回路深さ・エラーに強い

---

### 5.2 Cost Hamiltonian

[
H_C
===

c,\hat q
+
\lambda,\hat f
]

* (\hat q)：(R_q) 上の数値演算子
* (\hat f)：欠品フラグ演算子（0 or 1）

---

### 5.3 Phase Kickback Implementation

コストユニタリ：
[
U_C(\gamma)=e^{-i\gamma H_C}
]

#### 分解

1. **発注コスト項**
   [
   e^{-i\gamma c \hat q}
   ]
   → (R_q) 各ビットへの位相回転

2. **欠品確率項**
   [
   e^{-i\gamma \lambda \hat f}
   ]
   → (R_f) が (|1\rangle) のときのみ位相回転

3. 欠品フラグのアンコンピュート

---

## 6. QAOA Structure

### 6.1 Mixer Hamiltonian

[
H_M = \sum_{j=1}^{n_q} X_j
]

[
U_M(\beta)=e^{-i\beta H_M}
]

* **(R_q) のみ作用**
* (R_d), (R_f) は固定

---

### 6.2 Full QAOA Circuit

深さ (p) のQAOA：

[
|\Psi(\boldsymbol{\gamma},\boldsymbol{\beta})\rangle
====================================================

\prod_{k=1}^p
\Big(
U_M(\beta_k),
U_C(\gamma_k)
\Big)
|\Psi_0\rangle
]

---

## 7. Objective Evaluation

量子期待値：
[
F(\boldsymbol{\gamma},\boldsymbol{\beta})
=========================================

# \langle \Psi | H_C | \Psi \rangle

\sum_{q} P(q)\Big(cq + \lambda \Pr(D>q)\Big)
]

* (\Pr(D>q)) は
  (|\psi_D\rangle) によって **自動的に重み付け**

---

## 8. Measurement & Decision

1. (R_q) を測定しサンプル ({q_i}) を取得
2. 最頻値 or 期待コスト最小の (q) を選択
3. 必要なら古典側で
   [
   \Pr(D>q)=\sum_{d>q} p_d
   ]
   を再評価して確定

---

## 9. Advantages of This Formulation

### 9.1 Quantum-Circuit Friendly

* 比較のみ（No subtraction, No multiplication）
* アンシラ最小
* NISQ向き

### 9.2 Interpretation

* 欠品確率を直接抑制
* サービスレベル設計と一致
* 経営・業務部門への説明が容易

### 9.3 Extensibility

* 複数品目：(\sum_i \lambda_i \Pr(D_i>q_i))
* CVaR近似：閾値を変えた複数フラグの和
* 制約付きQAOA（ロット・容量）と自然に結合可能

---

## 10. Known Limitations

* 欠品量の大きさは評価しない
* (\lambda) の意味は「1回欠品したときの期待的損失」
* 過剰在庫リスクは (c) に吸収されている

---

## 11. Summary (for Agent)

> **Goal**
> Minimize (c q + \lambda \Pr(D>q))

> **Key Insight**
> 欠品は「起きたかどうか」だけを見る
> → 比較1回で量子化可能

> **Quantum Core**
>
> * FSL: demand distribution
> * Comparator: stockout flag
> * QAOA: search over (q)

---

この定式化は
**「ニュースベンダーの本質（分位点的意思決定）」を保ちつつ、
量子回路を極限まで軽くする**
という意味で、FSL×QAOAの“入口問題”として非常に優秀です。

次の段階としては

* (\Pr(D>q)) を段階的に重み付けして **欠品量を近似的に導入**
* 複数品目・容量制約付きミキサー設計

などに自然に拡張できます。
