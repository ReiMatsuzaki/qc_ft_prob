# Newsvendor問題におけるFSL + QAOA：数学的定式化

## 1. 問題の数学的定式化

### 1.1 古典的最適化問題

発注量 $q \in \{0, 1, \ldots, Q_{\max}\}$ に対するコスト関数：

$$
C(q) = c \cdot q + \lambda \cdot \Pr(D > q)
$$

ここで：
- $c > 0$：単位発注コスト
- $\lambda > 0$：欠品ペナルティ
- $D$：需要（確率変数、確率質量関数 $p_d = \Pr(D = d)$ に従う）
- $\Pr(D > q) = \sum_{d > q} p_d$：欠品確率

**最適化目標**：

$$
q^* = \arg\min_{q \in \{0, \ldots, Q_{\max}\}} C(q)
$$

### 1.2 期待コストの展開

各発注量に対するコストは以下のように分解される：

$$
C(q) = c \cdot q + \lambda \sum_{d=q+1}^{D_{\max}} p_d
$$

これは凸関数であり、適切な条件下で一意の最適解を持つ。

## 2. 量子状態による確率分布の表現

### 2.1 需要の量子状態（FSLの出力）

需要の確率分布を量子状態の振幅として符号化：

$$
|\psi_{\text{demand}}\rangle = \sum_{d=0}^{D_{\max}} \sqrt{p_d} \, |d\rangle
$$

**重要な性質**：

$$
\langle \psi_{\text{demand}} | \psi_{\text{demand}} \rangle = \sum_{d=0}^{D_{\max}} p_d = 1
$$

測定すると確率 $p_d$ で状態 $|d\rangle$ が観測される。

### 2.2 完全な初期状態

需要レジスタ $R_d$（$n_d$ qubits）、発注量レジスタ $R_q$（$n_q$ qubits）、フラグqubit $R_f$、補助qubit群 $A$ を用いて：

$$
|\psi_0\rangle = |\psi_{\text{demand}}\rangle_d \otimes |\psi_{\text{uniform}}\rangle_q \otimes |0\rangle_f \otimes |0\rangle_A
$$

ここで、発注量レジスタは均等重ね合わせ：

$$
|\psi_{\text{uniform}}\rangle_q = \frac{1}{\sqrt{2^{n_q}}} \sum_{q=0}^{2^{n_q}-1} |q\rangle
$$

完全な形：

$$
|\psi_0\rangle = \frac{1}{\sqrt{2^{n_q}}} \sum_{d=0}^{D_{\max}} \sum_{q=0}^{2^{n_q}-1} \sqrt{p_d} \, |d\rangle |q\rangle |0\rangle |0\rangle
$$

## 3. FSLの数学的原理

### 3.1 フーリエ級数表現

確率密度関数 $f(x)$ を区間 $[0, T]$ で近似：

$$
f(x) \approx \sum_{k=-M}^{M} c_k e^{2\pi i k x / T}
$$

**フーリエ係数**：

$$
c_k = \frac{1}{T} \int_0^T f(x) e^{-2\pi i k x / T} \, dx
$$

### 3.2 量子状態としての符号化

フーリエ係数を振幅として符号化した状態：

$$
|\phi\rangle = \sum_{k=-M}^{M} c_k |k\rangle
$$

### 3.3 逆量子フーリエ変換（IQFT）

IQFT演算子の作用：

$$
\text{IQFT} |k\rangle = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} e^{-2\pi i k x / N} |x\rangle
$$

ここで $N = 2^{n_d}$。

### 3.4 FSLの最終出力

$$
|\psi_{\text{demand}}\rangle = \text{IQFT} \left( \sum_{k=-M}^{M} c_k |k\rangle \right)
$$

$$
= \sum_{k=-M}^{M} c_k \cdot \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} e^{-2\pi i k x / N} |x\rangle
$$

$$
= \sum_{x=0}^{N-1} \left[ \frac{1}{\sqrt{N}} \sum_{k=-M}^{M} c_k e^{-2\pi i k x / N} \right] |x\rangle
$$

$$
= \sum_{x=0}^{N-1} \sqrt{f(x)} \, |x\rangle
$$

ただし、適切な正規化により $\sum_x f(x) = 1$ となるようにする。

**FSLの本質**：フーリエ級数の係数 $\{c_k\}$ から、関数値 $f(x)$ を振幅とする量子状態を生成する。

## 4. QAOAのハミルトニアン

### 4.1 コストハミルトニアン

$$
H_C = c \cdot \hat{q} + \lambda \cdot \hat{f}
$$

**演算子の定義**：

#### 発注量演算子

$$
\hat{q} = \sum_{j=0}^{n_q - 1} 2^j \cdot \frac{I - Z_j}{2}
$$

これは、$|q\rangle$ 状態に対して固有値 $q$ を返す：

$$
\hat{q} |q\rangle = q |q\rangle
$$

#### フラグ演算子

$$
\hat{f} = \frac{I - Z_f}{2}
$$

これは、フラグqubitの値（0 or 1）を返す：

$$
\hat{f} |0\rangle = 0, \quad \hat{f} |1\rangle = 1
$$

### 4.2 コストユニタリ演算子

$$
U_C(\gamma) = e^{-i\gamma H_C} = e^{-i\gamma (c \cdot \hat{q} + \lambda \cdot \hat{f})}
$$

Pauli-Z 基底での実装：

$$
U_C(\gamma) = \prod_{j=0}^{n_q-1} R_Z^{(j)}(-\gamma c \cdot 2^j) \cdot R_Z^{(f)}(-\gamma \lambda)
$$

ここで $R_Z(\theta) = e^{-i\theta Z/2}$ は Z軸回転ゲート。

### 4.3 混合ハミルトニアン

$$
H_M = \sum_{j=0}^{n_q - 1} X_j
$$

$R_q$ レジスタのみに作用し、異なる $|q\rangle$ 状態の重ね合わせを生成する。

**混合ユニタリ演算子**：

$$
U_M(\beta) = e^{-i\beta H_M} = \prod_{j=0}^{n_q-1} R_X^{(j)}(2\beta)
$$

ここで $R_X(\theta) = e^{-i\theta X/2}$ は X軸回転ゲート。

## 5. 比較演算子（Comparator）

### 5.1 数学的定義

可逆的なユニタリ演算子 $U_{\text{comp}}$ で、以下の写像を実現：

$$
U_{\text{comp}} : |q\rangle |d\rangle |0\rangle \mapsto |q\rangle |d\rangle |f(q,d)\rangle
$$

ここで：

$$
f(q, d) = \begin{cases}
1 & \text{if } d > q \\
0 & \text{if } d \leq q
\end{cases}
$$

### 5.2 比較演算子の性質

**可逆性**：

$$
U_{\text{comp}}^\dagger \, U_{\text{comp}} = I
$$

**補助qubitの使用**：実際には $n_{\text{anc}}$ 個の補助qubitを使用して、リップルキャリー方式で比較を実行。

## 6. 単一QAOAレイヤーの動作

### 6.1 1層の演算子

$$
U(\beta, \gamma) = U_M(\beta) \cdot U_{\text{comp}}^\dagger \cdot U_C(\gamma) \cdot U_{\text{comp}}
$$

### 6.2 各ステップの状態変化

**初期状態**：

$$
|\psi_0\rangle = \frac{1}{\sqrt{2^{n_q}}} \sum_{d,q} \sqrt{p_d} \, |d\rangle |q\rangle |0\rangle
$$

**Comparator適用後**：

$$
|\psi_1\rangle = U_{\text{comp}} |\psi_0\rangle = \frac{1}{\sqrt{2^{n_q}}} \sum_{d,q} \sqrt{p_d} \, |d\rangle |q\rangle |f(q,d)\rangle
$$

**Cost Oracle適用後**：

$$
|\psi_2\rangle = U_C(\gamma) |\psi_1\rangle = \frac{1}{\sqrt{2^{n_q}}} \sum_{d,q} \sqrt{p_d} \, e^{-i\gamma(c \cdot q + \lambda \cdot f(q,d))} |d\rangle |q\rangle |f(q,d)\rangle
$$

位相項：

$$
e^{-i\gamma(c \cdot q + \lambda \cdot f(q,d))} = e^{-i\gamma C(q, d)}
$$

ここで $C(q, d) = c \cdot q + \lambda \cdot \mathbb{1}_{d > q}$ は、特定の需要 $d$ に対するコスト。

**Uncompute適用後**：

$$
|\psi_3\rangle = U_{\text{comp}}^\dagger |\psi_2\rangle = \frac{1}{\sqrt{2^{n_q}}} \sum_{d,q} \sqrt{p_d} \, e^{-i\gamma C(q,d)} |d\rangle |q\rangle |0\rangle
$$

**Mixer適用後**：

$$
|\psi_4\rangle = U_M(\beta) |\psi_3\rangle
$$

Mixerは $R_q$ のみに作用するので：

$$
|\psi_4\rangle = \sum_d \sqrt{p_d} \, |d\rangle \otimes \left[ U_M(\beta) \sum_q \frac{e^{-i\gamma C(q,d)}}{\sqrt{2^{n_q}}} |q\rangle \right] \otimes |0\rangle
$$

## 7. p層QAOAの状態

### 7.1 最終状態

$$
|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{i=1}^{p} U(\beta_i, \gamma_i) \, |\psi_0\rangle
$$

ここで $\boldsymbol{\gamma} = (\gamma_1, \ldots, \gamma_p)$、$\boldsymbol{\beta} = (\beta_1, \ldots, \beta_p)$。

### 7.2 状態の一般形

$$
|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \sum_{d,q} \sqrt{p_d} \, A_q^{(p)}(\boldsymbol{\gamma}, \boldsymbol{\beta} | d) \, |d\rangle |q\rangle |0\rangle
$$

ここで $A_q^{(p)}(\boldsymbol{\gamma}, \boldsymbol{\beta} | d)$ は、パラメータ $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ と需要 $d$ に依存する複素振幅。

### 7.3 測定確率

$R_q$ レジスタを測定したとき、発注量 $q$ が観測される確率：

$$
\Pr_{\text{QAOA}}(q \,|\, \boldsymbol{\gamma}, \boldsymbol{\beta}) = \sum_d p_d \, |A_q^{(p)}(\boldsymbol{\gamma}, \boldsymbol{\beta} | d)|^2
$$

これは、すべての需要シナリオ $d$ に対して確率 $p_d$ で重み付けした平均。

## 8. 最適化する目的関数

### 8.1 期待コストの定式化

QAOAで最適化する実際の目的関数：

$$
\mathcal{L}(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \mathbb{E}_q[C(q)] = \sum_{q=0}^{Q_{\max}} \Pr_{\text{QAOA}}(q \,|\, \boldsymbol{\gamma}, \boldsymbol{\beta}) \cdot C(q)
$$

期待コストの展開：

$$
\mathcal{L}(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \sum_q \left[ \sum_d p_d |A_q^{(p)}(\boldsymbol{\gamma}, \boldsymbol{\beta} | d)|^2 \right] \left[ c \cdot q + \lambda \sum_{d' > q} p_{d'} \right]
$$

### 8.2 最適化問題

$$
(\boldsymbol{\gamma}^*, \boldsymbol{\beta}^*) = \arg\min_{(\boldsymbol{\gamma}, \boldsymbol{\beta}) \in \mathbb{R}^{2p}} \mathcal{L}(\boldsymbol{\gamma}, \boldsymbol{\beta})
$$

通常、パラメータ空間を制約：$\gamma_i, \beta_i \in [-\pi, \pi]$

### 8.3 量子ハミルトニアンとの関係

量子ハミルトニアンの期待値：

$$
\langle H_C \rangle = \langle \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) | H_C | \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) \rangle
$$

展開すると：

$$
\langle H_C \rangle = c \langle \hat{q} \rangle + \lambda \langle \hat{f} \rangle
$$

Comparatorが $f = \mathbb{1}_{d > q}$ を計算しているため、Uncompute前の瞬間的な状態では：

$$
\langle \hat{f} \rangle \approx \sum_{d,q} p_d \, |A_q|^2 \cdot \mathbb{1}_{d > q}
$$

しかし、実際に最適化されるのは測定後の期待コスト $\mathcal{L}$ であり、$\langle H_C \rangle$ とは微妙に異なる。

## 9. QAOAの動作原理：位相と干渉

### 9.1 位相エンコーディング

Cost Oracleは、各 $(q, d)$ ペアに対してコストに比例した位相を付与：

$$
|d\rangle |q\rangle \xrightarrow{U_C(\gamma)} e^{-i\gamma C(q,d)} |d\rangle |q\rangle
$$

**高コスト**の状態 → 大きな負の位相
**低コスト**の状態 → 小さな負の位相

### 9.2 量子干渉による振幅調整

Mixerは $R_q$ レジスタ内で重ね合わせを生成し、異なる $|q\rangle$ 状態間で干渉を引き起こす。

複数層の繰り返しにより：
- **建設的干渉**：低コストな $q$ の振幅が増幅
- **破壊的干渉**：高コストな $q$ の振幅が減衰

### 9.3 最適パラメータでの状態

最適化後、$A_q^{(p)}$ は以下を満たすように調整される：

$$
|A_{q^*}^{(p)}|^2 \gg |A_{q \neq q^*}^{(p)}|^2
$$

ここで $q^*$ は最適発注量。

## 10. FSLとQAOAの役割の数学的分離

### 10.1 FSLの役割：確率分布の量子符号化

**入力**：離散確率分布 $\{p_d\}_{d=0}^{D_{\max}}$ または連続密度関数 $f(x)$

**出力**：量子状態

$$
|\psi_{\text{demand}}\rangle = \sum_{d=0}^{D_{\max}} \sqrt{p_d} \, |d\rangle
$$

**数学的意味**：
- 確率空間 $(\Omega, \mathcal{F}, P)$ を量子ヒルベルト空間 $\mathcal{H}$ に埋め込む
- 確率測度 $P$ を量子振幅 $\sqrt{P}$ として表現

**なぜフーリエ級数を使うのか**：
- 複雑な確率分布を少数のパラメータ（フーリエ係数）で近似
- IQFTによる効率的な量子状態生成（$O(\text{poly}(n))$ ゲート）

### 10.2 QAOAの役割：変分最適化

**入力**：
- 固定された需要状態 $|\psi_{\text{demand}}\rangle$
- 初期パラメータ $(\boldsymbol{\gamma}_0, \boldsymbol{\beta}_0)$

**出力**：最適パラメータ $(\boldsymbol{\gamma}^*, \boldsymbol{\beta}^*)$ と対応する最適発注量分布

**数学的意味**：
- 変分法によるヒルベルト空間内の最適化
- パラメータ化されたユニタリ演算子の族 $\{U(\boldsymbol{\gamma}, \boldsymbol{\beta})\}$ の中で最適なものを探索

**なぜQAOAなのか**：
- 量子並列性：すべての $(q, d)$ ペアを同時に評価
- 干渉効果：自然に最適解へ収束

### 10.3 情報の流れ

$$
\{p_d\} \xrightarrow{\text{FSL}} |\psi_{\text{demand}}\rangle \xrightarrow{\text{QAOA}} |\psi(\boldsymbol{\gamma}^*, \boldsymbol{\beta}^*)\rangle \xrightarrow{\text{測定}} q^*
$$

各ステップの数学的性質：
1. **FSL**：古典確率分布 → 量子状態（線形写像）
2. **QAOA**：初期状態 → 最適化状態（ユニタリ変換 + 古典最適化）
3. **測定**：量子状態 → 古典ビット列（確率的射影）

## 11. 最適化される量の本質

### 11.1 3つの視点

#### (1) 量子力学的視点

ハミルトニアン $H_C$ の最小固有値に近い期待値を持つ状態を探索：

$$
\min_{\boldsymbol{\gamma}, \boldsymbol{\beta}} \langle \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) | H_C | \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) \rangle
$$

#### (2) 統計力学的視点

Gibbs分布 $\rho(\gamma) = e^{-\gamma H_C} / Z$ に近い状態を断熱的に生成。

#### (3) 古典最適化的視点（実装上の真の目的）

測定統計から計算される期待コスト：

$$
\min_{\boldsymbol{\gamma}, \boldsymbol{\beta}} \mathcal{L}(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \min_{\boldsymbol{\gamma}, \boldsymbol{\beta}} \sum_q \Pr_{\text{QAOA}}(q) \cdot C(q)
$$

### 11.2 実装での目的関数の評価

1. パラメータ $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ で量子回路を構築
2. $N_{\text{shot}}$ 回測定して $\{q_i\}_{i=1}^{N_{\text{shot}}}$ を取得
3. 経験分布を計算：$\hat{P}(q) = \frac{1}{N_{\text{shot}}} \sum_{i=1}^{N_{\text{shot}}} \mathbb{1}_{q_i = q}$
4. 期待コストを評価：$\hat{\mathcal{L}} = \sum_q \hat{P}(q) \cdot C(q)$
5. COBYLA等で $\hat{\mathcal{L}}$ を最小化

### 11.3 量子と古典の境界

**量子的に実行**：
- 状態準備（FSL）
- ユニタリ演算（QAOA層）
- 測定

**古典的に実行**：
- コスト関数 $C(q)$ の評価
- 期待値の計算
- パラメータ最適化

## 12. 波動関数の数学的形状

### 12.1 初期状態（完全な形）

$$
|\psi_0\rangle = \frac{1}{\sqrt{2^{n_q}}} \sum_{d=0}^{D_{\max}} \sum_{q=0}^{Q_{\max}} \sqrt{p_d} \, |d, q, 0, \mathbf{0}\rangle
$$

ここで $\mathbf{0}$ は補助qubitの状態。

**密度行列表現**：

$$
\rho_0 = |\psi_0\rangle \langle \psi_0| = \rho_d \otimes \rho_q \otimes |0\rangle\langle 0| \otimes |\mathbf{0}\rangle\langle\mathbf{0}|
$$

ここで：

$$
\rho_d = \sum_d p_d |d\rangle\langle d|, \quad \rho_q = \frac{1}{2^{n_q}} \sum_q |q\rangle\langle q|
$$

### 12.2 第1層QAOA後

Comparator後（位相付与前）：

$$
|\psi_{\text{comp}}\rangle = \frac{1}{\sqrt{2^{n_q}}} \sum_{d,q} \sqrt{p_d} \, |d, q, f(q,d), \mathbf{a}_{q,d}\rangle
$$

ここで $\mathbf{a}_{q,d}$ は比較の中間結果を保持する補助qubitの状態。

Cost Oracle後：

$$
|\psi_{\text{cost}}\rangle = \frac{1}{\sqrt{2^{n_q}}} \sum_{d,q} \sqrt{p_d} \, e^{-i\gamma_1 [c \cdot q + \lambda \cdot f(q,d)]} |d, q, f(q,d), \mathbf{a}_{q,d}\rangle
$$

Uncompute後：

$$
|\psi_{\text{unc}}\rangle = \frac{1}{\sqrt{2^{n_q}}} \sum_{d,q} \sqrt{p_d} \, e^{-i\gamma_1 C(q,d)} |d, q, 0, \mathbf{0}\rangle
$$

Mixer後（$R_q$ のみ変換）：

$$
|\psi_1\rangle = \sum_d \sqrt{p_d} \, |d\rangle \otimes \left[ \sum_q A_q^{(1)}(d) |q\rangle \right] \otimes |0, \mathbf{0}\rangle
$$

ここで：

$$
A_q^{(1)}(d) = \langle q | U_M(\beta_1) \left[ \frac{1}{\sqrt{2^{n_q}}} \sum_{q'} e^{-i\gamma_1 C(q',d)} |q'\rangle \right]
$$

### 12.3 p層後の最終状態

$$
|\psi_{\text{final}}\rangle = \sum_d \sqrt{p_d} \, |d\rangle \otimes \left[ \sum_q A_q^{(p)}(\boldsymbol{\gamma}, \boldsymbol{\beta} | d) |q\rangle \right] \otimes |0, \mathbf{0}\rangle
$$

**振幅の再帰的定義**：

$$
A_q^{(i+1)}(d) = \sum_{q'} \langle q | U_M(\beta_{i+1}) | q' \rangle \cdot e^{-i\gamma_{i+1} C(q',d)} \cdot A_{q'}^{(i)}(d)
$$

初期条件：$A_q^{(0)}(d) = 1/\sqrt{2^{n_q}}$

### 12.4 波動関数の漸近的性質

最適化が成功した場合、$p \to \infty$ かつ最適パラメータで：

$$
A_q^{(\infty)}(\boldsymbol{\gamma}^*, \boldsymbol{\beta}^* | d) \approx \begin{cases}
\alpha_d & q = q^* \\
\epsilon_q(d) & q \neq q^*
\end{cases}
$$

ここで $|\alpha_d|^2 \approx 1$ かつ $|\epsilon_q(d)|^2 \ll 1$。

すなわち、すべての需要シナリオ $d$ に対して、最適な $q^*$ の振幅が支配的になる。

## 13. 数学的まとめ

### 13.1 解いている問題

**古典問題**：

$$
\min_{q \in \{0, \ldots, Q_{\max}\}} \left\{ c \cdot q + \lambda \sum_{d > q} p_d \right\}
$$

**量子変分問題**：

$$
\min_{(\boldsymbol{\gamma}, \boldsymbol{\beta})} \sum_q \left[ \sum_d p_d |A_q^{(p)}(\boldsymbol{\gamma}, \boldsymbol{\beta} | d)|^2 \right] \left[ c \cdot q + \lambda \sum_{d' > q} p_{d'} \right]
$$

### 13.2 FSLの数学的役割

写像 $\Phi: \mathcal{P}(\mathbb{N}) \to \mathcal{H}$ を定義：

$$
\Phi(\{p_d\}) = \sum_d \sqrt{p_d} \, |d\rangle
$$

ここで $\mathcal{P}(\mathbb{N})$ は確率分布の空間、$\mathcal{H}$ は量子ヒルベルト空間。

**性質**：
- 線形性：$\Phi(\alpha P_1 + (1-\alpha) P_2) \neq \alpha \Phi(P_1) + (1-\alpha) \Phi(P_2)$（非線形）
- 内積保存：$|\langle \Phi(P_1) | \Phi(P_2) \rangle|^2 = \sum_d \sqrt{p_d^{(1)} p_d^{(2)}}$（fidelity）

### 13.3 QAOAの数学的役割

パラメータ化されたユニタリ変換の族：

$$
\mathcal{U} = \{ U(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \prod_{i=1}^p U_M(\beta_i) U_{\text{comp}}^\dagger U_C(\gamma_i) U_{\text{comp}} \}
$$

最適化：

$$
U^* = \arg\min_{U \in \mathcal{U}} \mathbb{E}_{q \sim |U|\psi_0\rangle|^2}[C(q)]
$$

### 13.4 全体のアルゴリズム構造

$$
\boxed{
\begin{aligned}
&\text{Input: } \{p_d\}, c, \lambda \\
&\text{1. FSL: } \{p_d\} \xrightarrow{\text{Fourier}} \{c_k\} \xrightarrow{\text{Quantum}} |\psi_d\rangle = \sum_d \sqrt{p_d} |d\rangle \\
&\text{2. Initialize: } |\psi_0\rangle = |\psi_d\rangle \otimes |+\rangle^{\otimes n_q} \otimes |0\rangle \\
&\text{3. Optimize: } \\
&\qquad (\boldsymbol{\gamma}^*, \boldsymbol{\beta}^*) = \arg\min \mathcal{L}(\boldsymbol{\gamma}, \boldsymbol{\beta}) \\
&\qquad \text{where } \mathcal{L}(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \mathbb{E}_{q \sim \text{Measure}(U(\boldsymbol{\gamma}, \boldsymbol{\beta})|\psi_0\rangle)}[C(q)] \\
&\text{4. Measure: } q^* \sim |\langle q | U(\boldsymbol{\gamma}^*, \boldsymbol{\beta}^*)|\psi_0\rangle|^2 \\
&\text{Output: } q^*
\end{aligned}
}
$$

---

## 参考：主要な数式一覧

| 概念 | 数式 |
|------|------|
| コスト関数 | $C(q) = c \cdot q + \lambda \cdot \Pr(D > q)$ |
| 需要の量子状態 | $\|\psi_{\text{demand}}\rangle = \sum_d \sqrt{p_d} \, \|d\rangle$ |
| コストハミルトニアン | $H_C = c \cdot \hat{q} + \lambda \cdot \hat{f}$ |
| 混合ハミルトニアン | $H_M = \sum_j X_j$ |
| QAOAユニタリ | $U(\beta, \gamma) = U_M(\beta) U_{\text{comp}}^\dagger U_C(\gamma) U_{\text{comp}}$ |
| 最終状態 | $\|\psi_{\text{final}}\rangle = \prod_{i=1}^p U(\beta_i, \gamma_i) \|\psi_0\rangle$ |
| 測定確率 | $\Pr(q) = \sum_d p_d \|A_q^{(p)}(d)\|^2$ |
| 目的関数 | $\mathcal{L}(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \sum_q \Pr(q) \cdot C(q)$ |

**実装ファイル**：`notebooks/newsvendor.py`
