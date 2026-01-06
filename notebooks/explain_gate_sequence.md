# QAOA回路のゲートシーケンス解説

## レジスタ配置（NEW - R_d first）
- **q[0-2]**: R_d (demand) - 需要レジスタ
- **q[3-5]**: R_q (order quantity) - 発注量レジスタ
- **q[6]**: R_f (stockout flag) - 欠品フラグ
- **q[7-9]**: Ancilla - 補助量子ビット

---

## セクション1: FSL回路（Gate 0-16）- 需要分布の量子状態準備

### Part 1a: Fourier係数による制御回転（Gate 0-9）

**目的**: フーリエ級数の係数c_kを量子状態に埋め込む

```
Gate 0-1: DenseMatrix targets: q[2](R_d) controls: q[0](R_d), q[1](R_d)
  └→ 3キュービット制御ゲート: フーリエ係数c_k (k=4,5)に対応
  └→ q[0],q[1]の状態に応じてq[2]を回転

Gate 2-5: DenseMatrix targets: q[1](R_d) controls: q[0](R_d), q[2](R_d)
  └→ フーリエ係数c_k (k=2,3,6,7)に対応
  └→ q[0],q[2]の状態に応じてq[1]を回転

Gate 6-9: DenseMatrix targets: q[0](R_d) controls: q[1](R_d), q[2](R_d)
  └→ フーリエ係数c_k (k=1,3,5,7)に対応
  └→ q[1],q[2]の状態に応じてq[0]を回転
```

**重要**: ここでビット間の配線（制御ゲート）が**正しく保持されています** ✓

### Part 1b: IQFT（逆量子フーリエ変換）（Gate 10-16）

**目的**: フーリエ空間から確率分布への変換

```
Gate 10: H targets: q[2](R_d)
  └→ IQFT開始（最上位ビット）

Gate 11: DenseMatrix targets: q[2], q[1]
  └→ 制御回転: R_1^2 = exp(iπ/2) 型

Gate 12: H targets: q[1](R_d)
  └→ 中位ビットのHadamard

Gate 13: DenseMatrix targets: q[2], q[0]
  └→ 制御回転: R_1^4 = exp(iπ/4) 型

Gate 14: DenseMatrix targets: q[1], q[0]
  └→ 制御回転: R_1^2 型

Gate 15: H targets: q[0](R_d)
  └→ 最下位ビットのHadamard

Gate 16: SWAP targets: q[0], q[2]
  └→ ビット順序の反転（IQFTの最終ステップ）
```

**結果**: |ψ⟩_d = Σ_d √p_d |d⟩ （需要分布がR_dに準備された）

---

## セクション2: R_q初期化（Gate 17-19）- 発注量の重ね合わせ

```
Gate 17: H targets: q[3](R_q)
Gate 18: H targets: q[4](R_q)
Gate 19: H targets: q[5](R_q)
```

**目的**: 発注量qの均等な重ね合わせ状態を作成

**結果**: |ψ⟩_q = (1/√8) Σ_{q=0}^7 |q⟩

**全体状態**: |ψ⟩ = |ψ⟩_d ⊗ |ψ⟩_q （需要と発注量の積状態）

---

## セクション3: Comparator回路（Gate 20-31）- 欠品判定 d > q

### Phase 1: ビットごとの比較準備（Gate 20-28）

**目的**: 各ビット位置でd_i と q_i を比較し、補助ビットに結果を保存

```
Gate 20: X targets: q[5](R_q)
  └→ MSBの反転（比較のため）

Gate 21: DenseMatrix targets: q[9](anc) controls: q[2](R_d), q[5](R_q)
  └→ anc[9] = d_2 XOR q_2（MSB比較）

Gate 22: X targets: q[5](R_q)
  └→ 元に戻す

Gate 23: X targets: q[4](R_q)
Gate 24: DenseMatrix targets: q[8](anc) controls: q[1](R_d), q[4](R_q)
  └→ anc[8] = d_1 XOR q_1（中位ビット比較）

Gate 25: X targets: q[4](R_q)

Gate 26: X targets: q[3](R_q)
Gate 27: DenseMatrix targets: q[7](anc) controls: q[0](R_d), q[3](R_q)
  └→ anc[7] = d_0 XOR q_0（LSB比較）

Gate 28: X targets: q[3](R_q)
```

### Phase 2: 欠品フラグの計算（Gate 29-31）

**目的**: 補助ビットの結果を集約してR_fに欠品判定を保存

```
Gate 29: CNOT targets: q[6](R_f) controls: q[7](anc)
  └→ R_f に LSB比較結果を追加

Gate 30: CNOT targets: q[6](R_f) controls: q[8](anc)
  └→ R_f に中位ビット比較結果を追加

Gate 31: CNOT targets: q[6](R_f) controls: q[9](anc)
  └→ R_f に MSB比較結果を追加
```

**結果**: R_f = 1 ⟺ d > q （欠品状態）

---

## セクション4: Cost Oracle（Gate 32-35）- コスト関数の位相エンコーディング

**目的**: U_C(γ) = exp(-iγ H_C) を適用、ここで H_C = c·q̂ + λ·f̂

### 発注コスト項（Gate 32-34）

```
Gate 32: Z-rotation targets: q[3](R_q)
  └→ RZ(angle = -γ·c·2^0) = RZ(-γ·c·1)

Gate 33: Z-rotation targets: q[4](R_q)
  └→ RZ(angle = -γ·c·2^1) = RZ(-γ·c·2)

Gate 34: Z-rotation targets: q[5](R_q)
  └→ RZ(angle = -γ·c·2^2) = RZ(-γ·c·4)
```

**意味**: q = q_0·2^0 + q_1·2^1 + q_2·2^2 に対して、位相 exp(-iγ·c·q) を付与

### 欠品ペナルティ項（Gate 35）

```
Gate 35: Z-rotation targets: q[6](R_f)
  └→ RZ(angle = -γ·λ)
```

**意味**: f=1（欠品時）に位相 exp(-iγ·λ) を付与

**結果**: |ψ⟩ → exp(-iγ·C(q,d))|ψ⟩ （コスト関数が位相としてエンコードされた）

---

## セクション5: Uncompute Comparator（Gate 36-47）- 補助ビットのリセット

**目的**: Gate 20-31の逆操作により補助ビットを|0⟩に戻す

```
Gate 36-38: CNOT（逆順）
  └→ R_fから情報を削除（ただしR_fは保持）

Gate 39-47: X gates と DenseMatrix（逆順）
  └→ 補助ビット anc[7-9] を |0⟩ にリセット
```

**重要**: 補助ビットをリセットしないと量子もつれが残り、測定結果が正しくならない

**結果**: anc[7-9] = |000⟩、R_fはそのまま保持

---

## セクション6: Mixer（Gate 48-50）- 発注量空間の探索

**目的**: U_M(β) = exp(-iβ H_M) を適用、ここで H_M = Σ_j X_j

```
Gate 48: X-rotation targets: q[3](R_q)
  └→ RX(angle = 2β)

Gate 49: X-rotation targets: q[4](R_q)
  └→ RX(angle = 2β)

Gate 50: X-rotation targets: q[5](R_q)
  └→ RX(angle = 2β)
```

**意味**: 各発注量ビットに対してX方向の回転を適用

**効果**:
- 発注量qの値を変化させる（|q⟩ → 他のq値への重ね合わせ）
- コスト関数の谷を探索する

---

## 全体の流れ（p=1の場合）

```
初期状態: |0...0⟩

↓ Gate 0-16: FSL
需要分布準備: |ψ_d⟩ = Σ_d √p_d |d⟩

↓ Gate 17-19: Hadamard on R_q
発注量重ね合わせ: |ψ⟩ = Σ_{q,d} √p_d/√8 |d⟩|q⟩

↓ Gate 20-31: Comparator
欠品判定: |ψ⟩ = Σ_{q,d} √p_d/√8 |d⟩|q⟩|1[d>q]⟩

↓ Gate 32-35: Cost Oracle
位相付与: |ψ⟩ = Σ_{q,d} √p_d/√8 exp(-iγC(q,d)) |d⟩|q⟩|1[d>q]⟩

↓ Gate 36-47: Uncompute
補助リセット: |ψ⟩ = Σ_{q,d} √p_d/√8 exp(-iγC(q,d)) |d⟩|q⟩|1[d>q]⟩|0⟩_anc

↓ Gate 48-50: Mixer
発注量探索: |ψ⟩ = より良い発注量に重みが集中

↓ 測定（R_qのみ）
→ 最適な発注量qが高確率で得られる
```

---

## 重要なポイント

### 1. FSL回路の配線が正しく保持されている
- **Gate 0-9**: 制御ゲートによるビット間の配線が**正常に動作** ✓
- Remapを削除してR_dを先頭に配置したことで、2キュービット・3キュービット制御ゲートが正しく適用されている

### 2. 可逆性の確保
- Comparator回路（Gate 20-31）とその逆（Gate 36-47）により、補助ビットの情報が完全に削除される
- これにより、測定時にR_qの確率分布が正しく得られる

### 3. コスト関数のエンコーディング
- **C(q,d) = c·q + λ·1[d>q]** が位相として正確にエンコードされている
- Gate 32-35で位相 exp(-iγ·C(q,d)) が付与される

### 4. 測定戦略
- R_q (q[3-5])のみを測定することで発注量qを得る
- R_dは測定しない（需要分布の情報は位相に埋め込まれている）

---

## パラメータとの関係

- **γ (gamma)**: コスト関数の重み（大きいほど低コスト状態への集中が強い）
- **β (beta)**: 探索の範囲（適度な値で最適解周辺を探索）
- **M=2**: フーリエ打ち切り次数（Gate 0-9のゲート数に影響）
- **p=1**: QAOAの深さ（この例では1層のみ）

p > 1の場合、Gate 20-50のシーケンスが繰り返され、より最適な解に収束する。
