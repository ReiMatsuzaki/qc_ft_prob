"""
QAOA回路の各部分を詳しく説明するスクリプト
"""
import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsvendor import (
    solve_newsvendor_qaoa,
    build_full_qaoa_circuit,
    encode_demand_distribution
)

def analyze_circuit_structure():
    """回路の構造を段階的に説明"""
    print("=" * 70)
    print("QAOA Circuit Structure - Detailed Analysis")
    print("=" * 70)

    # 小さい問題設定
    mu, sigma = 3, 1
    D_max = 5
    Q_max = 5
    c = 1.0
    lam = 5.0

    demand_pdf = lambda x: norm.pdf(x, mu, sigma)

    # FSLエンコーディング
    M = 2
    ks, cs, meta = encode_demand_distribution(demand_pdf, D_max, M=M)

    # レジスタサイズ
    n_q = int(np.ceil(np.log2(Q_max + 1)))
    n_d = int(np.ceil(np.log2(D_max + 1)))

    print(f"\nProblem Size:")
    print(f"  Q_max = {Q_max}, D_max = {D_max}")
    print(f"  n_q = {n_q} qubits, n_d = {n_d} qubits")

    print(f"\nRegister Layout:")
    print(f"  q[0-{n_q-1}]:   R_q (order quantity)")
    print(f"  q[{n_q}-{n_q+n_d-1}]:   R_d (demand)")
    print(f"  q[{n_q+n_d}]:     R_f (stockout flag)")
    print(f"  q[{n_q+n_d+1}+]:  Ancilla")

    # QAOA回路を構築
    gammas = [0.5]  # p=1
    betas = [0.2]

    circuit = build_full_qaoa_circuit(
        gammas, betas, n_q, n_d, ks, cs, c, lam, Q_max, D_max
    )

    total_gates = circuit.get_gate_count()
    print(f"\nTotal Gates: {total_gates}")

    # ゲートを段階的に分析
    print("\n" + "=" * 70)
    print("GATE-BY-GATE ANALYSIS")
    print("=" * 70)

    # セクションの境界を特定するために、ゲートの種類と対象qubitを分析
    sections = []
    current_section = {"start": 0, "gates": [], "name": ""}

    hadamard_count = 0
    fsl_started = False
    comparator_started = False
    cost_started = False
    uncomp_started = False
    mixer_started = False

    for i in range(total_gates):
        gate = circuit.get_gate(i)
        gate_name = gate.get_name()
        targets = gate.get_target_index_list()
        controls = gate.get_control_index_list()

        # Hadamard on R_q (初期化)
        if gate_name == 'H' and not fsl_started and len(targets) == 1:
            if targets[0] < n_q:
                hadamard_count += 1
                if hadamard_count == 1:
                    current_section = {"start": i, "gates": [gate_name], "name": "R_q Initialization (Hadamard)"}
                else:
                    current_section["gates"].append(gate_name)

                if hadamard_count == n_q:
                    sections.append({**current_section, "end": i})
                    fsl_started = True
                continue

        # FSL回路（R_dに作用）
        if fsl_started and not comparator_started:
            # FSL回路は主にDenseMatrix, H, CNOT, SWAP, RZなどを使う
            # R_dレジスタ (q[3-5])に作用
            if any(t >= n_q and t < n_q + n_d for t in targets):
                if not current_section.get("name") or current_section["name"].startswith("R_q"):
                    sections.append(current_section)
                    current_section = {"start": i, "gates": [gate_name], "name": "FSL Circuit (Demand Distribution)"}
                else:
                    current_section["gates"].append(gate_name)
                continue
            else:
                # FSL終了
                if current_section.get("name", "").startswith("FSL"):
                    sections.append({**current_section, "end": i-1})
                    comparator_started = True
                    current_section = {"start": i, "gates": [gate_name], "name": "Comparator Circuit"}
                continue

        # 比較回路（ancillaやR_fに作用開始）
        if comparator_started and not cost_started:
            # X, TOFFOLI, CNOTなどがancillaやR_fに作用
            if any(t >= n_q + n_d for t in targets) or gate_name in ['X', 'Toffoli', 'CNOT']:
                current_section["gates"].append(gate_name)
                # RZゲートが出てきたらコストオラクル開始
                if 'rotation' in gate_name or 'RZ' in gate_name:
                    sections.append({**current_section, "end": i-1})
                    cost_started = True
                    current_section = {"start": i, "gates": [gate_name], "name": "Cost Oracle (RZ gates)"}
                continue

        # コストオラクル（RZゲート）
        if cost_started and not uncomp_started:
            if 'rotation' in gate_name or 'RZ' in gate_name:
                current_section["gates"].append(gate_name)
                continue
            else:
                # コストオラクル終了、アンコンピュート開始
                sections.append({**current_section, "end": i-1})
                uncomp_started = True
                current_section = {"start": i, "gates": [gate_name], "name": "Uncompute Comparator"}
                continue

        # アンコンピュート比較回路
        if uncomp_started and not mixer_started:
            if 'X' in gate_name or 'CNOT' in gate_name or 'Toffoli' in gate_name:
                current_section["gates"].append(gate_name)
                continue
            else:
                # アンコンピュート終了、ミキサー開始
                sections.append({**current_section, "end": i-1})
                mixer_started = True
                current_section = {"start": i, "gates": [gate_name], "name": "Mixer Circuit (RX gates)"}
                continue

        # ミキサー回路（RXゲート）
        if mixer_started:
            current_section["gates"].append(gate_name)

    # 最後のセクションを追加
    if current_section.get("gates"):
        sections.append({**current_section, "end": total_gates-1})

    # セクションごとに表示
    print("\nCircuit Sections:")
    print("-" * 70)
    for idx, section in enumerate(sections, 1):
        start = section["start"]
        end = section.get("end", "?")
        name = section["name"]
        n_gates = len(section["gates"])
        gate_types = set(section["gates"])

        print(f"\n{idx}. {name}")
        print(f"   Gate Range: {start} - {end}")
        print(f"   Number of Gates: {n_gates}")
        print(f"   Gate Types: {', '.join(sorted(gate_types))}")

    # 詳細説明
    print("\n" + "=" * 70)
    print("DETAILED EXPLANATION")
    print("=" * 70)

    print("""
1. R_q Initialization (Hadamard)
   --------------------------------
   目的: 発注量qを均一重ね合わせ状態に初期化

   |0⟩^⊗n_q → |+⟩^⊗n_q = (1/√2^n_q) Σ_q |q⟩

   これにより、すべての可能な発注量qが等確率で重ね合わされます。

2. FSL Circuit (Demand Distribution)
   -----------------------------------
   目的: R_dレジスタに需要分布をエンコード

   構造:
   - フーリエ係数をユニタリ変換として実装
   - IQFTで計算基底の確率分布に変換

   |0⟩^⊗n_d → Σ_d √p_d |d⟩

   ここで p_d は需要分布（ガウス分布）の確率です。

3. Comparator Circuit
   --------------------
   目的: d > q を判定し、R_fに結果を記録

   |q⟩|d⟩|0⟩_f → |q⟩|d⟩|f⟩  where f = 1[d>q]

   実装:
   - Ancilla qubitを使用
   - 各ビット位置でd_i > q_iを判定
   - TOFFOLIゲートでAND操作を実現
   - 結果をR_fに集約（CNOT）

4. Cost Oracle (RZ gates)
   ------------------------
   目的: コスト関数をエンコード

   H_C = c·q + λ·f

   実装:
   - R_qの各qubit j に RZ(-γ·c·2^j) を適用
   - R_f に RZ(-γ·λ) を適用

   これにより、状態 |q,d,f⟩ に位相 exp(-iγ[c·q + λ·f]) が付与されます。

5. Uncompute Comparator
   ----------------------
   目的: Ancilla qubitをクリーン状態に戻す

   |q⟩|d⟩|f⟩|anc⟩ → |q⟩|d⟩|0⟩_f|0⟩_anc

   比較回路の逆操作を実行。これは可逆計算のために必要です。

6. Mixer Circuit (RX gates)
   --------------------------
   目的: R_qレジスタに混合を適用

   H_M = Σ_j X_j (R_qのみ)

   実装:
   - R_qの各qubit j に RX(2β) を適用

   これにより、異なるq値間の遷移（重ね合わせの混合）が可能になります。
   R_dとR_fには作用しません（需要分布は固定）。
""")

    print("\n" + "=" * 70)
    print("QAOA PARAMETERS IN THE CIRCUIT")
    print("=" * 70)
    print(f"""
gamma (γ) = {gammas[0]:.4f}
  → Cost Oracleのパラメータ
  → RZゲートの角度: -γ·c·2^j (R_q), -γ·λ (R_f)

beta (β) = {betas[0]:.4f}
  → Mixerのパラメータ
  → RXゲートの角度: 2β (R_q)

これらのパラメータは古典的な最適化によって調整され、
期待コスト ⟨H_C⟩ を最小化します。
""")

if __name__ == "__main__":
    analyze_circuit_structure()
