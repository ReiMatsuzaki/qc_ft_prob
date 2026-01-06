"""
回路の全ゲートを詳細にプリントして構造を理解する
"""
import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsvendor import (
    build_full_qaoa_circuit,
    encode_demand_distribution
)

def print_all_gates():
    """すべてのゲートを詳細に出力"""
    # 設定
    mu, sigma = 3, 1
    D_max = 5
    Q_max = 5
    c = 1.0
    lam = 5.0
    M = 2

    demand_pdf = lambda x: norm.pdf(x, mu, sigma)
    ks, cs, meta = encode_demand_distribution(demand_pdf, D_max, M=M)

    n_q = 3
    n_d = 3

    print("Register Layout (NEW - R_d first):")
    print("  q[0-2]: R_d (demand)")
    print("  q[3-5]: R_q (order quantity)")
    print("  q[6]:   R_f (stockout flag)")
    print("  q[7-9]: Ancilla")
    print()

    # 回路を構築
    gammas = [0.5]
    betas = [0.2]

    circuit = build_full_qaoa_circuit(
        gammas, betas, n_q, n_d, ks, cs, c, lam, Q_max, D_max
    )

    total_gates = circuit.get_gate_count()
    print(f"Total Gates: {total_gates}\n")
    print("=" * 80)

    # すべてのゲートを出力
    for i in range(min(60, total_gates)):  # 最初の60ゲートを表示
        gate = circuit.get_gate(i)
        gate_name = gate.get_name()
        targets = list(gate.get_target_index_list())
        controls = list(gate.get_control_index_list())

        # ターゲットの説明 (NEW layout: R_d first)
        target_desc = []
        for t in targets:
            if t < 3:
                target_desc.append(f"q[{t}](R_d)")
            elif t < 6:
                target_desc.append(f"q[{t}](R_q)")
            elif t == 6:
                target_desc.append(f"q[{t}](R_f)")
            else:
                target_desc.append(f"q[{t}](anc)")

        # コントロールの説明 (NEW layout: R_d first)
        control_desc = []
        for c in controls:
            if c < 3:
                control_desc.append(f"q[{c}](R_d)")
            elif c < 6:
                control_desc.append(f"q[{c}](R_q)")
            elif c == 6:
                control_desc.append(f"q[{c}](R_f)")
            else:
                control_desc.append(f"q[{c}](anc)")

        print(f"Gate {i:3d}: {gate_name:20s}", end="")
        if targets:
            print(f" targets: {', '.join(target_desc):30s}", end="")
        if controls:
            print(f" controls: {', '.join(control_desc):30s}", end="")
        print()

        # セクション区切りの推測 (NEW layout)
        if i == 16:
            print("\n" + "-" * 80)
            print("↑ FSL Circuit (on R_d)")
            print("↓ R_q Initialization (Hadamard)")
            print("-" * 80 + "\n")

    if total_gates > 60:
        print(f"\n... ({total_gates - 60} more gates)")

if __name__ == "__main__":
    print_all_gates()
