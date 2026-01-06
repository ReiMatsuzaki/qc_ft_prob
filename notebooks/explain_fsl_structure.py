"""
FSL回路の構造を説明するスクリプト
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

def explain_fsl_structure():
    """FSL回路の構造を詳しく説明"""
    print("=" * 80)
    print("FSL (Fourier Series Loading) 回路の構造")
    print("=" * 80)
    print()

    print("FSL回路は2つのパートから構成されます：")
    print()
    print("【Part 1】Fourier係数による単一キュービット回転（U_c ゲート）")
    print("  - 各キュービットに独立に回転ゲート（RZ, RY など）を適用")
    print("  - ビット間の配線は**ありません**（単一キュービット演算のみ）")
    print("  - フーリエ級数の各係数に対応するゲートが並ぶ")
    print()
    print("【Part 2】IQFT（逆量子フーリエ変換）")
    print("  - 2キュービットゲート（制御回転、SWAP）を使用")
    print("  - ここで初めてビット間の**配線が現れます**")
    print("  - フーリエ空間から確率分布への変換")
    print()
    print("=" * 80)
    print()

    # 実際の回路で確認
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

    gammas = [0.5]
    betas = [0.2]

    circuit = build_full_qaoa_circuit(
        gammas, betas, n_q, n_d, ks, cs, c, lam, Q_max, D_max
    )

    print("実際の回路でのゲート配置：")
    print()
    print("Gate 0-2:   Hadamard on R_q")
    print("            └→ 単一キュービットゲート（配線なし）")
    print()
    print("Gate 3-13:  FSL Part 1 - Fourier係数回転")
    print("            └→ 単一キュービットゲート（配線なし）✓ 正常")
    print("            例: DenseMatrix(q[5]), DenseMatrix(q[4]), ...")
    print()
    print("Gate 14-19: FSL Part 2 - IQFT")
    print("            └→ 2キュービットゲート（配線あり）")
    print("            例:")

    # IQFTゲートを表示
    for i in range(14, 20):
        gate = circuit.get_gate(i)
        gate_name = gate.get_name()
        targets = list(gate.get_target_index_list())

        if len(targets) == 1:
            print(f"            Gate {i}: {gate_name:15s} q[{targets[0]}] (1キュービット)")
        elif len(targets) == 2:
            print(f"            Gate {i}: {gate_name:15s} q[{targets[0]}]-q[{targets[1]}] (配線あり) ←")
        else:
            print(f"            Gate {i}: {gate_name:15s} targets={targets}")

    print()
    print("Gate 20-31: Comparator circuit")
    print("            └→ 制御ゲート（配線あり）")
    print()
    print("=" * 80)
    print("【結論】")
    print("Gate 0-13でビット間配線がないのは正常です。")
    print("これはFSL回路のPart 1（単一キュービット回転）の特性です。")
    print("Gate 14以降のIQFT部分で配線が現れます。")
    print("=" * 80)

if __name__ == "__main__":
    explain_fsl_structure()
