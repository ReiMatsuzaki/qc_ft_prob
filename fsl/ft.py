import numpy as np

def fourier_series_coeffs(func, T=1.0, M=32, N=None, x0=0.0):
    """
    フーリエ求数
        f(x) ≈ Σ_{k=-M..M} c_k exp(-2π i k x / T)
    の係数を計算する

    Args:
        func : callable, 複素値可。ベクトル化推奨。
        T    : 周期
        M    : 取り出す最大次数（-M..M）
        N    : サンプル数（既定は 2*M+1（奇数））
        x0   : サンプリング開始点（位相補正により c_k は開始点に依存しない形で返す）
    
    Returns:
        ks (整数配列): kのリスト
        cs (複素係数配列): 係数c_kのリスト
        meta dict
    """
    if N is None:
        N = 2*M + 1  # 推奨：奇数 → kがちょうど -M..M
    if M > N//2:
        raise ValueError("M は N/2 以下にしてください。")

    # 等間隔サンプリング（1周期）
    ns = np.arange(N)
    xs = x0 + (T * ns / N)
    fs = np.asarray(func(xs), dtype=complex)

    # DFT → 係数（Riemann和より c_k ≈ (1/N) Σ f_n e^{-2π i k n/N}）
    F = np.fft.fft(fs) / N          # k=0..N-1
    Fc = np.fft.fftshift(F)        # k を中心化

    # 対応する整数 k 配列
    if N % 2 == 1:
        k_all = np.arange(-(N//2), N//2 + 1)   # 例: N=2M+1 → -M..M
    else:
        k_all = np.arange(-N//2, N//2)         # 偶数の場合

    # サンプリング開始 x0 の位相補正（基底は x に対する e^{-2π i k x/T}）
    phase = np.exp(2j * np.pi * k_all * (x0 / T))
    c_all = Fc * phase

    # -M..M を抽出
    pick = (k_all >= -M) & (k_all <= M)
    ks = k_all[pick]
    cs = c_all[pick]
    return ks, cs, {"T": T, "M": M, "N": N, "x0": x0}

def fourier_series_eval(xs, ks, cs, T):
    """
    フーリエ級数の係数から関数値を評価する
        f(x) ≈ Σ_{k} c_k exp(-2π i k x / T)

    Args:
        xs : array_like, 評価点
        ks : array, フーリエ係数のインデックス（整数配列）
        cs : array, フーリエ係数（複素数配列）
        T  : 周期

    Returns:
        array: 各評価点での関数値（複素数配列）
    """
    xs = np.asarray(xs)
    return np.sum(cs[:, None] * np.exp(-2j * np.pi * ks[:, None] * (xs / T)), axis=0)

def periodic_extension(f_base, L):
    """
    関数を周期的に拡張する

    区間 [-L, L] で定義された関数 f_base を周期 2L で周期的に拡張する。

    Args:
        f_base : callable, 元の関数（[-L, L] で定義）
        L      : 半周期（周期は 2L）

    Returns:
        callable: 周期拡張された関数
    """
    return lambda x: f_base(((np.asarray(x)+L) % (2*L)) - L)

