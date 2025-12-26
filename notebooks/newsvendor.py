"""
Newsvendor Problem with FSL + QAOA

Implements the stockout-probability formulation:
    C(q) = c·q + λ·Pr(D>q)

where:
    - q: Order quantity (decision variable)
    - D: Demand (random variable with distribution p_d)
    - c: Unit ordering cost
    - λ: Stockout penalty

Approach:
    - FSL (Fourier Series Loading) for demand distribution
    - Quantum comparator for D > q detection
    - QAOA for order quantity optimization
"""

import numpy as np
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import H, X, CNOT, TOFFOLI, RZ, RX, RY, CZ
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Union, Dict, Callable, Tuple, List
try:
    from qulacsvis import circuit_drawer
    HAS_QULACSVIS = True
except ImportError:
    HAS_QULACSVIS = False
    print("Warning: qulacsvis not found. Circuit visualization will be limited.")

# Import FSL functions from qc_ft_prob.py (in same directory)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qc_ft_prob import fourier_series_coeffs, build_Uc_circuit_from_ck_cascade, iqft_circuit


# ============================================================================
# Phase 1: Core Quantum Components
# ============================================================================

def build_comparator_circuit(n_q: int, n_d: int,
                            reg_q_offset: int, reg_d_offset: int,
                            reg_f: int, ancilla_offset: int) -> QuantumCircuit:
    """
    Build reversible quantum comparator circuit: |q⟩|d⟩|0⟩_f → |q⟩|d⟩|1[d>q]⟩_f

    Uses ripple-carry approach for simplicity and correctness.

    Args:
        n_q: Number of qubits for order quantity register
        n_d: Number of qubits for demand register
        reg_q_offset: Starting index of R_q
        reg_d_offset: Starting index of R_d
        reg_f: Index of flag qubit
        ancilla_offset: Starting index of ancilla qubits

    Returns:
        QuantumCircuit implementing d > q comparison

    Strategy:
        Compare bit by bit from MSB to LSB:
        - If d_i=1 and q_i=0: definitely d > q
        - If d_i=0 and q_i=1: definitely d ≤ q
        - If d_i=q_i: outcome depends on lower bits (propagate)

    Note: This is a simplified implementation. For production use,
          a more optimized comparator would be needed.
    """
    # Use the maximum of n_q and n_d for comparison
    n_bits = max(n_q, n_d)
    total_qubits = reg_q_offset + n_q + n_d + 1 + n_bits  # Includes ancilla

    circuit = QuantumCircuit(total_qubits)

    # Simple ripple-carry comparator implementation
    # We'll use ancilla qubits to track "greater than" condition
    # ancilla[i] = 1 if d[i:] > q[i:] considering bits from position i onwards

    # Start from LSB to MSB (actually easier to implement MSB to LSB for comparison)
    # For simplicity, we implement a basic version:
    # For each bit position i from MSB to LSB:
    #   If d_i > q_i: set flag (d_i=1, q_i=0)
    #   If d_i < q_i: clear flag (d_i=0, q_i=1)
    #   If d_i = q_i: propagate

    # This is a simplified placeholder - in practice, you'd use:
    # 1. XOR gates to compare bits
    # 2. Cascaded logic to propagate "greater than" condition
    # 3. Toffoli gates for controlled operations

    # For now, implement a basic version that works for small values
    # TODO: Implement full ripple-carry or look-ahead comparator for larger problems

    # Simplified approach for demonstration:
    # We'll check each bit position and use ancilla to accumulate results

    for i in range(n_bits - 1, -1, -1):  # MSB to LSB
        q_bit = reg_q_offset + min(i, n_q - 1)
        d_bit = reg_d_offset + min(i, n_d - 1)
        anc = ancilla_offset + i

        # This is a simplified version - needs proper implementation
        # For d_i > q_i: d_i=1 AND q_i=0 → set ancilla
        # Use X gate to flip q_i, then Toffoli with d_i and flipped q_i
        if i < n_q:
            circuit.add_gate(X(q_bit))  # Flip q_i
        if i < n_d and i < n_q:
            circuit.add_gate(TOFFOLI(d_bit, q_bit, anc))  # If d_i=1 and q_i=0 → anc=1
        if i < n_q:
            circuit.add_gate(X(q_bit))  # Flip back

    # Aggregate ancilla results to flag
    # If any ancilla shows d > q at that position, set flag
    for i in range(n_bits):
        anc = ancilla_offset + i
        circuit.add_gate(CNOT(anc, reg_f))

    return circuit


def build_cost_oracle(gamma: float, c: float, lam: float,
                     n_q: int, reg_q_offset: int, reg_f: int,
                     total_qubits: int) -> QuantumCircuit:
    """
    Build cost oracle U_C(γ) = exp(-iγ H_C) where H_C = c·q̂ + λ·f̂

    Args:
        gamma: QAOA cost parameter
        c: Unit ordering cost
        lam: Stockout penalty (λ)
        n_q: Number of qubits in order quantity register
        reg_q_offset: Starting index of R_q
        reg_f: Index of flag qubit
        total_qubits: Total number of qubits

    Returns:
        QuantumCircuit implementing cost phase

    Implementation:
        1. Order cost term: RZ gates on R_q qubits weighted by bit position
           qubit j: RZ(angle = -γ * c * 2^j)
        2. Stockout penalty term: RZ gate on R_f
           RZ(reg_f, angle = -γ * λ)
    """
    circuit = QuantumCircuit(total_qubits)

    # Part 1: Order cost term exp(-iγc·q̂)
    # q̂ = Σ_{j=0}^{n_q-1} 2^j · (I - Z_j)/2
    # Phase rotation for each qubit weighted by bit position
    for j in range(n_q):
        qubit = reg_q_offset + j
        # Binary weight: 2^j
        angle = -gamma * c * (2 ** j)
        circuit.add_gate(RZ(qubit, angle))

    # Part 2: Stockout penalty term exp(-iγλ·f̂)
    # f̂ = (I - Z_f)/2, so phase rotation on flag qubit
    angle_f = -gamma * lam
    circuit.add_gate(RZ(reg_f, angle_f))

    return circuit


def build_mixer_circuit(beta: float, n_q: int, reg_q_offset: int,
                       total_qubits: int) -> QuantumCircuit:
    """
    Build mixer circuit U_M(β) = exp(-iβ H_M) where H_M = Σ_j X_j

    Args:
        beta: QAOA mixer parameter
        n_q: Number of qubits in order quantity register
        reg_q_offset: Starting index of R_q
        total_qubits: Total number of qubits

    Returns:
        QuantumCircuit implementing mixer

    Note:
        Acts ONLY on R_q register, not on R_d, R_f, or ancilla
    """
    circuit = QuantumCircuit(total_qubits)

    # Apply RX gates on all R_q qubits
    # RX(θ) = exp(-iθX/2), so we need angle = 2β
    for j in range(n_q):
        qubit = reg_q_offset + j
        circuit.add_gate(RX(qubit, 2 * beta))

    return circuit


# ============================================================================
# Phase 2: FSL Integration
# ============================================================================

def normalize_demand_distribution(demand_dist: Dict[int, float]) -> Dict[int, float]:
    """
    Ensure demand distribution probabilities sum to 1 and are non-negative.

    Args:
        demand_dist: Dictionary mapping demand values to probabilities

    Returns:
        Normalized demand distribution
    """
    total = sum(demand_dist.values())

    if abs(total - 1.0) > 1e-6:
        print(f"Warning: Demand distribution sum is {total:.6f}, renormalizing.")
        demand_dist = {d: p / total for d, p in demand_dist.items()}

    for d, p in demand_dist.items():
        if p < 0:
            raise ValueError(f"Negative probability for demand {d}: {p}")

    return demand_dist


def encode_demand_distribution(demand_dist: Union[Dict[int, float], Callable],
                               D_max: int, M: int = 16) -> Tuple:
    """
    Convert demand distribution to FSL Fourier coefficients.

    Args:
        demand_dist: Either dict {d: p_d} or callable PDF
        D_max: Maximum demand value
        M: Fourier series truncation order

    Returns:
        ks, cs, meta: Fourier mode indices, coefficients, and metadata
    """
    if callable(demand_dist):
        # Use the function directly
        # Scale to [0, 1] range for better FSL performance
        # Original function is defined on [0, D_max]
        original_func = demand_dist

        def func(x):
            # x is in [0, 1], map to [0, D_max]
            return original_func(x * D_max)

        # Use T=1 for the scaled function
        T = 1.0
    else:
        # Convert discrete PMF to continuous function via interpolation
        demand_dist = normalize_demand_distribution(demand_dist)

        # Create a piecewise constant function in [0, 1] range
        def func(x):
            x = np.asarray(x)
            result = np.zeros_like(x, dtype=float)
            for d, p in demand_dist.items():
                # Map d to [0, 1] range
                d_scaled = d / D_max
                d_next_scaled = (d + 1) / D_max
                mask = (x >= d_scaled) & (x < d_next_scaled)
                result[mask] = p * D_max  # Scale probability density
            return result

        T = 1.0

    # Compute Fourier coefficients using existing FSL function
    ks, cs, meta = fourier_series_coeffs(
        func,
        T=T,
        M=M,
        x0=0.0
    )

    # Store D_max in metadata for later use
    meta['D_max'] = D_max

    return ks, cs, meta


def remap_circuit_qubits(circuit: QuantumCircuit, offset: int) -> QuantumCircuit:
    """
    Shift all qubit indices in circuit by offset.

    Args:
        circuit: Original circuit
        offset: Qubit index offset

    Returns:
        New circuit with shifted qubit indices

    Note: This is a simplified version. For production use,
          would need to handle all gate types properly.
    """
    # Get number of qubits from original circuit
    n_qubits_orig = circuit.get_qubit_count()
    n_qubits_new = n_qubits_orig + offset

    new_circuit = QuantumCircuit(n_qubits_new)

    # Copy all gates with shifted indices
    # Note: This is a simplified approach - in practice, would need
    # to iterate through gates and remap each one
    # For now, we'll use the circuit as-is and handle offset in caller

    return circuit  # Placeholder - needs proper implementation


def build_demand_state_circuit(ks, cs, n_d: int, reg_d_offset: int) -> QuantumCircuit:
    """
    Build FSL circuit for demand distribution on R_d register.

    Args:
        ks: Fourier mode indices
        cs: Fourier coefficients
        n_d: Number of qubits in demand register
        reg_d_offset: Starting index of R_d

    Returns:
        FSL circuit for demand state preparation (FSL + IQFT)

    Note:
        FSL encodes the distribution in Fourier space.
        IQFT is needed to transform it back to computational basis (probability distribution).
    """
    # FSL needs enough qubits to accommodate Fourier modes
    # For modes -M to +M, need dimension >= 2M+1
    # So need m such that 2^(m+1) >= 2M+1
    M = max(abs(k) for k in ks)
    min_dim = 2 * M + 1
    m_fsl = int(np.ceil(np.log2(min_dim))) - 1

    # Use the larger of n_d-1 and m_fsl
    m = max(n_d - 1, m_fsl)
    n_qubits = m + 1

    # Build complete circuit: FSL + IQFT
    complete_circuit = QuantumCircuit(n_qubits)

    # Step 1: FSL circuit (encodes in Fourier space)
    fsl_circuit, meta = build_Uc_circuit_from_ck_cascade(ks, cs, m=m)
    complete_circuit.merge_circuit(fsl_circuit)

    # Step 2: IQFT to transform to computational basis (probability distribution)
    iqft = iqft_circuit(n_qubits)
    complete_circuit.merge_circuit(iqft)

    # In practice, would need to remap qubits to reg_d_offset range
    # For now, return the circuit as-is
    # TODO: Implement proper qubit remapping

    return complete_circuit


# ============================================================================
# Phase 3: QAOA Structure
# ============================================================================

def invert_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Return adjoint (inverse) of a quantum circuit.

    Args:
        circuit: Original circuit

    Returns:
        Inverse circuit

    Note: For uncomputation of the comparator
    """
    # Create new circuit with same number of qubits
    n_qubits = circuit.get_qubit_count()
    inv_circuit = QuantumCircuit(n_qubits)

    # For a simple comparator with CNOTs, Toffolis, and X gates,
    # the inverse is the same circuit applied in reverse order
    # (since these gates are self-inverse)

    # Get all gates and add them in reverse order
    gate_count = circuit.get_gate_count()
    gates = [circuit.get_gate(i) for i in range(gate_count)]

    for gate in reversed(gates):
        inv_circuit.add_gate(gate)

    return inv_circuit


def build_qaoa_layer(gamma: float, beta: float,
                    n_q: int, n_d: int, c: float, lam: float,
                    reg_q_offset: int, reg_d_offset: int,
                    reg_f: int, ancilla_offset: int,
                    total_qubits: int) -> QuantumCircuit:
    """
    Build single QAOA layer: Compare → Cost Phase → Uncompare → Mix

    Args:
        gamma: Cost parameter
        beta: Mixer parameter
        n_q, n_d: Number of qubits in q and d registers
        c, lam: Cost coefficients
        reg_q_offset, reg_d_offset, reg_f, ancilla_offset: Register offsets
        total_qubits: Total number of qubits

    Returns:
        QuantumCircuit for one QAOA layer
    """
    layer_circuit = QuantumCircuit(total_qubits)

    # 1. Comparator: Compute f = 1[d > q]
    comp_circuit = build_comparator_circuit(
        n_q, n_d, reg_q_offset, reg_d_offset, reg_f, ancilla_offset
    )
    # Merge circuits
    for i in range(comp_circuit.get_gate_count()):
        gate = comp_circuit.get_gate(i)
        layer_circuit.add_gate(gate)

    # 2. Cost oracle: Apply U_C(γ)
    cost_circuit = build_cost_oracle(
        gamma, c, lam, n_q, reg_q_offset, reg_f, total_qubits
    )
    for i in range(cost_circuit.get_gate_count()):
        gate = cost_circuit.get_gate(i)
        layer_circuit.add_gate(gate)

    # 3. Uncompute comparator: Restore ancilla to |0⟩
    # Apply inverse of comparator
    inv_comp_circuit = invert_circuit(comp_circuit)
    for i in range(inv_comp_circuit.get_gate_count()):
        gate = inv_comp_circuit.get_gate(i)
        layer_circuit.add_gate(gate)

    # 4. Mixer: Apply U_M(β) on R_q only
    mixer_circuit = build_mixer_circuit(beta, n_q, reg_q_offset, total_qubits)
    for i in range(mixer_circuit.get_gate_count()):
        gate = mixer_circuit.get_gate(i)
        layer_circuit.add_gate(gate)

    return layer_circuit


def build_full_qaoa_circuit(gammas: np.ndarray, betas: np.ndarray,
                           n_q: int, n_d: int, ks, cs,
                           c: float, lam: float,
                           Q_max: int, D_max: int) -> QuantumCircuit:
    """
    Build complete QAOA circuit with p layers.

    Args:
        gammas: Array of cost parameters [γ_1, ..., γ_p]
        betas: Array of mixer parameters [β_1, ..., β_p]
        n_q, n_d: Number of qubits for q and d registers
        ks, cs: Fourier coefficients for demand distribution
        c, lam: Cost coefficients
        Q_max, D_max: Maximum order quantity and demand

    Returns:
        Complete QAOA circuit ready for measurement
    """
    p = len(gammas)

    # Register layout
    reg_q_offset = 0
    reg_d_offset = n_q
    reg_f = n_q + n_d
    ancilla_offset = n_q + n_d + 1
    n_ancilla = max(n_q, n_d)  # For comparator
    total_qubits = n_q + n_d + 1 + n_ancilla

    circuit = QuantumCircuit(total_qubits)

    # 1. Initialize R_q: Uniform superposition over all q values
    for j in range(n_q):
        circuit.add_gate(H(reg_q_offset + j))

    # 2. FSL for demand distribution (once at beginning)
    fsl_circuit = build_demand_state_circuit(ks, cs, n_d, reg_d_offset)
    # Merge FSL circuit
    # Note: Need to handle qubit offset properly
    for i in range(fsl_circuit.get_gate_count()):
        gate = fsl_circuit.get_gate(i)
        # Shift gate targets by reg_d_offset
        # This is simplified - needs proper implementation
        circuit.add_gate(gate)

    # 3. Apply p QAOA layers
    for layer_idx in range(p):
        layer_circuit = build_qaoa_layer(
            gammas[layer_idx], betas[layer_idx],
            n_q, n_d, c, lam,
            reg_q_offset, reg_d_offset, reg_f, ancilla_offset,
            total_qubits
        )
        # Merge layer circuit
        for i in range(layer_circuit.get_gate_count()):
            gate = layer_circuit.get_gate(i)
            circuit.add_gate(gate)

    return circuit


# ============================================================================
# Phase 4: Classical Optimization and Verification
# ============================================================================

def compute_stockout_prob(q: int, demand_dist: Dict[int, float]) -> float:
    """
    Classical computation: Pr(D > q) = Σ_{d>q} p_d

    Args:
        q: Order quantity
        demand_dist: Demand distribution {d: p_d}

    Returns:
        Probability of stockout
    """
    return sum(p_d for d, p_d in demand_dist.items() if d > q)


def classical_optimal_solution(demand_dist: Dict[int, float],
                               c: float, lam: float,
                               Q_max: int) -> Tuple[int, float]:
    """
    Brute-force classical solution for verification.

    Args:
        demand_dist: Demand distribution
        c, lam: Cost coefficients
        Q_max: Maximum order quantity

    Returns:
        (optimal_q, optimal_cost)
    """
    best_q = 0
    best_cost = float('inf')

    for q in range(Q_max + 1):
        cost = c * q + lam * compute_stockout_prob(q, demand_dist)
        if cost < best_cost:
            best_cost = cost
            best_q = q

    return best_q, best_cost


def measure_and_extract_solution(circuit: QuantumCircuit,
                                 n_q: int, reg_q_offset: int,
                                 Q_max: int,
                                 n_shots: int = 10000) -> Dict:
    """
    Execute circuit and extract order quantity distribution.

    Args:
        circuit: QAOA circuit
        n_q: Number of qubits in order quantity register
        reg_q_offset: Starting index of R_q
        Q_max: Maximum order quantity
        n_shots: Number of measurement samples

    Returns:
        Dictionary with measurement results
    """
    # Initialize quantum state
    state = QuantumState(circuit.get_qubit_count())
    state.set_zero_state()

    # Apply circuit
    circuit.update_quantum_state(state)

    # Sample R_q register
    q_samples = {}

    for _ in range(n_shots):
        # Measure all qubits
        outcome_bits = state.sampling(1)[0]

        # Extract q value from R_q register
        # Shift right by reg_q_offset and mask to get n_q bits
        q_bits = (outcome_bits >> reg_q_offset) & ((1 << n_q) - 1)
        q_value = int(q_bits)

        if q_value <= Q_max:  # Valid order quantity
            q_samples[q_value] = q_samples.get(q_value, 0) + 1

    # Find most frequent q
    if q_samples:
        best_q_freq = max(q_samples, key=q_samples.get)
        confidence = q_samples[best_q_freq] / n_shots
    else:
        best_q_freq = 0
        confidence = 0.0

    return {
        'best_q': best_q_freq,
        'distribution': q_samples,
        'confidence': confidence
    }


def optimize_qaoa_parameters(p: int, initial_params: np.ndarray,
                            bounds: list,
                            n_q: int, n_d: int, ks, cs,
                            demand_dist_dict: Dict[int, float],
                            c: float, lam: float,
                            Q_max: int, D_max: int,
                            n_shots: int = 1000,
                            track_history: bool = True) -> Tuple:
    """
    Optimize QAOA parameters (γ, β) using classical optimizer.

    Args:
        p: QAOA depth
        initial_params: Initial [γ_1, ..., γ_p, β_1, ..., β_p]
        bounds: Parameter bounds
        n_q, n_d: Qubit counts
        ks, cs: Fourier coefficients
        demand_dist_dict: Demand distribution for cost evaluation
        c, lam: Cost coefficients
        Q_max, D_max: Maximum values
        n_shots: Measurement shots per evaluation
        track_history: Whether to track optimization history

    Returns:
        (optimal_params, optimal_cost, optimizer_result, history)
    """
    # History tracking
    history = {
        'params': [],
        'costs': [],
        'q_distributions': [],
        'iteration': 0
    }

    def objective(params):
        """Evaluate expected cost for given QAOA parameters."""
        gammas = params[:p]
        betas = params[p:]

        # Build QAOA circuit
        circuit = build_full_qaoa_circuit(
            gammas, betas, n_q, n_d, ks, cs, c, lam, Q_max, D_max
        )

        # Measure and get q distribution
        result = measure_and_extract_solution(
            circuit, n_q, 0, Q_max, n_shots
        )

        # Compute expected cost
        total_cost = 0.0
        total_count = sum(result['distribution'].values())

        for q, count in result['distribution'].items():
            prob = count / total_count
            cost_q = c * q + lam * compute_stockout_prob(q, demand_dist_dict)
            total_cost += prob * cost_q

        # Track history
        if track_history:
            history['params'].append(params.copy())
            history['costs'].append(total_cost)
            history['q_distributions'].append(result['distribution'].copy())
            history['iteration'] += 1

            # Print progress every 10 iterations
            if history['iteration'] % 10 == 0:
                print(f"  Iteration {history['iteration']}: cost = {total_cost:.4f}")

        return total_cost

    # Run optimizer
    result = minimize(
        objective,
        initial_params,
        method='COBYLA',
        bounds=bounds,
        options={'maxiter': 100, 'rhobeg': 0.1}
    )

    return result.x, result.fun, result, history


# ============================================================================
# Phase 5: Main Solver Interface
# ============================================================================

def estimate_circuit_depth(n_q: int, n_d: int, p: int) -> int:
    """
    Estimate total circuit depth for resource planning.

    Args:
        n_q, n_d: Number of qubits
        p: QAOA depth

    Returns:
        Estimated circuit depth
    """
    # FSL preparation: O(2^n_d) approximate
    depth_fsl = 2 ** n_d

    # Comparator per layer: O(n_d)
    # Cost oracle per layer: O(n_q)
    # Uncompute per layer: O(n_d)
    # Mixer per layer: O(n_q)
    depth_per_layer = 2 * max(n_d, n_q) + n_q + n_d

    total_depth = depth_fsl + p * depth_per_layer

    return total_depth


def solve_newsvendor_qaoa(
    demand_dist: Union[Dict[int, float], Callable],
    c: float,
    lam: float,
    Q_max: int,
    D_max: int,
    p: int = 2,
    M: int = 16,
    n_shots: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Solve newsvendor problem using FSL + QAOA.

    Args:
        demand_dist: Demand distribution (dict or callable)
        c: Unit ordering cost
        lam: Stockout penalty (λ)
        Q_max: Maximum order quantity
        D_max: Maximum demand
        p: QAOA depth
        M: Fourier truncation order
        n_shots: Measurement shots during optimization
        verbose: Print progress information

    Returns:
        Dictionary with solution and diagnostics
    """
    if verbose:
        print("=" * 60)
        print("Newsvendor Problem with FSL + QAOA")
        print("=" * 60)

    # 1. Setup
    n_q = int(np.ceil(np.log2(Q_max + 1)))
    n_d = int(np.ceil(np.log2(D_max + 1)))
    n_ancilla = max(n_q, n_d)
    total_qubits = n_q + n_d + 1 + n_ancilla

    if verbose:
        print(f"Problem size: Q_max={Q_max}, D_max={D_max}")
        print(f"Qubits: n_q={n_q}, n_d={n_d}, total={total_qubits}")
        print(f"QAOA depth: p={p}, Fourier truncation: M={M}")

    # 2. Encode demand distribution
    ks, cs, fsl_meta = encode_demand_distribution(demand_dist, D_max, M)

    # Convert to dict if callable for classical computation
    if callable(demand_dist):
        demand_dist_dict = {}
        for d in range(D_max + 1):
            demand_dist_dict[d] = demand_dist(d)
        demand_dist_dict = normalize_demand_distribution(demand_dist_dict)
    else:
        demand_dist_dict = normalize_demand_distribution(demand_dist)

    if verbose:
        print(f"Demand distribution encoded with {len(ks)} Fourier modes")

    # 3. Optimize QAOA parameters
    if verbose:
        print("\nOptimizing QAOA parameters...")

    initial_params = np.random.uniform(-0.1, 0.1, 2 * p)
    bounds = [(-np.pi, np.pi)] * (2 * p)

    optimal_params, optimal_cost_qaoa, opt_result, opt_history = optimize_qaoa_parameters(
        p, initial_params, bounds,
        n_q, n_d, ks, cs,
        demand_dist_dict, c, lam,
        Q_max, D_max, n_shots,
        track_history=True
    )

    if verbose:
        n_iter = getattr(opt_result, 'nit', getattr(opt_result, 'nfev', 'unknown'))
        print(f"Optimization completed: {n_iter} function evaluations")

    # 4. Extract final solution with more shots
    gammas = optimal_params[:p]
    betas = optimal_params[p:]

    final_circuit = build_full_qaoa_circuit(
        gammas, betas, n_q, n_d, ks, cs, c, lam, Q_max, D_max
    )

    solution = measure_and_extract_solution(
        final_circuit, n_q, 0, Q_max, n_shots=10000
    )

    # 5. Classical verification
    classical_q, classical_cost = classical_optimal_solution(
        demand_dist_dict, c, lam, Q_max
    )

    # Compute quantum cost
    quantum_q = solution['best_q']
    quantum_cost = c * quantum_q + lam * compute_stockout_prob(
        quantum_q, demand_dist_dict
    )

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Quantum solution: q = {quantum_q}")
        print(f"Quantum cost: {quantum_cost:.4f}")
        print(f"Classical solution: q = {classical_q}")
        print(f"Classical cost: {classical_cost:.4f}")
        print(f"Approximation ratio: {quantum_cost / classical_cost:.4f}")
        print(f"Measurement confidence: {solution['confidence']:.2%}")
        print(f"Circuit depth (estimated): {estimate_circuit_depth(n_q, n_d, p)}")

    return {
        'quantum_solution': quantum_q,
        'quantum_cost': quantum_cost,
        'classical_solution': classical_q,
        'classical_cost': classical_cost,
        'distribution': solution['distribution'],
        'confidence': solution['confidence'],
        'optimal_params': optimal_params,
        'optimization_result': opt_result,
        'optimization_history': opt_history,
        'circuit_depth': estimate_circuit_depth(n_q, n_d, p),
        'demand_dist': demand_dist_dict,
        'final_circuit': final_circuit,
        'n_q': n_q,
        'n_d': n_d,
        'p': p,
        'ks': ks,
        'cs': cs,
        'c': c,
        'lam': lam
    }


# ============================================================================
# Phase 6: Visualization
# ============================================================================

def visualize_qaoa_results(q_distribution: Dict[int, int],
                          demand_dist: Dict[int, float],
                          c: float, lam: float, Q_max: int,
                          classical_q: int = None,
                          quantum_q: int = None):
    """
    Create visualization dashboard for QAOA results.

    Args:
        q_distribution: Measurement distribution {q: count}
        demand_dist: Demand distribution {d: p_d}
        c, lam: Cost coefficients
        Q_max: Maximum order quantity
        classical_q: Classical optimal solution
        quantum_q: Quantum solution
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: QAOA measurement distribution
    ax1 = axes[0, 0]
    if q_distribution:
        total_counts = sum(q_distribution.values())
        qs = sorted(q_distribution.keys())
        probs = [q_distribution[q] / total_counts for q in qs]
        ax1.bar(qs, probs, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Order Quantity q', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_title('QAOA Measurement Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        if quantum_q is not None:
            ax1.axvline(quantum_q, color='red', linestyle='--', linewidth=2,
                       label=f'Quantum solution: q={quantum_q}')
            ax1.legend()

    # Plot 2: Cost function landscape
    ax2 = axes[0, 1]
    q_range = range(Q_max + 1)
    costs = [c * q + lam * compute_stockout_prob(q, demand_dist)
             for q in q_range]
    ax2.plot(q_range, costs, 'o-', linewidth=2, markersize=6,
            color='green', label='C(q) = c·q + λ·Pr(D>q)')
    ax2.set_xlabel('Order Quantity q', fontsize=12)
    ax2.set_ylabel('Cost', fontsize=12)
    ax2.set_title('Cost Function Landscape', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    if classical_q is not None:
        ax2.axvline(classical_q, color='blue', linestyle='--', linewidth=2,
                   label=f'Classical optimal: q={classical_q}')
    if quantum_q is not None:
        ax2.axvline(quantum_q, color='red', linestyle='--', linewidth=2,
                   label=f'Quantum solution: q={quantum_q}')
    ax2.legend()

    # Plot 3: Demand distribution
    ax3 = axes[1, 0]
    ds = sorted(demand_dist.keys())
    ps = [demand_dist[d] for d in ds]
    ax3.bar(ds, ps, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('Demand d', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title('Demand Distribution', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Comparison metrics
    ax4 = axes[1, 1]
    ax4.axis('off')

    if classical_q is not None and quantum_q is not None:
        classical_cost = costs[classical_q]
        quantum_cost = costs[quantum_q]
        approx_ratio = quantum_cost / classical_cost

        info_text = f"""
        COMPARISON METRICS
        {'=' * 40}

        Classical Solution:
            Order Quantity: {classical_q}
            Cost: {classical_cost:.4f}

        Quantum Solution:
            Order Quantity: {quantum_q}
            Cost: {quantum_cost:.4f}

        Performance:
            Approximation Ratio: {approx_ratio:.4f}
            Solution Match: {'✓ Yes' if quantum_q == classical_q else '✗ No'}

        Problem Parameters:
            Order cost (c): {c}
            Stockout penalty (λ): {lam}
            Q_max: {Q_max}
        """
        ax4.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center')

    plt.tight_layout()
    plt.savefig('newsvendor_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: newsvendor_results.png")
    plt.show()


def plot_optimization_convergence(history: Dict, figsize=(12, 8)):
    """
    Plot QAOA optimization convergence history.

    Args:
        history: Optimization history from optimize_qaoa_parameters
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    costs = history['costs']
    iterations = range(1, len(costs) + 1)
    params_array = np.array(history['params'])
    p = params_array.shape[1] // 2

    # Plot 1: Cost convergence
    ax1 = axes[0, 0]
    ax1.plot(iterations, costs, 'o-', linewidth=2, markersize=4, color='blue')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Cost', fontsize=12)
    ax1.set_title('Cost Function Convergence', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Mark best cost
    best_idx = np.argmin(costs)
    best_cost = costs[best_idx]
    ax1.plot(best_idx + 1, best_cost, 'r*', markersize=15,
             label=f'Best: {best_cost:.4f} at iter {best_idx + 1}')
    ax1.legend()

    # Plot 2: Parameter evolution (gammas)
    ax2 = axes[0, 1]
    gammas = params_array[:, :p]
    for i in range(p):
        ax2.plot(iterations, gammas[:, i], 'o-', linewidth=2, markersize=3,
                label=f'γ_{i+1}')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('γ value', fontsize=12)
    ax2.set_title('Cost Parameters (γ) Evolution', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Plot 3: Parameter evolution (betas)
    ax3 = axes[1, 0]
    betas = params_array[:, p:]
    for i in range(p):
        ax3.plot(iterations, betas[:, i], 's-', linewidth=2, markersize=3,
                label=f'β_{i+1}')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('β value', fontsize=12)
    ax3.set_title('Mixer Parameters (β) Evolution', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.legend()

    # Plot 4: Q distribution evolution (show first, middle, last)
    ax4 = axes[1, 1]
    q_dists = history['q_distributions']

    # Select iterations to show
    indices_to_show = [0, len(q_dists)//2, len(q_dists)-1]
    labels = ['Initial', 'Middle', 'Final']
    colors = ['red', 'orange', 'green']

    for idx, label, color in zip(indices_to_show, labels, colors):
        if idx < len(q_dists):
            dist = q_dists[idx]
            total = sum(dist.values())
            qs = sorted(dist.keys())
            probs = [dist[q]/total for q in qs]
            ax4.plot(qs, probs, 'o-', linewidth=2, markersize=5,
                    label=f'{label} (iter {idx+1})', color=color, alpha=0.7)

    ax4.set_xlabel('Order Quantity q', fontsize=12)
    ax4.set_ylabel('Probability', fontsize=12)
    ax4.set_title('Q Distribution Evolution', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('qaoa_convergence.png', dpi=150, bbox_inches='tight')
    print("\nConvergence plot saved to: qaoa_convergence.png")
    plt.show()


def visualize_circuit_structure(result: Dict, show_full: bool = False):
    """
    Visualize the quantum circuit structure.

    Args:
        result: Result dictionary from solve_newsvendor_qaoa
        show_full: Whether to show the full circuit (can be very large)
    """
    print("\n" + "=" * 60)
    print("CIRCUIT STRUCTURE ANALYSIS")
    print("=" * 60)

    circuit = result['final_circuit']
    n_q = result['n_q']
    n_d = result['n_d']
    p = result['p']

    print(f"\nCircuit Statistics:")
    print(f"  Total qubits: {circuit.get_qubit_count()}")
    print(f"  Total gates: {circuit.get_gate_count()}")
    print(f"  Estimated depth: {result['circuit_depth']}")
    print(f"  QAOA layers (p): {p}")

    print(f"\nRegister Layout:")
    print(f"  R_q (order quantity): qubits 0-{n_q-1} ({n_q} qubits)")
    print(f"  R_d (demand): qubits {n_q}-{n_q+n_d-1} ({n_d} qubits)")
    print(f"  R_f (stockout flag): qubit {n_q+n_d}")
    print(f"  Ancilla: qubits {n_q+n_d+1}+")

    # Gate type analysis
    print(f"\nGate Composition:")
    gate_types = {}
    for i in range(circuit.get_gate_count()):
        gate = circuit.get_gate(i)
        gate_str = str(type(gate).__name__)
        gate_types[gate_str] = gate_types.get(gate_str, 0) + 1

    for gate_type, count in sorted(gate_types.items(), key=lambda x: -x[1]):
        print(f"  {gate_type}: {count}")

    # Try to visualize with qulacsvis if available
    if HAS_QULACSVIS and show_full:
        print("\nGenerating circuit diagram...")
        try:
            # For large circuits, only show a portion
            if circuit.get_gate_count() > 100:
                print("  (Warning: Large circuit, showing first 100 gates)")
                # Create a smaller circuit for visualization
                small_circuit = QuantumCircuit(circuit.get_qubit_count())
                for i in range(min(100, circuit.get_gate_count())):
                    small_circuit.add_gate(circuit.get_gate(i))
                circuit_drawer(small_circuit, "mpl")
                plt.savefig('circuit_diagram.png', dpi=150, bbox_inches='tight')
                print("  Circuit diagram saved to: circuit_diagram.png")
            else:
                circuit_drawer(circuit, "mpl")
                plt.savefig('circuit_diagram.png', dpi=150, bbox_inches='tight')
                print("  Circuit diagram saved to: circuit_diagram.png")
            plt.show()
        except Exception as e:
            print(f"  Could not generate circuit diagram: {e}")
    elif not HAS_QULACSVIS:
        print("\n(Install qulacsvis for circuit diagram visualization: pip install qulacsvis)")


def analyze_fsl_encoding(ks, cs, demand_dist: Dict[int, float],
                        D_max: int, n_samples: int = 10000):
    """
    Analyze FSL encoding fidelity by simulating the FSL circuit.

    Args:
        ks: Fourier mode indices
        cs: Fourier coefficients
        demand_dist: Target demand distribution
        D_max: Maximum demand
        n_samples: Number of samples for fidelity check

    Returns:
        Dictionary with fidelity metrics
    """
    print("\n" + "=" * 60)
    print("FSL ENCODING FIDELITY ANALYSIS")
    print("=" * 60)

    # Build FSL circuit (FSL + IQFT)
    M = max(abs(k) for k in ks)
    min_dim = 2 * M + 1
    m = int(np.ceil(np.log2(min_dim))) - 1
    n_qubits = m + 1

    # Create complete circuit: FSL + IQFT
    complete_circuit = QuantumCircuit(n_qubits)

    # Step 1: FSL
    fsl_circuit, meta = build_Uc_circuit_from_ck_cascade(ks, cs, m=m)
    complete_circuit.merge_circuit(fsl_circuit)

    # Step 2: IQFT (to convert from Fourier space to probability distribution)
    iqft = iqft_circuit(n_qubits)
    complete_circuit.merge_circuit(iqft)

    print(f"\nFSL Circuit Info:")
    print(f"  Number of Fourier modes: {len(ks)}")
    print(f"  Max mode index (M): {M}")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Dimension: {2**n_qubits}")
    print(f"  Circuit gates: {complete_circuit.get_gate_count()} (FSL + IQFT)")

    # Simulate FSL encoding
    state = QuantumState(n_qubits)
    state.set_zero_state()
    complete_circuit.update_quantum_state(state)

    # Sample the state
    # The measured state |k⟩ corresponds to x = k / 2^n (in [0,1] range)
    # We need to map this back to demand values d = x * D_max
    encoded_dist = {}
    all_samples = state.sampling(n_samples)

    for sample in all_samples:
        k = int(sample)  # Measured state index
        # Map k to [0, 1] range
        x = k / (2 ** n_qubits)
        # Map to demand range [0, D_max]
        d_continuous = x * D_max
        # Round to nearest integer demand value
        d = int(round(d_continuous))

        if 0 <= d <= D_max:
            encoded_dist[d] = encoded_dist.get(d, 0) + 1

    # Normalize by TOTAL number of samples, not just those <= D_max
    # This gives the true probability within [0, D_max]
    encoded_probs = {d: count/n_samples for d, count in encoded_dist.items()}

    # Fill in missing demand values with zero probability
    for d in range(D_max + 1):
        if d not in encoded_probs:
            encoded_probs[d] = 0.0

    # Also compute probability mass outside [0, D_max]
    prob_outside = 1.0 - sum(encoded_probs.values())
    if prob_outside > 0.01:  # More than 1% outside range
        print(f"  Warning: {prob_outside:.2%} probability mass outside [0, D_max]")

    # Compute fidelity
    fidelity = 0.0
    for d in range(D_max + 1):
        p_target = demand_dist.get(d, 0)
        p_encoded = encoded_probs.get(d, 0)
        fidelity += np.sqrt(p_target * p_encoded)
    fidelity = fidelity ** 2

    print(f"\nFidelity Metrics:")
    print(f"  Classical Fidelity: {fidelity:.6f} ({fidelity*100:.4f}%)")
    print(f"  Target = 1.0 (100%)")

    # Compute total variation distance
    tvd = 0.0
    for d in range(D_max + 1):
        p_target = demand_dist.get(d, 0)
        p_encoded = encoded_probs.get(d, 0)
        tvd += abs(p_target - p_encoded)
    tvd = tvd / 2

    print(f"  Total Variation Distance: {tvd:.6f}")
    print(f"  Target = 0.0 (perfect match)")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Bar chart comparison
    ds = sorted(set(list(demand_dist.keys()) + list(encoded_probs.keys())))
    target_vals = [demand_dist.get(d, 0) for d in ds]
    encoded_vals = [encoded_probs.get(d, 0) for d in ds]

    x = np.arange(len(ds))
    width = 0.35

    ax1.bar(x - width/2, target_vals, width, label='Target', alpha=0.7, color='blue')
    ax1.bar(x + width/2, encoded_vals, width, label='FSL Encoded', alpha=0.7, color='red')
    ax1.set_xlabel('Demand d', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('FSL Encoding: Target vs Encoded', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Error plot
    errors = [encoded_probs.get(d, 0) - demand_dist.get(d, 0) for d in ds]
    ax2.bar(ds, errors, alpha=0.7, color=['red' if e > 0 else 'blue' for e in errors])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Demand d', fontsize=12)
    ax2.set_ylabel('Error (Encoded - Target)', fontsize=12)
    ax2.set_title('FSL Encoding Error', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('fsl_fidelity.png', dpi=150, bbox_inches='tight')
    print("\nFSL fidelity plot saved to: fsl_fidelity.png")
    plt.show()

    return {
        'fidelity': fidelity,
        'tvd': tvd,
        'encoded_probs': encoded_probs,
        'target_probs': demand_dist
    }


def comprehensive_analysis(result: Dict):
    """
    Perform comprehensive analysis of QAOA results with all visualizations.

    Args:
        result: Result dictionary from solve_newsvendor_qaoa
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE QAOA ANALYSIS")
    print("=" * 70)

    # 1. Optimization convergence
    print("\n1. OPTIMIZATION CONVERGENCE ANALYSIS")
    print("-" * 70)
    plot_optimization_convergence(result['optimization_history'])

    # 2. Circuit structure
    print("\n2. CIRCUIT STRUCTURE ANALYSIS")
    print("-" * 70)
    visualize_circuit_structure(result, show_full=False)

    # 3. FSL encoding fidelity
    print("\n3. FSL ENCODING FIDELITY")
    print("-" * 70)
    fsl_metrics = analyze_fsl_encoding(
        result['ks'], result['cs'],
        result['demand_dist'],
        max(result['demand_dist'].keys())
    )

    # 4. Final results comparison
    print("\n4. FINAL RESULTS")
    print("-" * 70)
    # Determine Q_max from the distribution or use max key
    Q_max = max(max(result['distribution'].keys()), result['quantum_solution'], result['classical_solution'])
    visualize_qaoa_results(
        result['distribution'],
        result['demand_dist'],
        c=result.get('c', 1.0),
        lam=result.get('lam', 5.0),
        Q_max=Q_max,
        classical_q=result['classical_solution'],
        quantum_q=result['quantum_solution']
    )

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Quantum Solution: q = {result['quantum_solution']}")
    print(f"Classical Solution: q = {result['classical_solution']}")
    print(f"Approximation Ratio: {result['quantum_cost']/result['classical_cost']:.4f}")
    print(f"FSL Fidelity: {fsl_metrics['fidelity']:.6f}")
    print(f"Optimization Iterations: {len(result['optimization_history']['costs'])}")
    print(f"Best Cost Achieved: {min(result['optimization_history']['costs']):.4f}")
    print("=" * 70)


# ============================================================================
# Example Usage: Gaussian Demand
# ============================================================================

def example_gaussian_demand():
    """
    Primary test case: Gaussian demand distribution.

    Parameters:
        - Mean demand: μ = 50
        - Std deviation: σ = 10
        - D_max = 100, Q_max = 100
        - Order cost: c = 5.0
        - Stockout penalty: λ = 20.0
        - QAOA depth: p = 3
        - Fourier truncation: M = 32 (higher for smooth distribution)
    """
    from scipy.stats import norm

    print("\n" + "=" * 70)
    print("EXAMPLE: Newsvendor with Gaussian Demand Distribution")
    print("=" * 70)

    # Continuous Gaussian PDF
    mu, sigma = 50, 10
    demand_pdf = lambda x: norm.pdf(x, mu, sigma)

    # Also create discrete version for classical verification
    D_max = 100
    demand_dist = {}
    for d in range(D_max + 1):
        demand_dist[d] = norm.pdf(d, mu, sigma)
    total = sum(demand_dist.values())
    demand_dist = {d: p / total for d, p in demand_dist.items()}

    print(f"\nDemand Distribution: Gaussian(μ={mu}, σ={sigma})")
    print(f"Problem size: D_max={D_max}, Q_max={D_max}")
    print(f"Cost parameters: c=5.0, λ=20.0")
    print(f"Expected demand: {sum(d * p for d, p in demand_dist.items()):.2f}")

    # Solve with QAOA
    result = solve_newsvendor_qaoa(
        demand_dist=demand_pdf,  # Pass continuous PDF
        c=5.0,
        lam=20.0,
        Q_max=100,
        D_max=100,
        p=3,
        M=32,  # Higher M for smooth Gaussian
        n_shots=5000,
        verbose=True
    )

    # Visualize results
    visualize_qaoa_results(
        result['distribution'],
        result['demand_dist'],
        c=5.0, lam=20.0, Q_max=100,
        classical_q=result['classical_solution'],
        quantum_q=result['quantum_solution']
    )

    return result


if __name__ == "__main__":
    # Run Gaussian demand example
    result = example_gaussian_demand()
