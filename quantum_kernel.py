"""
Fidelity quantum kernel with an IQP / ZZ-type feature map, simulated exactly
with a vectorised NumPy statevector simulator.

Feature map (one layer), for an n-qubit register and input x in R^n:
    U(x) = U_Z(x) H^{(x)n}
    U_Z(x) = exp( i * sum_j x_j Z_j  +  i * sum_{(j,k) in E} (pi - x_j)(pi - x_k) Z_j Z_k )
The layer is repeated L times. E is the set of entangling pairs
('full' = all pairs, 'linear' = nearest neighbours, 'ring' = linear + wrap-around).

Kernel:  k(x, x') = |<psi(x) | psi(x')>|^2,   psi(x) = U(gamma*x) |0...0>
gamma is a bandwidth (scaling) hyper-parameter which controls kernel concentration.
"""
from __future__ import annotations

import itertools
import numpy as np


def _pairs(n: int, entanglement: str):
    if entanglement == "full":
        return list(itertools.combinations(range(n), 2))
    if entanglement == "linear":
        return [(j, j + 1) for j in range(n - 1)]
    if entanglement == "ring":
        p = [(j, j + 1) for j in range(n - 1)]
        if n > 2:
            p.append((n - 1, 0))
        return p
    raise ValueError(entanglement)


def _spin_table(n: int) -> np.ndarray:
    """s[z, j] = +1 if bit j of basis state z is 0 else -1  (eigenvalue of Z_j).
    Qubit 0 is the most-significant bit (PennyLane wire ordering)."""
    z = np.arange(2 ** n)[:, None]
    bits = (z >> (n - 1 - np.arange(n))[None, :]) & 1
    return 1.0 - 2.0 * bits


def _hadamard_all(states: np.ndarray, n: int) -> np.ndarray:
    """Apply H on every qubit to a batch of statevectors (B, 2^n) - fast Walsh-Hadamard."""
    B = states.shape[0]
    psi = states.reshape((B,) + (2,) * n)
    for q in range(n):
        a = np.take(psi, 0, axis=q + 1)
        b = np.take(psi, 1, axis=q + 1)
        psi = np.stack(((a + b), (a - b)), axis=q + 1) / np.sqrt(2.0)
    return psi.reshape(B, 2 ** n)


class FidelityQuantumKernel:
    def __init__(self, n_qubits: int, layers: int = 2, gamma: float = 1.0,
                 entanglement: str = "full"):
        self.n = n_qubits
        self.layers = layers
        self.gamma = gamma
        self.entanglement = entanglement
        self.S = _spin_table(n_qubits)                 # (2^n, n)
        self.P = _pairs(n_qubits, entanglement)

    def statevectors(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float) * self.gamma
        assert X.shape[1] == self.n, f"expected {self.n} features, got {X.shape[1]}"
        B, N = X.shape[0], 2 ** self.n
        # diagonal phase  phi(x, z)  (B, 2^n)
        phase = X @ self.S.T
        for (j, k) in self.P:
            phase += np.outer((np.pi - X[:, j]) * (np.pi - X[:, k]),
                              self.S[:, j] * self.S[:, k])
        diag = np.exp(1j * phase)
        psi = np.zeros((B, N), dtype=complex)
        psi[:, 0] = 1.0
        for _ in range(self.layers):
            psi = _hadamard_all(psi, self.n)
            psi = psi * diag
        return psi

    def __call__(self, XA: np.ndarray, XB: np.ndarray | None = None) -> np.ndarray:
        A = self.statevectors(XA)
        Bv = A if XB is None else self.statevectors(XB)
        return np.abs(A.conj() @ Bv.T) ** 2


def verify_against_pennylane(n_qubits=4, layers=2, gamma=0.7, n_samples=4, seed=0):
    """Cross-check the NumPy simulator with an explicit PennyLane circuit.
    Returns the max absolute difference of the kernel entries."""
    import pennylane as qml

    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.pi, size=(n_samples, n_qubits))
    qk = FidelityQuantumKernel(n_qubits, layers, gamma, "full")
    K_np = qk(X)

    dev = qml.device("default.qubit", wires=n_qubits)
    pairs = _pairs(n_qubits, "full")

    def feature_map(x):
        x = gamma * x
        for _ in range(layers):
            for w in range(n_qubits):
                qml.Hadamard(wires=w)
            for w in range(n_qubits):          # exp(i x Z) = RZ(-2x)
                qml.RZ(-2 * x[w], wires=w)
            for (j, k) in pairs:               # exp(i t ZZ) = IsingZZ(-2t)
                qml.IsingZZ(-2 * (np.pi - x[j]) * (np.pi - x[k]), wires=[j, k])

    @qml.qnode(dev)
    def state(x):
        feature_map(x)
        return qml.state()

    S = np.array([state(x) for x in X])
    K_pl = np.abs(S.conj() @ S.T) ** 2
    return float(np.max(np.abs(K_np - K_pl)))


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Centred kernel-target alignment (Cortes et al., 2012) for multi-class labels."""
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(float)
    Ky = Y @ Y.T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Kyc = H @ K @ H, H @ Ky @ H
    return float(np.sum(Kc * Kyc) / (np.linalg.norm(Kc) * np.linalg.norm(Kyc) + 1e-12))


def effective_dimension(K: np.ndarray) -> float:
    """Participation ratio of the kernel spectrum: (sum l)^2 / sum l^2."""
    lam = np.clip(np.linalg.eigvalsh(K), 0, None)
    return float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-12))
