"""Core HMM: stationary distribution, sampling, beliefs, k-suffix beliefs."""
import numpy as np
try:
    import numba; HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

def stationary_distribution(T_matrices):
    T_full = sum(T_matrices)
    eigvals, eigvecs = np.linalg.eig(T_full.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    return pi / pi.sum()

def sample_hmm_sequence(T_matrices, pi, seq_len, seed=None):
    rng = np.random.default_rng(seed)
    n_states, n_tokens = len(pi), len(T_matrices)
    state = rng.choice(n_states, p=pi); tokens = []
    for _ in range(seq_len):
        tp = np.array([T_matrices[z][state].sum() for z in range(n_tokens)]); tp /= tp.sum()
        z = rng.choice(n_tokens, p=tp); tokens.append(z)
        nsp = T_matrices[z][state] / T_matrices[z][state].sum()
        state = rng.choice(n_states, p=nsp)
    return np.array(tokens)

def emission_matrix(T_matrices):
    """M_{ik} = sum_j T^{(o_k)}_{ij}. Maps beliefs to NTP vectors."""
    return np.stack([T.sum(axis=1) for T in T_matrices], axis=1)

def next_token_probs(beliefs, T_matrices):
    """Compute NTP from beliefs: p_k = (beliefs @ T^(k)).sum(axis=1)."""
    return beliefs @ emission_matrix(T_matrices)

# ── Belief computation ──
def _beliefs_numpy(tokens, T_stack, pi):
    n, n_states = len(tokens), len(pi)
    beliefs = np.zeros((n, n_states)); b = pi.copy()
    for t in range(n):
        b = b @ T_stack[tokens[t]]; s = b.sum()
        if s > 0: b /= s
        beliefs[t] = b
    return beliefs

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _beliefs_numba(tokens, T_stack, pi):
        n, n_states = len(tokens), len(pi)
        beliefs = np.zeros((n, n_states)); b = pi.copy()
        for t in range(n):
            b = b @ T_stack[tokens[t]]
            s = 0.0
            for j in range(n_states): s += b[j]
            if s > 0:
                for j in range(n_states): b[j] /= s
            for j in range(n_states): beliefs[t, j] = b[j]
        return beliefs

    @numba.njit(cache=True)
    def _reduced_beliefs_table(tok_at_pos, start_pos, n, k, table, n_tok):
        n_states = table.shape[1]; reduced = np.zeros((n, n_states))
        for i in range(n):
            t = start_pos + i; s = max(0, t - k + 1)
            idx = 0
            for j in range(s, t + 1): idx = idx * n_tok + tok_at_pos[j]
            reduced[i] = table[idx]
        return reduced

    @numba.njit(cache=True)
    def _reduced_beliefs_direct(tok_at_pos, start_pos, n, k, T_stack, pi):
        n_states = len(pi); reduced = np.zeros((n, n_states))
        for i in range(n):
            t = start_pos + i; s = max(0, t - k + 1)
            b = pi.copy()
            for j in range(s, t + 1):
                b = b @ T_stack[tok_at_pos[j]]
                sm = 0.0
                for q in range(n_states): sm += b[q]
                for q in range(n_states): b[q] /= sm
            for q in range(n_states): reduced[i, q] = b[q]
        return reduced

def full_bayesian_beliefs(tokens, T_stack, pi):
    """beliefs[t] = posterior after observing tokens[0:t+1]."""
    if HAS_NUMBA: return _beliefs_numba(tokens, T_stack, pi)
    return _beliefs_numpy(tokens, T_stack, pi)

def precompute_belief_tables(K_values, T_matrices, pi):
    """Precompute lookup tables for k-suffix beliefs."""
    n_tok = len(T_matrices); T = np.stack(T_matrices); n_states = len(pi)
    max_k = max(K_values); prev = pi.reshape(1, n_states); tables = {}
    for k in range(1, max_k + 1):
        cur = np.einsum('ps,zsd->pzd', prev, T).reshape(-1, n_states)
        cur /= cur.sum(axis=1, keepdims=True)
        if k in K_values: tables[k] = cur
        prev = cur
    return tables

def compute_k_beliefs(tok, probe_start, n, K_values, tables, T_stack, pi, n_tok, max_k_lookup):
    """Compute k-suffix beliefs for multiple k values."""
    beliefs_by_k = {}
    for k in K_values:
        if HAS_NUMBA:
            if k <= max_k_lookup:
                beliefs_by_k[k] = _reduced_beliefs_table(tok, probe_start, n, k, tables[k], n_tok)
            else:
                beliefs_by_k[k] = _reduced_beliefs_direct(tok, probe_start, n, k, T_stack, pi)
        else:
            # Pure numpy fallback for k-suffix
            n_states = len(pi); reduced = np.zeros((n, n_states))
            for i in range(n):
                t = probe_start + i; s = max(0, t - k + 1)
                b = pi.copy()
                for j in range(s, t + 1):
                    b = b @ T_stack[tok[j]]; b /= b.sum()
                reduced[i] = b
            beliefs_by_k[k] = reduced
    return beliefs_by_k

# Warmup numba
if HAS_NUMBA:
    _d = np.random.rand(3, 4, 4); _p = np.array([0.25, 0.25, 0.25, 0.25])
    _t = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    _ = _beliefs_numba(_t[:3], _d, _p)
    _ = _reduced_beliefs_table(_t, 1, 2, 2, np.random.rand(9, 4), 3)
    _ = _reduced_beliefs_direct(_t, 1, 2, 2, _d, _p)
    del _d, _p, _t
