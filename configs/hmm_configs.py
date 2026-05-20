"""HMM configs with order-1 and order-0 constructors."""
import numpy as np
from src.hmm.definitions import *

HMMS = {
    "Mess3": {
        "fn": mess3_matrices, "order_one_fn": mess3_order_one, "order_zero_fn": None,
        "params": [(0.005,0.01),(0.005,0.02),(0.01,0.02),(0.05,0.02),(0.10,0.02),
                   (0.60,0.02),(0.70,0.02),(0.80,0.02),(0.85,0.02),(0.90,0.02)],
        "label_fn": lambda p: f"a={p[0]}, x={p[1]}",
        "token_names": np.array(["F","Q","V"]), "n_tokens": 3, "n_states": 3,
    },
    "Arch": {
        "fn": arch_matrices, "order_one_fn": arch_order_one, "order_zero_fn": arch_order_zero,
        "params": [(a,) for a in np.arange(0.90, 1.00, 0.01).round(2)],
        "label_fn": lambda p: f"a={p[0]}",
        "token_names": np.array(["F","Q","V"]), "n_tokens": 3, "n_states": 4,
    },
    "Wing": {
        "fn": wing_matrices, "order_one_fn": wing_order_one, "order_zero_fn": wing_order_zero,
        "params": [(a, 0.4) for a in np.arange(0.90, 1.00, 0.01).round(2)],
        "label_fn": lambda p: f"a={p[0]}, x={p[1]}",
        "token_names": np.array(["F","Q"]), "n_tokens": 2, "n_states": 3,
    },
    "Strata": {
        "fn": strata_matrices, "order_one_fn": strata_order_one, "order_zero_fn": strata_order_zero,
        "params": [(a, 0.38, 0.54) for a in np.arange(0.90, 1.00, 0.01).round(2)],
        "label_fn": lambda p: f"a={p[0]}, t0={p[1]}, t1={p[2]}",
        "token_names": np.array(["F","Q"]), "n_tokens": 2, "n_states": 3,
    },
    "Spiral": {
        "fn": spiral_matrices, "order_one_fn": spiral_order_one, "order_zero_fn": spiral_order_zero,
        "params": [(a,) for a in np.arange(0.01, 0.11, 0.01).round(2)],
        "label_fn": lambda p: f"a={p[0]}",
        "token_names": np.array(["F","Q"]), "n_tokens": 2, "n_states": 3,
    },
}
REPRESENTATIVES = {"Mess3": (0.01,0.02), "Arch": (0.9,), "Wing": (0.98,0.4), "Strata": (0.97,0.38,0.54)}
