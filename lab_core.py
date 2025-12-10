# lab_core.py
import os
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn

# -------------------------
# Global config / paths
# -------------------------

# You can switch this to "cuda" if you want to default to GPU
DEVICE = torch.device("cpu")

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "results"
TASKS_DIR = ROOT / "tasks"
LOG_DIR.mkdir(exist_ok=True)
TASKS_DIR.mkdir(exist_ok=True)

EXPERIMENT_LOG = LOG_DIR / "experiment_log.jsonl"
GENERATED_TASKS = TASKS_DIR / "generated_tasks.jsonl"


# -------------------------
# Task / dataset helpers
# -------------------------

def _builtin_task_fn(task: str):
    """Return a Python function f(x) implementing the built-in task on scalars."""
    if task == "sin":
        return lambda x: math.sin(x)
    if task == "sin_combo":
        # slightly harder: sin(x) + 0.3 cos(2x)
        return lambda x: math.sin(x) + 0.3 * math.cos(2 * x)
    return None


def _load_generated_task_expr(task: str) -> Optional[str]:
    """Look up a custom task by name in tasks/generated_tasks.jsonl."""
    if not GENERATED_TASKS.exists():
        return None
    with GENERATED_TASKS.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("name") == task:
                return rec.get("expression")
    return None


def _make_task_callable(task: str):
    """
    Return a callable f(x) for the given task name.

    Supports:
      - built-in: "sin", "sin_combo"
      - generated: tasks in tasks/generated_tasks.jsonl
        with fields { "name": ..., "expression": "math.sin(x)+..." }

    NOTE: f(x) is defined on *scalar* x (float), not tensors.
    """
    builtin = _builtin_task_fn(task)
    if builtin is not None:
        return builtin

    expr = _load_generated_task_expr(task)
    if expr is None:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Add it to {GENERATED_TASKS} or use one of: 'sin', 'sin_combo'."
        )

    # Build a safe-ish f(x) using the math module only.
    def f(x):
        return eval(expr, {"__builtins__": {}}, {"math": math, "x": x})

    return f


def make_task_data(
    task: str,
    device: torch.device = DEVICE,
    n_points: int = 1024,
    x_min: float = -2 * math.pi,
    x_max: float = 2 * math.pi,
    val_fraction: float = 0.2,
):
    """
    Create (train_X, train_y, val_X, val_y) for a given task.

    - Generates x in [x_min, x_max] with `n_points` samples.
    - y = f(x) where f is either a built-in or a generated expression.
    - Returns float32 tensors of shape [N, 1] on `device`.
    """
    f = _make_task_callable(task)

    # Make things deterministic and put xs directly on the target device.
    torch.manual_seed(0)
    xs = torch.linspace(
        x_min,
        x_max,
        steps=n_points,
        device=device,
        dtype=torch.float32,
    )

    # f(x) is defined on scalars; we evaluate pointwise and build a tensor.
    ys_list = [f(float(x)) for x in xs]
    ys = torch.tensor(ys_list, dtype=torch.float32, device=device)

    xs = xs.view(-1, 1)
    ys = ys.view(-1, 1)

    # Simple contiguous train/val split
    n_val = int(n_points * val_fraction)
    n_train = n_points - n_val

    X_train = xs[:n_train]
    y_train = ys[:n_train]
    X_val = xs[n_train:]
    y_val = ys[n_train:]

    return X_train, y_train, X_val, y_val


# -------------------------
# Model definition
# -------------------------

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def activation_from_name(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "swish":
        return Swish()
    raise ValueError(f"Unknown activation '{name}'")


class FlexibleMLP(nn.Module):
    """
    Simple MLP for 1D regression.

    `layers`: list of hidden layer sizes, e.g. [32], [64, 32], [32, 32, 16]
    """

    def __init__(self, layers: List[int], activation: str = "relu"):
        super().__init__()
        if not layers:
            layers = [32]

        act = activation_from_name(activation)
        modules: List[nn.Module] = []

        in_dim = 1
        for h in layers:
            modules.append(nn.Linear(in_dim, h))
            modules.append(act)
            in_dim = h

        modules.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Logging utilities
# -------------------------

def append_experiment_log(record: Dict[str, Any]):
    with EXPERIMENT_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_all_experiments() -> List[Dict[str, Any]]:
    if not EXPERIMENT_LOG.exists():
        return []
    records = []
    with EXPERIMENT_LOG.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def find_best_experiment(
    records: Optional[List[Dict[str, Any]]] = None,
    task: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if records is None:
        records = load_all_experiments()
    if not records:
        return None

    filtered = []
    for rec in records:
        if "final_loss" not in rec:
            continue
        if task is not None and rec.get("task") != task:
            continue
        filtered.append(rec)

    if not filtered:
        return None

    return min(filtered, key=lambda r: r["final_loss"])


def normalize_layers_and_hidden(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure cfg has both:
      - 'layers': List[int]  (full architecture)
      - 'hidden': int        (scalar proxy for plotting, etc.)
    """
    cfg = dict(cfg)  # shallow copy
    layers = cfg.get("layers")
    hidden = cfg.get("hidden")

    if layers is None and hidden is None:
        layers = [32]
    elif layers is None and hidden is not None:
        layers = [int(hidden)]
    elif layers is not None and hidden is None:
        # choose a simple scalar proxy for plotting
        hidden = int(layers[0]) if len(layers) == 1 else int(sum(layers))

    if layers is None:
        layers = [int(hidden)]

    cfg["layers"] = [int(h) for h in layers]
    cfg["hidden"] = int(hidden if hidden is not None else layers[0])
    return cfg
