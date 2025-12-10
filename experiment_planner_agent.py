# experiment_planner_agent.py
"""
Experiment planner + runner for tiny ML lab.
Updated for openai>=1.0.0 using the new OpenAI client.
"""

import os
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from openai import OpenAI

# ========= OpenAI client =========
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========= Torch setup =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "experiment_log.jsonl"

TASKS_PATH = Path("tasks/generated_tasks.jsonl")


# ========= Data generation =========
def eval_generated_task(name: str, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a generated task expression from tasks/generated_tasks.jsonl.
    This is intentionally conservative: we only allow math.* and x.
    """
    import math as _math

    # Load tasks
    if not TASKS_PATH.exists():
        raise ValueError(f"Task '{name}' not found and no generated_tasks.jsonl present.")

    with TASKS_PATH.open("r") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("name") == name:
                expr = rec["expression"]
                break
        else:
            raise ValueError(f"Task '{name}' not found in {TASKS_PATH}")

    # Evaluate expression elementwise
    # We map tensor to Python float, compute expression, and map back.
    xs = x.detach().cpu().numpy().tolist()
    ys = []
    for xv in xs:
        # Safe eval context
        local_env = {"x": xv, "math": _math}
        yv = eval(expr, {"__builtins__": {}}, local_env)
        ys.append(float(yv))
    return torch.tensor(ys, dtype=torch.float32, device=x.device)


def make_task_data(task: str, device=device):
    """
    Build training/validation data for the given task name.
    task can be:
      - "sin":         y = sin(x)
      - "sin_combo":   y = sin(x) + 0.3 * cos(2x)
      - or any task defined in tasks/generated_tasks.jsonl
    """
    # Simple, portable seeding (works across torch versions)
    torch.manual_seed(0)

    # Input range
    X = torch.linspace(-2 * math.pi, 2 * math.pi, 400, device=device).unsqueeze(-1)

    if task == "sin":
        y = torch.sin(X)
    elif task == "sin_combo":
        y = torch.sin(X) + 0.3 * torch.cos(2 * X)
    else:
        # Load custom tasks from JSONL
        task_file = Path("tasks/generated_tasks.jsonl")
        if not task_file.exists():
            raise ValueError(f"Unknown task '{task}' and no tasks/generated_tasks.jsonl found.")

        import json
        import math as _math

        # Find task definition
        expr = None
        with task_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("name") == task or obj.get("task") == task:
                    expr = obj["expression"]
                    break

        if expr is None:
            raise ValueError(f"Task '{task}' not found in {task_file}")

        # Evaluate expression safely with x as a tensor
        x = X  # alias
        # Build namespace where "math" is Python math and "torch" is available if needed
        ns = {"math": _math, "torch": torch, "x": x}
        # Evaluate tensor expression. Important: we expect expr to use PyTorch ops (torch.sin, etc.)
        # or broadcast-safe Python math with x converted appropriately.
        y = eval(expr, {"__builtins__": {}}, ns)

    # Train/val split (80/20)
    n = X.shape[0]
    n_train = int(0.8 * n)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    return X_train, y_train, X_val, y_val


# ========= Model =========
class MLP(nn.Module):
    def __init__(self, in_dim: int, layers: List[int], out_dim: int, activation: str = "tanh"):
        super().__init__()
        if not layers:
            layers = [16]

        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "swish": lambda: nn.SiLU(),
            "sigmoid": nn.Sigmoid,
        }
        act_cls = activations.get(activation.lower(), nn.Tanh)

        modules = []
        last = in_dim
        for h in layers:
            modules.append(nn.Linear(last, h))
            modules.append(act_cls())
            last = h
        modules.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


# ========= JSON helper =========
def extract_json_from_text(text: str) -> Any:
    """
    Try to robustly extract JSON from a model response that may contain
    code fences or extra text.
    """
    # If inside ```json ... ```
    if "```" in text:
        # Take first fenced block
        parts = text.split("```")
        for part in parts:
            if part.strip().lower().startswith("json"):
                # remove 'json' and take rest
                body = part.strip()[4:].strip()
                text = body
                break

    # Strip whitespace
    text = text.strip()
    # Try direct load
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to locate first '[' and last ']'
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            return json.loads(snippet)
        raise


# ========= OpenAI call =========
def ask_gpt_for_configs(past_results: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    """
    Call OpenAI (new client) to propose new experiment configs.
    """
    system_msg = (
        "You are an ML research assistant designing small neural network experiments.\n"
        "You will receive a list of past runs (for tasks like 'sin' or 'sin_combo') with hyperparameters and losses.\n"
        "You must return a JSON list of new configurations ONLY, no extra commentary.\n"
        "Each config must be a JSON object with fields like:\n"
        "{\n"
        '  "task": "sin_combo",\n'
        '  "lr": 0.005,\n'
        '  "epochs": 3000,\n'
        '  "activation": "relu",\n'
        '  "scheduler": "none" | "step" | "cosine",\n'
        '  "patience": 300,\n'
        '  "layers": [16, 16]   // list of hidden layer sizes\n'
        '}\n'
        "You may also include an optional 'note' field with a brief rationale.\n"
        "Return ONLY a JSON array, no prose."
    )

    past_str = json.dumps(past_results, indent=2)
    user_msg = (
        f"Here are previous runs:\n\n{past_str}\n\n"
        f"Please propose {n} new configurations as described."
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
    )

    content = resp.choices[0].message.content
    configs = extract_json_from_text(content)

    if not isinstance(configs, list):
        raise ValueError("Model did not return a JSON list of configs.")
    return configs


# ========= Training =========
def run_single_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train one model on a given task + hyperparameters.
    Returns a dict with metrics plus the original config.
    """
    task = cfg.get("task", "sin_combo")
    lr = float(cfg.get("lr", 0.01))
    epochs = int(cfg.get("epochs", 2000))
    activation = cfg.get("activation", "tanh")
    scheduler_type = cfg.get("scheduler", "none")
    patience = int(cfg.get("patience", 400))

    # Layers: prefer 'layers' if present; fallback to single 'hidden'
    if "layers" in cfg and isinstance(cfg["layers"], list):
        layers = [int(x) for x in cfg["layers"]]
    else:
        h = int(cfg.get("hidden", 16))
        layers = [h]

    X_train, y_train, X_val, y_val = make_task_data(task, device=device)

    model = MLP(1, layers, 1, activation=activation).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, epochs // 10))
    else:
        scheduler = None

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if scheduler is not None:
            scheduler.step()

        if bad_epochs >= patience:
            # Early stop
            break

    # Use final validation loss as final_loss
    final_loss = best_val

    result = {
        "task": task,
        "layers": layers,
        "hidden": layers[0] if len(layers) == 1 else None,
        "activation": activation,
        "lr": lr,
        "epochs": epochs,
        "scheduler": scheduler_type,
        "patience": patience,
        "best_epoch": best_epoch,
        "stopped_epoch": best_epoch + bad_epochs if best_epoch > 0 else epoch,
        "final_loss": final_loss,
    }

    return result, model, (X_train, y_train, X_val, y_val)


def append_log_record(record: Dict[str, Any]) -> None:
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_all_logs() -> List[Dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    records = []
    with LOG_PATH.open("r") as f:
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


def find_best_config(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    return min(records, key=lambda r: r["final_loss"])


# ========= Main =========
def main():
    # Seeded past results for GPT context
    seeded_past = [
        {"task": "sin_combo", "hidden": 8, "lr": 0.01, "epochs": 2000, "final_loss": 0.0015},
        {"task": "sin_combo", "hidden": 16, "lr": 0.01, "epochs": 3000, "final_loss": 0.0007},
        {"task": "sin_combo", "hidden": 32, "lr": 0.001, "epochs": 2000, "final_loss": 0.0012},
    ]

    print("\n=== PAST RESULTS (SEEDED) ===")
    for r in seeded_past:
        print(
            f"task={r['task']}, hidden={r['hidden']}, lr={r['lr']}, "
            f"epochs={r['epochs']}, loss≈{r['final_loss']}"
        )

    print("\nAsking GPT for new experiment suggestions...\n")

    configs = ask_gpt_for_configs(seeded_past, n=5)

    print("GPT suggested these configs:\n")
    for cfg in configs:
        print(cfg)
    print("\n=== RUNNING NEW EXPERIMENTS ===\n")

    all_records = []

    for cfg in configs:
        note = cfg.get("note", "")
        desc = (
            f"task={cfg.get('task','sin_combo')}, "
            f"layers={cfg.get('layers', cfg.get('hidden'))}, "
            f"activation={cfg.get('activation', 'tanh')}, "
            f"lr={cfg.get('lr')}, epochs={cfg.get('epochs')}, "
            f"scheduler={cfg.get('scheduler', 'none')}"
        )
        print(f"Running: {desc}  ({note})")

        result, model, _ = run_single_experiment(cfg)
        print(
            f" → final_loss={result['final_loss']:.6f}, "
            f"best_epoch={result['best_epoch']}, "
            f"stopped_epoch={result['stopped_epoch']}\n"
        )

        append_log_record(result)
        all_records.append(result)

    # Load all logs (including previous runs if any)
    all_logs = load_all_logs()
    best = find_best_config(all_logs)

    print("\n==============================")
    print(" BEST CONFIG FOUND SO FAR")
    print("==============================")
    if best:
        print(f"task        : {best.get('task')}")
        layers = best.get("layers")
        if layers:
            print(f"layers      : {layers}")
        else:
            print(f"hidden      : {best.get('hidden')}")
        print(f"activation  : {best.get('activation')}")
        print(f"lr          : {best.get('lr')}")
        print(f"scheduler   : {best.get('scheduler')}")
        print(f"epochs      : {best.get('epochs')}")
        print(f"best_epoch  : {best.get('best_epoch')}")
        print(f"stopped_ep  : {best.get('stopped_epoch')}")
        print(f"final_loss  : {best.get('final_loss'):.6f}")
    else:
        print("No records yet.")

    print()


if __name__ == "__main__":
    main()
