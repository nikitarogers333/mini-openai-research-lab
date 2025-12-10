# interpretability_agent.py
import matplotlib.pyplot as plt
import torch

from lab_core import (
    DEVICE,
    make_task_data,
    FlexibleMLP,
    load_all_experiments,
    find_best_experiment,
    normalize_layers_and_hidden,
    LOG_DIR,
)

PLOTS_DIR = LOG_DIR.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def build_model_from_cfg(cfg: dict) -> torch.nn.Module:
    """
    Build a FlexibleMLP from an experiment config.
    """
    cfg = normalize_layers_and_hidden(cfg)
    layers = cfg["layers"]
    activation = cfg.get("activation", "relu")
    model = FlexibleMLP(layers, activation=activation).to(DEVICE)
    return model


def main():
    records = load_all_experiments()
    if not records:
        print("No logs yet.")
        return

    best = find_best_experiment(records)
    if best is None:
        print("No valid experiments.")
        return

    best = normalize_layers_and_hidden(best)
    print(f"Loaded best config: {best}")

    task = best.get("task", "sin_combo")
    X_train, y_train, X_val, y_val = make_task_data(task, device=DEVICE)
    X = torch.cat([X_train, X_val], dim=0)
    y = torch.cat([y_train, y_val], dim=0)

    model = build_model_from_cfg(best)

    # Quick retrain so we get meaningful activations
    optimizer = torch.optim.Adam(model.parameters(), lr=float(best.get("lr", 0.01)))
    loss_fn = torch.nn.MSELoss()

    for _ in range(1000):
        model.train()
        optimizer.zero_grad()
        preds = model(X.to(DEVICE))
        loss = loss_fn(preds, y.to(DEVICE))
        loss.backward()
        optimizer.step()

    # Grab activations of the first hidden layer via a forward hook
    first_linear = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            first_linear = module
            break

    if first_linear is None:
        print("No Linear layers found in model.")
        return

    activations = []

    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu())

    handle = first_linear.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(X.to(DEVICE))

    handle.remove()

    if not activations:
        print("No activations captured.")
        return

    acts = activations[0].numpy()  # shape [N, H]
    xs = X.cpu().numpy().squeeze()

    # Sort by x for nicer plots
    order = xs.argsort()
    xs = xs[order]
    acts = acts[order]

    plt.figure(figsize=(8, 5))
    for i in range(acts.shape[1]):
        plt.plot(xs, acts[:, i], alpha=0.7)
    plt.title("Hidden layer activations (first hidden layer)")
    plt.xlabel("x")
    plt.ylabel("activation")
    out = PLOTS_DIR / "hidden_activations.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved hidden activation plot to {out}")


if __name__ == "__main__":
    main()
