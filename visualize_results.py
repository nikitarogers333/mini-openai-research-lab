# visualize_results.py
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

# Where plots will be saved (e.g. logs/../plots)
PLOTS_DIR = LOG_DIR.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def reconstruct_model_and_data(cfg: dict):
    """
    Rebuild a model matching the best config and re-train it quickly
    on the full dataset so we can visualize a function fit.
    """
    cfg = normalize_layers_and_hidden(cfg)
    task = cfg.get("task", "sin_combo")
    layers = cfg["layers"]
    activation = cfg.get("activation", "relu")

    # Get train/val data and merge for visualization
    X_train, y_train, X_val, y_val = make_task_data(task, device=DEVICE)
    X_full = torch.cat([X_train, X_val], dim=0)
    y_full = torch.cat([y_train, y_val], dim=0)

    # Build model
    model = FlexibleMLP(layers, activation=activation).to(DEVICE)

    # Quick re-train solely for visualization (weights are not saved elsewhere)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 0.01)))
    loss_fn = torch.nn.MSELoss()

    for _ in range(1000):
        model.train()
        optimizer.zero_grad()
        preds = model(X_full.to(DEVICE))
        loss = loss_fn(preds, y_full.to(DEVICE))
        loss.backward()
        optimizer.step()

    return model, X_full.cpu(), y_full.cpu(), task


def plot_function_fit(model: torch.nn.Module, X: torch.Tensor, y_true: torch.Tensor, task_name: str):
    """
    Plot the true function values and the model predictions over X.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X.to(DEVICE)).cpu()

    X_np = X.squeeze().numpy()
    y_true_np = y_true.squeeze().numpy()
    y_pred_np = y_pred.squeeze().numpy()

    # Sort by x so the lines look nice
    order = X_np.argsort()
    X_np = X_np[order]
    y_true_np = y_true_np[order]
    y_pred_np = y_pred_np[order]

    plt.figure()
    plt.plot(X_np, y_true_np, label="true", linewidth=2)
    plt.plot(X_np, y_pred_np, label="pred", linestyle="--")
    plt.title(f"Function Fit – task='{task_name}'")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    out_path = PLOTS_DIR / "function_fit.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved function-fit plot to {out_path}")


def plot_hparam_trends(records):
    """
    Scatter plots:
      - hidden (normalized from layers) vs final_loss
      - lr vs final_loss
    """
    norm_recs = [normalize_layers_and_hidden(r) for r in records if "final_loss" in r]
    if not norm_recs:
        return

    hs = [r["hidden"] for r in norm_recs]
    lrs = [r.get("lr", 0.01) for r in norm_recs]
    losses = [r["final_loss"] for r in norm_recs]

    # hidden vs loss
    plt.figure()
    plt.scatter(hs, losses)
    plt.xlabel("hidden (proxy from layers)")
    plt.ylabel("final_loss")
    plt.title("Hidden size vs final loss")
    out1 = PLOTS_DIR / "hidden_vs_loss.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved hidden-vs-loss plot to {out1}")

    # lr vs loss
    plt.figure()
    plt.scatter(lrs, losses)
    plt.xlabel("learning rate")
    plt.ylabel("final_loss")
    plt.title("LR vs final loss")
    out2 = PLOTS_DIR / "lr_vs_loss.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved lr-vs-loss plot to {out2}")


def main():
    records = load_all_experiments()
    if not records:
        print("No experiment logs yet.")
        return

    best = find_best_experiment(records)
    if best is None:
        print("No valid experiments.")
        return

    best = normalize_layers_and_hidden(best)

    print("Loaded last log entry.")
    print(f"Best config: {best}")

    model, X, y_true, task = reconstruct_model_and_data(best)
    plot_function_fit(model, X, y_true, task)
    plot_hparam_trends(records)


if __name__ == "__main__":
    main()
