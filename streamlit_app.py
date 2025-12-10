import os
import json
import ast
from typing import List, Any

import pandas as pd
import streamlit as st
import altair as alt


# ---------- Helpers ----------

def load_experiments(path: str = "results/experiment_log.jsonl") -> pd.DataFrame:
    """Load JSONL experiment log into a DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame()

    records: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                # If something weird is logged, just skip it
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Try to normalize some common fields
    if "layers" in df.columns:
        def _parse_layers(x: Any):
            if isinstance(x, (list, tuple)):
                return list(x)
            try:
                return list(ast.literal_eval(str(x)))
            except Exception:
                return None

        df["layers_parsed"] = df["layers"].apply(_parse_layers)
        df["num_layers"] = df["layers_parsed"].apply(
            lambda x: len(x) if isinstance(x, list) else None
        )
        df["layers_str"] = df["layers_parsed"].apply(
            lambda x: str(x) if isinstance(x, list) else str(x)
        )

    # Try to coerce numeric fields
    for col in ["lr", "epochs", "best_epoch", "stopped_epoch", "final_loss", "hidden"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If there’s a timestamp, parse it
    for col in ["timestamp", "time", "created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def get_best_experiment(df: pd.DataFrame) -> pd.Series | None:
    if df.empty or "final_loss" not in df.columns:
        return None
    return df.loc[df["final_loss"].idxmin()]


# ---------- Streamlit App ----------

st.set_page_config(
    page_title="Mini OpenAI Research Lab Dashboard",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Mini OpenAI Research Lab Dashboard")
st.caption("Browse experiments, plots, and best configurations from your autonomous lab loop.")

st.divider()

# Load data
df = load_experiments()

if df.empty:
    st.error(
        "No experiment logs found. "
        "Make sure you've run `python lab_loop.py` and that `results/experiment_log.jsonl` exists."
    )
    st.stop()

# ---------- Sidebar Filters ----------

st.sidebar.header("🔍 Filters")

# Task filter
task_col = "task" if "task" in df.columns else None
if task_col:
    tasks = sorted(df[task_col].dropna().unique().tolist())
    selected_tasks = st.sidebar.multiselect(
        "Task", options=tasks, default=tasks
    )
else:
    selected_tasks = None

# Activation filter
act_col = "activation" if "activation" in df.columns else None
if act_col:
    activations = sorted(df[act_col].dropna().unique().tolist())
    selected_activations = st.sidebar.multiselect(
        "Activation", options=activations, default=activations
    )
else:
    selected_activations = None

# Scheduler filter
sch_col = "scheduler" if "scheduler" in df.columns else None
if sch_col:
    schedulers = sorted(df[sch_col].dropna().unique().tolist())
    selected_schedulers = st.sidebar.multiselect(
        "Scheduler", options=schedulers, default=schedulers
    )
else:
    selected_schedulers = None

# LR slider
if "lr" in df.columns:
    lr_min, lr_max = float(df["lr"].min()), float(df["lr"].max())
    lr_range = st.sidebar.slider(
        "Learning rate range",
        min_value=float(lr_min),
        max_value=float(lr_max),
        value=(float(lr_min), float(lr_max)),
        format="%.5f",
    )
else:
    lr_range = None

# Hidden filter (if present)
if "hidden" in df.columns and df["hidden"].notnull().any():
    hidden_min, hidden_max = int(df["hidden"].min()), int(df["hidden"].max())
    hidden_range = st.sidebar.slider(
        "Hidden size range",
        min_value=int(hidden_min),
        max_value=int(hidden_max),
        value=(int(hidden_min), int(hidden_max)),
        step=1,
    )
else:
    hidden_range = None

# Apply filters
mask = pd.Series(True, index=df.index)

if selected_tasks is not None:
    mask &= df[task_col].isin(selected_tasks)

if selected_activations is not None:
    mask &= df[act_col].isin(selected_activations)

if selected_schedulers is not None:
    mask &= df[sch_col].isin(selected_schedulers)

if lr_range is not None:
    mask &= df["lr"].between(lr_range[0], lr_range[1])

if hidden_range is not None and "hidden" in df.columns:
    mask &= df["hidden"].between(hidden_range[0], hidden_range[1])

filtered_df = df[mask].copy()

st.subheader("📊 Experiment Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total experiments", len(df))

with col2:
    st.metric("Experiments (filtered)", len(filtered_df))

with col3:
    if "final_loss" in df.columns:
        st.metric("Best loss so far", f"{df['final_loss'].min():.4f}")
    else:
        st.metric("Best loss so far", "N/A")

# ---------- Best config card ----------

best_exp = get_best_experiment(filtered_df if not filtered_df.empty else df)

st.markdown("### 🏆 Best Configuration (under current filters)")
if best_exp is not None:
    cols = st.columns(4)
    with cols[0]:
        st.write("**Task**")
        st.code(str(best_exp.get("task", "—")))
    with cols[1]:
        st.write("**Activation**")
        st.code(str(best_exp.get("activation", "—")))
    with cols[2]:
        st.write("**Scheduler**")
        st.code(str(best_exp.get("scheduler", "—")))
    with cols[3]:
        st.write("**Final loss**")
        st.code(f"{best_exp.get('final_loss', float('nan')):.4f}")

    st.write("**Layers**:", best_exp.get("layers", best_exp.get("layers_str", "—")))
    st.write("**LR**:", best_exp.get("lr", "—"))
    st.write("**Epochs**:", best_exp.get("epochs", "—"))
    st.write("**Best epoch**:", best_exp.get("best_epoch", "—"))
    st.write("**Stopped epoch**:", best_exp.get("stopped_epoch", "—"))
else:
    st.info("No experiments available to summarize.")

st.divider()

# ---------- Plots from log ----------

st.subheader("📈 Hyperparameter Relationships")

if filtered_df.empty:
    st.warning("No experiments match the current filters.")
else:
    exp_cols = st.columns(3)

    # LR vs loss
    with exp_cols[0]:
        if "lr" in filtered_df.columns and "final_loss" in filtered_df.columns:
            chart = (
                alt.Chart(filtered_df)
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X("lr:Q", title="Learning rate", scale=alt.Scale(type="log")),
                    y=alt.Y("final_loss:Q", title="Final loss"),
                    color=alt.Color("activation:N", title="Activation", legend=None)
                    if "activation" in filtered_df.columns
                    else alt.value("#1f77b4"),
                    tooltip=list(filtered_df.columns),
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Need 'lr' and 'final_loss' columns to plot LR vs loss.")

    # Hidden vs loss
    with exp_cols[1]:
        if "hidden" in filtered_df.columns and "final_loss" in filtered_df.columns:
            chart = (
                alt.Chart(filtered_df)
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X("hidden:Q", title="Hidden size"),
                    y=alt.Y("final_loss:Q", title="Final loss"),
                    color=alt.Color("scheduler:N", title="Scheduler", legend=None)
                    if "scheduler" in filtered_df.columns
                    else alt.value("#ff7f0e"),
                    tooltip=list(filtered_df.columns),
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Need 'hidden' and 'final_loss' columns to plot hidden size vs loss.")

    # Epochs vs loss
    with exp_cols[2]:
        if "epochs" in filtered_df.columns and "final_loss" in filtered_df.columns:
            chart = (
                alt.Chart(filtered_df)
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X("epochs:Q", title="Max epochs"),
                    y=alt.Y("final_loss:Q", title="Final loss"),
                    color=alt.Color("task:N", title="Task", legend=None)
                    if "task" in filtered_df.columns
                    else alt.value("#2ca02c"),
                    tooltip=list(filtered_df.columns),
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Need 'epochs' and 'final_loss' columns to plot epochs vs loss.")

st.divider()

# ---------- Plots from /plots directory ----------

st.subheader("🖼️ Saved Plots")

plot_dir = "plots"
plot_files = {
    "Function fit (model vs true sin_combo)": "function_fit.png",
    "Hidden size vs loss": "hidden_vs_loss.png",
    "Learning rate vs loss": "lr_vs_loss.png",
    "Hidden activations (best model)": "hidden_activations.png",
}

cols = st.columns(2)
idx = 0

for title, fname in plot_files.items():
    full_path = os.path.join(plot_dir, fname)
    if os.path.exists(full_path):
        with cols[idx % 2]:
            st.markdown(f"**{title}**")
            st.image(full_path, use_column_width=True)
    idx += 1

if idx == 0:
    st.info("No images found in `plots/`. Run `visualize_results.py` and `interpretability_agent.py` first.")

st.divider()

# ---------- Raw table ----------

st.subheader("📄 Raw Experiment Table")

st.dataframe(
    filtered_df.sort_values(
        by="final_loss" if "final_loss" in filtered_df.columns else filtered_df.columns[0]
    ),
    use_container_width=True,
)
