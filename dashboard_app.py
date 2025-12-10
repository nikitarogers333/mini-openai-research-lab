# dashboard_app.py
"""
Streamlit dashboard for the Mini-OpenAI Research Lab.

Features:
- Lists all experiments from logs/experiment_log.jsonl
- Shows best experiment (globally or per task)
- Displays saved plots (function fit, hyperparam trends, activations)
- Buttons to regenerate plots via visualize_results + interpretability_agent
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from lab_core import load_all_experiments, find_best_experiment, LOG_DIR
import visualize_results
import interpretability_agent


PLOTS_DIR = LOG_DIR.parent / "plots"


def load_records_df():
    records = load_all_experiments()
    if not records:
        return None
    df = pd.DataFrame(records)
    # nice ordering if columns exist
    preferred_cols = [
        "task",
        "layers",
        "hidden",
        "activation",
        "lr",
        "epochs",
        "scheduler",
        "patience",
        "best_epoch",
        "stopped_epoch",
        "final_loss",
        "note",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [
        c for c in df.columns if c not in preferred_cols
    ]
    return df[cols]


def show_best_config(df, task_filter=None):
    records = df.to_dict(orient="records")
    best = find_best_experiment(records, task=task_filter)
    if best is None:
        st.info("No valid experiments with `final_loss` found.")
        return

    st.subheader("Best Configuration")
    st.json(best)


def show_plots():
    st.subheader("Plots")

    # Helper for each plot
    def show_plot(name: str, label: str):
        path = PLOTS_DIR / name
        if path.exists():
            st.image(str(path), caption=label, use_column_width=True)
        else:
            st.caption(f"{label} — not generated yet.")

    col1, col2 = st.columns(2)
    with col1:
        show_plot("function_fit.png", "Function Fit")
        show_plot("hidden_vs_loss.png", "Hidden Size vs Final Loss")

    with col2:
        show_plot("lr_vs_loss.png", "Learning Rate vs Final Loss")
        show_plot("hidden_activations.png", "Hidden Layer Activations")


def main():
    st.set_page_config(page_title="Mini-OpenAI Research Lab", layout="wide")
    st.title("🧪 Mini-OpenAI Research Lab Dashboard")

    st.markdown(
        """
This dashboard visualizes the experiments run by your lab:

- **Experiment planner agent** suggests new configs using GPT.
- **Trainer** runs on synthetic tasks like `sin_combo`.
- **Logs** go to `logs/experiment_log.jsonl`.
- **Visualizers** generate plots under `plots/`.
        """
    )

    df = load_records_df()
    if df is None or df.empty:
        st.warning("No experiment logs found yet. Run `python lab_loop.py` first.")
        return

    # Sidebar controls
    st.sidebar.header("Controls")

    # Task filter
    tasks = sorted(df["task"].dropna().unique().tolist()) if "task" in df.columns else []
    selected_task = None
    if tasks:
        selected_task = st.sidebar.selectbox(
            "Filter by task (optional)", ["(all)"] + tasks
        )
        if selected_task == "(all)":
            selected_task = None

    # Regenerate plots
    if st.sidebar.button("🔄 Regenerate plots from latest logs"):
        visualize_results.main()
        st.sidebar.success("Plots regenerated.")

    if st.sidebar.button("🧬 Regenerate hidden activation plot"):
        interpretability_agent.main()
        st.sidebar.success("Hidden activation plot regenerated.")

    # Filtered dataframe
    if selected_task is not None and "task" in df.columns:
        df_view = df[df["task"] == selected_task].copy()
    else:
        df_view = df

    st.subheader("Experiment Log")
    st.dataframe(df_view, use_container_width=True)

    show_best_config(df_view, task_filter=selected_task)
    show_plots()


if __name__ == "__main__":
    main()
