# 🧪 Mini-OpenAI Research Lab

A tiny self-optimizing ML lab built in PyTorch and powered by the OpenAI API.

This project runs automated experiments on synthetic function-approximation tasks (like fitting sine waves), lets a GPT model propose new hyperparameters, trains networks, logs results, and generates research-style reports and visualizations.

Think of it as a mini version of an internal research loop: **planner → experiments → analysis → interpretability → new directions**.

---

## ✨ Features

- **Experiment Planner Agent**
  - Reads past experiment logs
  - Uses the OpenAI API to propose new hyperparameter configs  
    (layers, activation, learning rate, scheduler, epochs, etc.)

- **Training Loop (`lab_loop.py`)**
  - Runs multiple “lab cycles”
  - Each cycle:
    - Fetches new configs from the planner agent
    - Trains MLPs on synthetic tasks (e.g. `sin`, `sin_combo`)
    - Logs results to `logs/experiment_log.jsonl`
    - Auto-generates a **research report** summarizing trends

- **Task System**
  - Built-in tasks like:
    - `sin`
    - `sin_combo` (sin + cos mixture)
  - GPT-powered **dataset generator agent** can write new task definitions to  
    `tasks/generated_tasks.jsonl` as Python expressions, e.g.  
    `math.sin(x) + 0.3 * math.cos(2 * x) + 0.05 * x**3`.

- **Visualization (`visualize_results.py`)**
  - Reconstructs the best model’s architecture
  - Quickly retrains it to approximate the function again
  - Produces:
    - `plots/function_fit.png` – true vs predicted curve
    - `plots/hidden_vs_loss.png` – hidden size vs final loss
    - `plots/lr_vs_loss.png` – learning rate vs final loss

- **Interpretability (`interpretability_agent.py`)**
  - Hooks into the first hidden layer
  - Plots hidden activations across input space:
    - `plots/hidden_activations.png`

- **Research Director Agent (`research_director_agent.py`)**
  - Reads the log history
  - Proposes higher-level research directions:
    - new function families
    - regularization strategies
    - optimizers and LR schedules
    - alternative architectures

- **Streamlit Dashboard (`dashboard_app.py`)**
  - Web UI for the lab:
    - View all experiments as a table
    - Filter by task
    - See best config + metrics
    - View function-fit + hyperparam trend plots
    - Regenerate plots and hidden activations from the browser

---

## 🧱 Project Structure

```text
pytorch_practice/
├── lab_core.py                 # Core models, task generation, logging utilities
├── experiment_planner_agent.py # GPT-based hyperparameter planner
├── lab_loop.py                 # Runs multi-cycle lab experiments & reports
├── visualize_results.py        # Function fit + hyperparameter trend plots
├── interpretability_agent.py   # Hidden-unit activation visualization
├── dataset_generator_agent.py  # GPT-based new task generator
├── research_director_agent.py  # High-level research suggestions
├── dashboard_app.py            # Streamlit dashboard
├── logs/
│   └── experiment_log.jsonl    # Experiment records (appended over time)
└── tasks/
    └── generated_tasks.jsonl   # GPT-generated task definitions
