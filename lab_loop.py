# lab_loop.py
"""
Outer loop that runs multiple lab cycles:
1) Ask GPT for new configs + train models (experiment_planner_agent)
2) Generate a research report (research_report_agent)

Updated for openai>=1.0.0 (uses updated modules).
"""

import datetime

import experiment_planner_agent
import research_report_agent


def run_one_cycle(cycle_idx: int):
    print("\n============================================================")
    print(f"🚀 STARTING LAB CYCLE {cycle_idx}")
    print("============================================================\n")

    print("[1] Running experiment planner + training...\n")
    experiment_planner_agent.main()

    print("[2] Generating research report from latest logs...\n")
    research_report_agent.main()

    ts = datetime.datetime.now().isoformat()
    print(f"✅ Cycle {cycle_idx} complete at {ts}")
    print("============================================================\n")


def main():
    for i in range(1, 4):
        run_one_cycle(i)


if __name__ == "__main__":
    main()
