# research_report_agent.py
"""
Reads experiment logs and asks GPT to write a research-style summary.
Updated for openai>=1.0.0.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LOG_PATH = Path("results/experiment_log.jsonl")


def load_logs() -> List[Dict[str, Any]]:
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


def ask_gpt_for_report(records: List[Dict[str, Any]]) -> str:
    system_msg = (
        "You are an AI research assistant. You will be given a list of small neural network "
        "experiments (hyperparameters + final loss). Write a concise, structured research report "
        "in Markdown with these sections:\n"
        "1. Overview\n"
        "2. Hyperparameter Trends\n"
        "3. Best Configurations (and why they work)\n"
        "4. Failure Modes\n"
        "5. Recommendations for Next Experiments\n"
        "6. Possible Improvements\n"
        "Assume the tasks are function regression (e.g., sin(x) or composite functions)."
    )

    user_msg = "Here are the experiment records:\n\n" + json.dumps(records, indent=2)

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
    )

    return resp.choices[0].message.content


def main():
    records = load_logs()
    if not records:
        print("No logs yet.")
        return

    report = ask_gpt_for_report(records)

    print("\n==============================")
    print("  AI RESEARCH REPORT")
    print("==============================\n")
    print(report)
    print("\n==============================\n")


if __name__ == "__main__":
    main()
