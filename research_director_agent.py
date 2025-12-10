# research_director_agent.py
"""
Reads experiment logs and asks GPT to propose higher-level research directions.
Updated for openai>=1.0.0.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LOG_PATH = Path("logs/experiment_log.jsonl")


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


def ask_gpt_for_directions(records: List[Dict[str, Any]]) -> str:
    system_msg = (
        "You are a 'research director' overseeing a tiny ML lab. You will see past experiment "
        "logs for small MLPs regressing functions like sin(x) or composite functions.\n"
        "Write a structured list of proposed research directions in Markdown. For each direction, "
        "include:\n"
        "- A short title\n"
        "- Intuition\n"
        "- Concrete plan (what to sweep, what to change)\n"
        "- Success criteria\n"
        "Focus on things like new tasks, architectures, regularization, adaptive LR, interpretability, etc."
    )

    user_msg = "Here are the experiment records:\n\n" + json.dumps(records, indent=2)

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.5,
    )

    return resp.choices[0].message.content


def main():
    records = load_logs()
    if not records:
        print("No logs yet.")
        return

    text = ask_gpt_for_directions(records)
    print("\n==============================")
    print(" RESEARCH DIRECTOR SUGGESTIONS")
    print("==============================\n")
    print(text)
    print("\n==============================\n")


if __name__ == "__main__":
    main()
