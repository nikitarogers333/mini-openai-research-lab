# dataset_generator_agent.py
"""
Dataset generator agent: asks GPT to propose new synthetic regression tasks.
Updated for openai>=1.0.0 (OpenAI client).
"""

import os
import json
from pathlib import Path
from typing import Any, List, Dict

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TASKS_DIR = Path("tasks")
TASKS_DIR.mkdir(exist_ok=True)
TASKS_PATH = TASKS_DIR / "generated_tasks.jsonl"


def extract_json_from_text(text: str) -> Any:
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if part.strip().lower().startswith("json"):
                body = part.strip()[4:].strip()
                text = body
                break
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            return json.loads(snippet)
        raise


def ask_gpt_for_tasks(n: int = 3) -> List[Dict[str, Any]]:
    system_msg = (
        "You are a synthetic regression task designer.\n"
        "Return ONLY a JSON array of objects. Each object must have:\n"
        '  "name": short_snake_case_name,\n'
        '  "expression": a Python expression in terms of x and math, e.g. "math.sin(x) + 0.3 * x**2",\n'
        '  "notes": a short human-readable description.\n'
        "Use only safe math library functions via 'math', and x.\n"
        "Return only the JSON, no explanation."
    )

    user_msg = f"Please propose {n} new 1D regression tasks."

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
    )

    content = resp.choices[0].message.content
    tasks = extract_json_from_text(content)
    if not isinstance(tasks, list):
        raise ValueError("Model did not return a JSON list of tasks.")
    return tasks


def append_tasks(tasks: List[Dict[str, Any]]) -> None:
    with TASKS_PATH.open("a") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")


def main():
    tasks = ask_gpt_for_tasks(n=3)
    append_tasks(tasks)

    print(f"Wrote {len(tasks)} new task definitions to {TASKS_PATH}")
    if tasks:
        print("Examples:")
        for t in tasks[:3]:
            print(f"- {t['name']} : {t['expression']}")


if __name__ == "__main__":
    main()
