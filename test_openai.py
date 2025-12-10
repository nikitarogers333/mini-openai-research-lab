# test_openai.py
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    resp = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": "Say 'Hi there!' exactly once."},
        ],
        temperature=0.0,
    )
    print(resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
