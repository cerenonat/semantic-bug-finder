import requests


# ============================================================
# Generation Module for Semantic Bug Finder
#
# This module sends the user query and retrieved FAISS results
# to Ollama/Mistral and asks for a structured debugging report.
# ============================================================


def generate_bug_summary(user_query, retrieved_context, model="mistral"):
    prompt = f"""
You are a senior software debugging assistant.

Your task is to analyze a user's software bug report using ONLY:
1. The user query
2. The retrieved similar bug reports

User query:
{user_query}

Retrieved similar bug reports:
{retrieved_context}

Generate a structured debugging report using exactly these sections:

1. Observed Behaviour
Describe what currently happens in the software according to the user query and retrieved reports.

2. Expected Behaviour
Describe what should happen instead.

3. Steps to Reproduce
Write 3 to 5 clear steps that could reproduce the issue.
Use only reasonable steps based on the user query and retrieved bug reports.

4. If there is not enough information to write exact reproduction steps, write cautious and general steps instead of inventing details.

5. Evidence from Retrieved Bug Reports
Briefly mention which retrieved reports are related and why.

6. Possible Cause
Explain the most likely cause based only on the retrieved reports.

7. Suggested Fix
Give a practical debugging direction or fix.

Important rules:
- Do not invent unsupported details.
- Do not mention facts that are not supported by the user query or retrieved bug reports.
- If the retrieved reports are only partially related, explicitly say that the evidence is partial.
- Keep the answer concise and clear.
"""

    request = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    url = "http://localhost:11434/api/generate"

    response = requests.post(url, json=request, timeout=120)

    if response.status_code != 200:
        raise Exception(f"Ollama request failed: {response.text}")

    data = response.json()

    if "response" not in data:
        raise Exception(f"No 'response' field returned by Ollama: {data}")

    return data["response"]


def test_ollama_connection(model="mistral"):
    prompt = "Say hello in one short sentence."

    request = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    url = "http://localhost:11434/api/generate"

    response = requests.post(url, json=request, timeout=60)

    if response.status_code != 200:
        raise Exception(f"Ollama test failed: {response.text}")

    data = response.json()

    return data.get("response", "")