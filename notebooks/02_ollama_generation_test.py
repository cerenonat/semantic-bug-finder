import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generation import test_ollama_connection


model_name = "mistral"

print("Testing Ollama connection with model:", model_name)

answer = test_ollama_connection(model=model_name)

print("\nOllama response:")
print(answer)