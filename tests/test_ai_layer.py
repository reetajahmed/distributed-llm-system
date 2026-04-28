import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workers.rag import retrieve_context
from llm.model import run_llm


query = "What is load balancing?"

context = retrieve_context(query)
answer = run_llm(query, context)

print("Query:")
print(query)

print("\nRetrieved Context:")
print(context)

print("\nLLM Answer:")
print(answer)