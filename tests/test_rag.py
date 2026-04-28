import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workers.rag import retrieve_context


query = "What is load balancing?"
context = retrieve_context(query)

print("Query:", query)
print("\nRetrieved Context:")
print(context)