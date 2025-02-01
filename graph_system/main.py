import query_llm
from update_graph import *

while True:
    message = input("Enter a message: ")
    update_graph(message)
    response = query_llm.get_response(message, None, llm)
    update_graph(response)
