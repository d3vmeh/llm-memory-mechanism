from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_response(query,context,llm):
    #context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
    context_text = context
    #print(context_text)
    prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            """
        You are a chatbot that stores memory in the form of a graph database of nodes and relationships.
        """
        ),
        (
            "user",
            """
            Here is your memory:
            {context}


            Here is the question:
            {question}"""
        ),
        ]
        )
    
    chain = (
         {"context": lambda x: context_text, "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
    )

    response_text = chain.invoke(query)
    formatted_response = f"Response: {response_text}\n"
    return formatted_response, response_text


