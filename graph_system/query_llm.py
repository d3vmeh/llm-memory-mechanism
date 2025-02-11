from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate 
from typing import List
from pydantic import BaseModel, Field
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

import os


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
graph = Neo4jGraph()

vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid", 
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )



class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, law, technology, policy, country, organization, or business entities that "
        "appear in the text",
    )   

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

entity_chain = prompt | llm.with_structured_output(Entities)

def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()



def get_response(query,llm):
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", """
        You are a chatbot that stores memory in the form of a graph database of nodes and relationships. Respond and talk to the user.
        """),
        ("user", """
        The following sources contain your memory:                                                                                                               
        
        1. **Structured Data**: This includes major entities and their relationships from your conversations with the user. Use this information to understand the connections and significance of each entity.

        2. **Unstructured Data**: This contains relevant text extracted from various conversations. Analyze this text for pertinent information that supports your response.

        
    
        {context}

        Here is the question for you to answer: {question}
        Use natural language and be detailed and thorough. Remember, you are a chatbot. Respond to the user, do not randomly summarize unless the user asks you to.
        Do not present false information. If you are unsure of something, say you don't know.
        Answer:
        """)
        ]
        )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response_text = chain.invoke(query)
    formatted_response = f"Response: {response_text}\n"
    return formatted_response, response_text


def structured_retriever(question: str) -> str:
    result = ""
    nodes= []
    neighbors = []
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        print(entities.names)
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output, node.id AS nodeId, neighbor.id AS neighborId
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output, node.id AS nodeId, neighbor.id AS neighborId
            }
            RETURN output, nodeId, neighborId LIMIT 100
            """,
            {"query": generate_full_text_query(entity)},
        )
        

        # This would be used for an additional neighbor. Needs further testing.
        alternate_query = """CALL db.index.fulltext.queryNodes('entity', $query, {limit:3})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)-[r2:!MENTIONS]->(neighbor2)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id + ' - ' + type(r2) + '->' + neighbor2.id AS output, node.id AS nodeId, neighbor.id AS neighborId, neighbor2.id AS neighbor2Id
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)<-[r2:!MENTIONS]-(neighbor2)
              RETURN neighbor2.id + ' - ' + type(r2) + '->' + neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output, node.id AS nodeId, neighbor.id AS neighborId, neighbor2.id AS neighbor2Id
            }
            RETURN output, nodeId, neighborId LIMIT 100
            """



        for n in response:
            node = n['nodeId']
            if node not in nodes:
                nodes.append(node)        
        for e in response:
            neighbor = e['neighborId']
            if neighbor not in neighbors:
                neighbors.append(neighbor)
        result += "\n".join([el['output'] for el in response])

    return result, nodes, neighbors

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data, related_nodes, neighbors = structured_retriever(question) 
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)] 
    final_data = f"""Structured data:
                    {structured_data}

                    Unstructured data:
                    {"#Document ". join(unstructured_data)}

                    """

    print("Final data:")
    print(final_data)
    return final_data

