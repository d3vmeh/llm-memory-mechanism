import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document



OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_DB_PASSWORD"]




graph = Neo4jGraph()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)

def update_graph(text):
    raw_documents = [Document(page_content=str(text))]
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=12)
    documents = text_splitter.split_documents(raw_documents)

    print(documents)
    print("Converting to graph documents")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    print("Adding to graph")
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)


    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid", 
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    graph._driver.close()