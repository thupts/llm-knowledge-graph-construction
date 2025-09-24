import os
from dotenv import load_dotenv
load_dotenv()

from llm import llm, embeddings
from graph import graph

from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Kết nối tới vector index đã được tạo trong Neo4j
# Lưu ý: tên index phải khớp với phần tạo index trong pipeline build KG (ví dụ `chunkVector`)
chunk_vector = Neo4jVector.from_existing_index(
    embedding=embeddings,
    graph=graph,
    index_name="chunkVector",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="textEmbedding"
)

# Prompt để tổng hợp kết quả tìm được
instructions = """You are an assistant that helps answer questions based on retrieved lesson chunks.
Use the following context to answer the question. If the answer is not in the context, say you don't know.
Always be clear and concise."""

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("human", "{input}"),
])

# Tạo retriever từ vector index
chunk_retriever = chunk_vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Chain để tổng hợp kết quả từ retriever
chunk_chain = create_stuff_documents_chain(llm, prompt)

# Tích hợp retriever + chain
retrieval_chain = create_retrieval_chain(chunk_retriever, chunk_chain)

def find_chunk(q: str):
    """
    Nhận câu hỏi -> tìm các chunk liên quan nhất trong Neo4j vector index -> trả kết quả
    """
    try:
        result = retrieval_chain.invoke({"input": q})
        return result
    except Exception as e:
        return {"error": str(e), "input": q, "context": []}
    


