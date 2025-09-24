import os
from dotenv import load_dotenv
load_dotenv()

from llm import llm
from graph import graph

from langchain_neo4j import GraphCypherQAChain
from langchain.prompts import PromptTemplate

# Prompt template để hướng dẫn LLM sinh Cypher query
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a Neo4j graph database.
Instructions:
- Use only the provided relationship types and properties in the schema.
- Do not invent properties or relationship types that are not in the schema.
- Always use case-insensitive search when matching string values.
- Only return the Cypher statement (no explanation, no natural language).

Schema:
{schema}

The question is:
{question}"""

# Tạo PromptTemplate
cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

# Tạo GraphCypherQAChain
cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    allow_dangerous_requests=True  
)

def run_cypher(q: str):
    try:
        result = cypher_chain.invoke({"query": q})
        return result
    except Exception as e:
        return {"error": str(e)}