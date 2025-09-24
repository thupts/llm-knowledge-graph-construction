import os

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship

from dotenv import load_dotenv, find_dotenv
load_dotenv()

DOCS_PATH = "llm-knowledge-graph/data/course/pdfs"

# --- LLM and Embeddings: Azure/OpenAI switch ---
if os.getenv("AZURE_OPENAI_API_KEY"):
    llm = ChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )
    embedding_provider = OpenAIEmbeddings(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
else:
    llm = ChatOpenAI(
        openai_api_key=os.getenv('OPENAI_API_KEY'), 
        model_name="gpt-4.1"
    )
    embedding_provider = OpenAIEmbeddings(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model="text-embedding-ada-002"
    )

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

doc_transformer = LLMGraphTransformer(
    llm=llm,
)

# Load and split the documents
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

for chunk in chunks:

    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }
    
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:
            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="HAS_ENTITY"
                )
            )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)

# Create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};""")

import unittest

class TestEnvironment(unittest.TestCase):

    skip_env_variable_tests = True
    skip_openai_test = True
    skip_neo4j_test = True

    def test_env_file_exists(self):
        env_file_exists = True if find_dotenv() > "" else False
        if env_file_exists:
            TestEnvironment.skip_env_variable_tests = False
        self.assertTrue(env_file_exists, ".env file not found.")

    def env_variable_exists(self, variable_name):
        self.assertIsNotNone(
            os.getenv(variable_name),
            f"{variable_name} not found in .env file")

    def test_openai_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping OpenAI/Azure env variable test")

        # Check for either OpenAI or Azure OpenAI variables
        if os.getenv("AZURE_OPENAI_API_KEY"):
            self.env_variable_exists('AZURE_OPENAI_API_KEY')
            self.env_variable_exists('AZURE_OPENAI_ENDPOINT')
            self.env_variable_exists('AZURE_OPENAI_DEPLOYMENT')
            self.env_variable_exists('AZURE_OPENAI_API_VERSION')
        else:
            self.env_variable_exists('OPENAI_API_KEY')
        TestEnvironment.skip_openai_test = False

    def test_neo4j_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping Neo4j env variables test")

        self.env_variable_exists('NEO4J_URI')
        self.env_variable_exists('NEO4J_USERNAME')
        self.env_variable_exists('NEO4J_PASSWORD')
        TestEnvironment.skip_neo4j_test = False

    def test_openai_connection(self):
        if TestEnvironment.skip_openai_test:
            self.skipTest("Skipping OpenAI/Azure test")

        # Azure OpenAI connection test
        if os.getenv("AZURE_OPENAI_API_KEY"):
            import openai
            openai.api_type = "azure"
            openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            try:
                # List models for Azure deployment
                models = openai.Model.list()
            except Exception as e:
                models = None
            self.assertIsNotNone(
                models,
                "Azure OpenAI connection failed. Check the Azure OpenAI variables in .env file."
            )
        else:
            from openai import OpenAI, AuthenticationError
            llm = OpenAI()
            try:
                models = llm.models.list()
            except AuthenticationError as e:
                models = None
            self.assertIsNotNone(
                models,
                "OpenAI connection failed. Check the OPENAI_API_KEY key in .env file."
            )

    def test_neo4j_connection(self):
        if TestEnvironment.skip_neo4j_test:
            self.skipTest("Skipping Neo4j connection test")

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USERNAME'), 
                  os.getenv('NEO4J_PASSWORD'))
        )
        try:
            driver.verify_connectivity()
            connected = True
        except Exception as e:
            connected = False

        driver.close()

        self.assertTrue(
            connected,
            "Neo4j connection failed. Check the NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD values in .env file."
            )
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnvironment('test_env_file_exists'))
    suite.addTest(TestEnvironment('test_openai_variables'))
    suite.addTest(TestEnvironment('test_neo4j_variables'))
    suite.addTest(TestEnvironment('test_openai_connection'))
    suite.addTest(TestEnvironment('test_neo4j_connection'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
