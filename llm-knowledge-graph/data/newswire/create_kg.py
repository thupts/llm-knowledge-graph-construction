import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_experimental.graph_transformers import LLMGraphTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from chatbot.graph import graph
from chatbot.llm import llm

load_dotenv()

ARTICLES_REQUIRED = [6, 8, 22]
DATA_PATH = Path("llm-knowledge-graph/data/newswire")
ARTICLE_FILENAME = DATA_PATH / "articles.csv"


def create_kg() -> None:
    with ARTICLE_FILENAME.open(encoding="utf8", newline="") as csvfile:
        articles = list(csv.DictReader(csvfile))

    article_transformer = LLMGraphTransformer(llm=llm)

    for index, article in enumerate(articles):
        if index not in ARTICLES_REQUIRED:
            continue

        print(f"Processing article {index}: {article['id']}")

        article_doc = [
            Document(
                page_content=article["text"],
                metadata={"id": article["id"]},
            )
        ]

        graph_docs = article_transformer.convert_to_graph_documents(article_doc)

        graph.query(
            "MERGE (a:Article {id: $id}) SET a.date = $date, a.text = $text",
            {"id": article["id"], "date": article["date"], "text": article["text"]},
        )

        article_node = Node(id=article["id"], type="Article")

        for graph_doc in graph_docs:
            for node in graph_doc.nodes:
                graph_doc.relationships.append(
                    Relationship(source=article_node, target=node, type="HAS_ENTITY")
                )

            graph.add_graph_documents([graph_doc])


if __name__ == "__main__":
    create_kg()
