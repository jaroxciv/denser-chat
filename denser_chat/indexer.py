from denser_retriever.retriever import DenserRetriever
from denser_retriever.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from langchain_core.documents import Document
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, index_name):
        self.retriever = DenserRetriever(
            index_name=index_name,
            keyword_search=ElasticKeywordSearch(
                top_k=100,
                es_connection=create_elasticsearch_client(
                    url="http://localhost:9200",
                    username="elastic",
                    password="",
                ),
                drop_old=True,
                analysis="default",  # default or ik
            ),
            vector_db=None,
            reranker=None,
            embeddings=None,
            gradient_boost=None,
            search_fields=["annotations:keyword"],
        )
        self.ingest_bs = 2000

    def index(self, docs_file):
        logger.info(f"== Ingesting file {docs_file}")
        out = open(docs_file, "r")
        docs = []
        num_docs = 0
        for line in out:
            doc_dict = json.loads(line)
            docs.append(Document(**doc_dict))
            if len(docs) == self.ingest_bs:
                self.retriever.ingest(docs, overwrite_pid=True)
                docs = []
                num_docs += self.ingest_bs
                logger.info(f"Ingested {num_docs} documents")
        if len(docs) > 0:
            self.retriever.ingest(docs, overwrite_pid=True)
            logger.info(f"Ingested {num_docs + len(docs)} documents")

    def retrieve(self, query, top_k, meta_data):
        passages = self.retriever.retrieve(query, top_k, meta_data)
        return passages
