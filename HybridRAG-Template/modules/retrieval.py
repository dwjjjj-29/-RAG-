import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss
import jieba
from rank_bm25 import BM25Okapi

from config.config import (
    VectorRetrievalConfig,
    BM25Config,
    FAISS_INDEX_DIR,
    BM25_INDEX_DIR,
)
from models.embeddings import EmbeddingModel
from utils.logging import logger


class VectorRetriever:
    def __init__(
        self,
        config: VectorRetrievalConfig = VectorRetrievalConfig(),
        embedding_model: EmbeddingModel = None,
    ):
        self.config = config
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index = None
        self.documents: List[Dict[str, Any]] = []
        self.is_trained = False

    def _create_index(self, dimension: int):
        if self.config.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
            self.is_trained = True
        elif self.config.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(100, max(1, len(self.documents) // 10))
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            self.is_trained = False
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.is_trained = True
        logger.info(f"创建向量索引: {self.config.index_type}, 维度: {dimension}")

    def build_index(
        self, documents: List[Dict[str, Any]], embeddings: np.ndarray
    ):
        self.documents = documents
        dimension = embeddings.shape[1]
        self._create_index(dimension)

        if not self.is_trained:
            logger.info("训练 IVF 索引...")
            self.index.train(embeddings)
            self.is_trained = True

        self.index.add(embeddings)
        logger.info(f"向量索引构建完成，共 {self.index.ntotal} 个向量")

    def search(
        self, query_embedding: np.ndarray, top_k: int = None
    ) -> List[Tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("向量索引为空")
            return []

        k = top_k or self.config.top_k
        k = min(k, self.index.ntotal)

        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.config.nprobe

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc_id = self.documents[idx].get("chunk_id", str(idx))
            distance = float(distances[0][i])
            score = 1.0 / (1.0 + distance)
            results.append((doc_id, score))

        return results

    def save(self, directory: str | Path = None):
        if directory is None:
            directory = FAISS_INDEX_DIR
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.index:
            faiss.write_index(self.index, str(directory / "index.faiss"))

        with open(directory / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"向量索引保存至: {directory}")

    def load(self, directory: str | Path = None):
        if directory is None:
            directory = FAISS_INDEX_DIR
        directory = Path(directory)

        index_path = directory / "index.faiss"
        doc_path = directory / "documents.pkl"

        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.is_trained = True
            logger.info(f"加载向量索引: {index_path}")

        if doc_path.exists():
            with open(doc_path, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(f"加载文档数据: {len(self.documents)} 条")

    def search_by_text(
        self, query: str, top_k: int = None
    ) -> List[Tuple[str, float]]:
        query_embedding = self.embedding_model.encode_query(query)
        return self.search(query_embedding, top_k)


class BM25Retriever:
    def __init__(self, config: BM25Config = BM25Config()):
        self.config = config
        self.bm25 = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text))

    def build_index(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.tokenized_docs = [
            self._tokenize(doc.get("content", "")) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"BM25 索引构建完成，共 {len(self.documents)} 篇文档")

    def search(
        self, query: str, top_k: int = None
    ) -> List[Tuple[str, float]]:
        if self.bm25 is None:
            logger.warning("BM25 索引为空")
            return []

        k = top_k or self.config.top_k
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        doc_score_pairs = [
            (self.documents[i].get("chunk_id", str(i)), float(scores[i]))
            for i in range(len(self.documents))
            if scores[i] > 0
        ]
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        results = doc_score_pairs[:k]

        return results

    def save(self, directory: str | Path = None):
        if directory is None:
            directory = BM25_INDEX_DIR
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "bm25_index.pkl", "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "tokenized_docs": self.tokenized_docs,
                },
                f,
            )
        logger.info(f"BM25 索引保存至: {directory}")

    def load(self, directory: str | Path = None):
        if directory is None:
            directory = BM25_INDEX_DIR
        directory = Path(directory)

        index_path = directory / "bm25_index.pkl"
        if index_path.exists():
            with open(index_path, "rb") as f:
                data = pickle.load(f)
            self.documents = data["documents"]
            self.tokenized_docs = data["tokenized_docs"]
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info(f"BM25 索引加载完成，共 {len(self.documents)} 篇文档")


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever = None,
        bm25_retriever: BM25Retriever = None,
    ):
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.documents: List[Dict[str, Any]] = []

    def build_index(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ):
        self.documents = documents
        self.vector_retriever.build_index(documents, embeddings)
        self.bm25_retriever.build_index(documents)

    def retrieve(
        self,
        query: str,
        top_k_vector: int = None,
        top_k_bm25: int = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        logger.info(f"开始双路检索: query='{query[:50]}...'")

        vector_k = top_k_vector or VectorRetrievalConfig.top_k
        bm25_k = top_k_bm25 or BM25Config.top_k

        vector_results = []
        bm25_results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_vector = executor.submit(
                self.vector_retriever.search_by_text, query, vector_k
            )
            future_bm25 = executor.submit(
                self.bm25_retriever.search, query, bm25_k
            )

            vector_results = future_vector.result()
            bm25_results = future_bm25.result()

        logger.info(
            f"双路检索完成: 向量检索 {len(vector_results)} 条, "
            f"关键词检索 {len(bm25_results)} 条"
        )

        return {
            "vector": vector_results,
            "bm25": bm25_results,
        }

    def get_document_by_id(self, doc_id: str) -> Dict[str, Any] | None:
        for doc in self.documents:
            if doc.get("chunk_id") == doc_id:
                return doc
        return None

    def save(self):
        self.vector_retriever.save()
        self.bm25_retriever.save()

    def load(self):
        self.vector_retriever.load()
        self.bm25_retriever.load()
        self.documents = self.vector_retriever.documents
