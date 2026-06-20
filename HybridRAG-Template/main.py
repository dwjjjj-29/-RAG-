import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import (
    AppConfig,
    ChunkConfig,
    EmbeddingConfig,
    VectorRetrievalConfig,
    BM25Config,
    RerankConfig,
    LLMConfig,
    RAW_DATA_DIR,
)
from modules.data_processing import DataProcessor
from modules.retrieval import HybridRetriever
from modules.reranking import FusionReranker
from modules.qa import QAModule
from ui.gradio_app import GradioApp
from utils.logging import logger, setup_logger


class HybridRAGSystem:
    def __init__(
        self,
        data_dir: str | Path = None,
        rebuild_index: bool = False,
    ):
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.rebuild_index = rebuild_index

        self.data_processor = DataProcessor(
            chunk_config=ChunkConfig(),
            embedding_config=EmbeddingConfig(),
        )
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=None,
            bm25_retriever=None,
        )
        self.fusion_reranker = FusionReranker(config=RerankConfig())
        self.qa_module = QAModule(config=LLMConfig())

        self.documents: List[Dict[str, Any]] = []
        self.last_result: Dict[str, Any] = {}

        self._initialize()

    def _initialize(self):
        if self.rebuild_index:
            logger.info("开始重建索引...")
            self._build_index()
        else:
            logger.info("尝试加载已有索引...")
            try:
                self.hybrid_retriever.load()
                self.documents = self.hybrid_retriever.documents
                logger.info(
                    f"索引加载成功，共 {len(self.documents)} 篇文档"
                )
            except Exception as e:
                logger.warning(f"加载索引失败，将重建: {e}")
                self._build_index()
        # 预加载重排模型，避免首次提问时才下载
        self.fusion_reranker.rerank_model.warmup()

    def _build_index(self):
        chunks = self.data_processor.process_all(self.data_dir)
        if not chunks:
            logger.error("未加载到任何文档，系统无法正常工作")
            return

        self.documents = [chunk.to_dict() for chunk in chunks]
        embeddings = np.array(
            [chunk.embedding for chunk in chunks if chunk.embedding is not None]
        )

        if len(embeddings) == 0:
            logger.error("所有文档块均未计算出向量，无法构建索引")
            return
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.hybrid_retriever.build_index(self.documents, embeddings)
        self.hybrid_retriever.save()
        logger.info("索引构建并保存完成")

    def add_documents(
        self, file_paths: List[str]
    ) -> int:
        new_chunks = []
        for fp in file_paths:
            chunks = self.data_processor.load_document(fp)
            chunks = self.data_processor.compute_embeddings(chunks)
            new_chunks.extend(chunks)

        if not new_chunks:
            return 0

        new_docs = [chunk.to_dict() for chunk in new_chunks]
        new_embeddings = np.array(
            [chunk.embedding for chunk in new_chunks if chunk.embedding is not None]
        )

        self.documents.extend(new_docs)

        self.hybrid_retriever.vector_retriever.index.add(new_embeddings)
        self.hybrid_retriever.vector_retriever.documents = self.documents

        self.hybrid_retriever.bm25_retriever.build_index(self.documents)

        self.hybrid_retriever.save()

        logger.info(f"增量添加 {len(new_chunks)} 个文档块")
        return len(new_chunks)

    def query(
        self,
        query_text: str,
        history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not self.documents:
            return {
                "answer": "系统尚未加载任何文档，请先添加文档后再提问。",
                "references": [],
                "context_length": 0,
                "model": "",
                "doc_count": 0,
            }

        logger.info(f"处理查询: '{query_text[:50]}...'")

        retrieval_results = self.hybrid_retriever.retrieve(query_text)

        reranked = self.fusion_reranker.process(
            query_text, retrieval_results, self.documents
        )

        result = self.qa_module.generate_answer(
            query_text, reranked, history
        )

        self.last_result = result
        return result

    def run_ui(
        self,
        server_name: str = "127.0.0.1",
        server_port: int = 7860,
        share: bool = False,
    ):
        def qa_chain(query, history=None):
            return self.query(query, history)

        app = GradioApp(qa_chain, system=self, config=AppConfig())
        logger.info(f"启动 Gradio 界面: http://{server_name}:{server_port}")
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
        )


def main():
    parser = argparse.ArgumentParser(
        description="混合检索RAG智能问答助手"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="文档数据目录路径",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建索引",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=AppConfig.port,
        help="Gradio 服务端口 (默认从 .env 读取)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=AppConfig.host,
        help="Gradio 服务主机地址 (默认从 .env 读取)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="是否创建公共分享链接",
    )
    args = parser.parse_args()

    setup_logger(level="INFO")

    logger.info("=" * 50)
    logger.info("混合检索RAG智能问答助手")
    logger.info("=" * 50)

    system = HybridRAGSystem(
        data_dir=args.data_dir,
        rebuild_index=args.rebuild,
    )

    system.run_ui(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
