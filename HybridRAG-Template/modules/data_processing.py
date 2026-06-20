import os
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import (
    ChunkConfig,
    EmbeddingConfig,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FAISS_INDEX_DIR,
    BM25_INDEX_DIR,
    SUPPORTED_FILE_TYPES,
)
from models.embeddings import EmbeddingModel
from utils.logging import logger
from utils.text_utils import clean_text


class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray | None = None
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = metadata
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
        }


class DataProcessor:
    def __init__(
        self,
        chunk_config: ChunkConfig = ChunkConfig(),
        embedding_config: EmbeddingConfig = EmbeddingConfig(),
    ):
        self.chunk_config = chunk_config
        self.embedding_model = EmbeddingModel(embedding_config)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_config.chunk_size,
            chunk_overlap=chunk_config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )

    def _get_loader(self, file_path: str):
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(file_path)
        elif ext == ".docx":
            return UnstructuredWordDocumentLoader(file_path)
        elif ext == ".md":
            return UnstructuredMarkdownLoader(file_path)
        elif ext == ".txt":
            return TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _generate_chunk_id(self, content: str, source: str, index: int) -> str:
        raw = f"{source}:{index}:{content[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def load_document(self, file_path: str) -> List[DocumentChunk]:
        logger.info(f"加载文档: {file_path}")
        loader = self._get_loader(file_path)
        documents = loader.load()
        source = Path(file_path).name

        full_text = "\n".join([doc.page_content for doc in documents])
        full_text = clean_text(full_text)

        splits = self.text_splitter.split_text(full_text)
        chunks = []
        for i, split in enumerate(splits):
            chunk_id = self._generate_chunk_id(split, source, i)
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=split,
                metadata={
                    "source": source,
                    "file_path": file_path,
                    "chunk_index": i,
                    "total_chunks": len(splits),
                },
            )
            chunks.append(chunk)

        logger.info(f"文档 {source} 分割完成: {len(chunks)} 个块")
        return chunks

    def load_documents_from_directory(
        self, directory: str | Path = None
    ) -> List[DocumentChunk]:
        if directory is None:
            directory = RAW_DATA_DIR
        directory = Path(directory)

        all_chunks = []
        for file_path in directory.glob("*"):
            if file_path.suffix.lower() in SUPPORTED_FILE_TYPES:
                try:
                    chunks = self.load_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"加载文档 {file_path} 失败: {e}")

        logger.info(f"共加载 {len(all_chunks)} 个文档块")
        return all_chunks

    def compute_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        texts = [chunk.content for chunk in chunks]
        logger.info(f"计算 {len(texts)} 个文本块的向量")
        embeddings = self.embedding_model.encode(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return chunks

    def save_processed_data(
        self, chunks: List[DocumentChunk], filename: str = "chunks.pkl"
    ):
        save_path = PROCESSED_DATA_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "chunks": [chunk.to_dict() for chunk in chunks],
            "embeddings": np.array(
                [chunk.embedding for chunk in chunks if chunk.embedding is not None]
            ),
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"处理后的数据保存至: {save_path}")

    def load_processed_data(
        self, filename: str = "chunks.pkl"
    ) -> tuple:
        load_path = PROCESSED_DATA_DIR / filename
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"加载处理后的数据: {load_path}")
        return data["chunks"], data["embeddings"]

    def process_all(
        self, directory: str | Path = None, save: bool = True
    ) -> List[DocumentChunk]:
        chunks = self.load_documents_from_directory(directory)
        if not chunks:
            logger.warning("未加载到任何文档")
            return chunks
        chunks = self.compute_embeddings(chunks)
        if save:
            self.save_processed_data(chunks)
        logger.info(f"数据处理完成，共 {len(chunks)} 个块")
        return chunks
