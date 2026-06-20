import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from config.config import EmbeddingConfig
from utils.logging import logger


class EmbeddingModel:
    def __init__(self, config: EmbeddingConfig = EmbeddingConfig()):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"加载向量化模型: {self.config.model_name}")
            self._model = SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.model_cache_dir,
                device=self.config.device
            )
            self._model.max_seq_length = self.config.max_seq_length
            logger.info("向量化模型加载完成")

    def encode(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        if not texts:
            return np.array([], dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        self._load_model()
        embedding = self._model.encode(
            query,
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embedding.astype(np.float32).reshape(1, -1)

    @property
    def dimension(self) -> int:
        return self.config.vector_dim
