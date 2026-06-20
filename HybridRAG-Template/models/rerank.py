from typing import List, Tuple
from sentence_transformers import CrossEncoder

from config.config import RerankConfig
from utils.logging import logger


class RerankModel:
    def __init__(self, config: RerankConfig = RerankConfig()):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"加载重排模型: {self.config.model_name}")
            self._model = CrossEncoder(
                self.config.model_name,
                cache_folder=self.config.model_cache_dir,
                device=self.config.device
            )
            logger.info("重排模型加载完成")

    def warmup(self):
        """启动时预加载重排模型，避免首次提问时才下载"""
        self._load_model()
        logger.info("重排模型预热完成")

    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str] | None = None
    ) -> List[Tuple[str, str, float]]:
        self._load_model()
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False
        )

        results = []
        for i, score in enumerate(scores):
            doc_id = doc_ids[i] if doc_ids else str(i)
            score_val = float(score)
            if score_val >= self.config.score_threshold:
                results.append((doc_id, documents[i], score_val))

        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:self.config.top_k]

        return results

    def predict_score(self, query: str, document: str) -> float:
        self._load_model()
        score = self._model.predict([[query, document]])
        return float(score[0])
