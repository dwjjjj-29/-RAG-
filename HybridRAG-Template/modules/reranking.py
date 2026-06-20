from typing import List, Tuple, Dict, Any

from config.config import RerankConfig
from models.rerank import RerankModel
from utils.logging import logger


def _normalize_scores(
    results: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    if not results:
        return results
    scores = [r[1] for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-8:
        return [(r[0], 1.0) for r in results]
    return [(r[0], (r[1] - min_s) / (max_s - min_s)) for r in results]


def _reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


class FusionReranker:
    def __init__(self, config: RerankConfig = RerankConfig()):
        self.config = config
        self.rerank_model = RerankModel(config)

    def fuse_results(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        documents: List[Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        logger.info(
            f"融合结果: 向量 {len(vector_results)} 条, "
            f"BM25 {len(bm25_results)} 条"
        )

        fused = _reciprocal_rank_fusion(vector_results, bm25_results)

        seen_ids = set()
        deduped = []
        for doc_id, score in fused:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                deduped.append((doc_id, score))

        logger.info(f"融合去重后: {len(deduped)} 条")
        return deduped

    def rerank(
        self,
        query: str,
        fused_results: List[Tuple[str, float]],
        documents: List[Dict[str, Any]],
    ) -> List[Tuple[str, str, float, str]]:
        if not fused_results:
            return []

        doc_map = {doc.get("chunk_id", ""): doc for doc in documents}

        doc_texts = []
        doc_ids = []
        doc_sources = []
        for doc_id, _ in fused_results:
            doc = doc_map.get(doc_id)
            if doc:
                doc_texts.append(doc.get("content", ""))
                doc_ids.append(doc_id)
                metadata = doc.get("metadata", {})
                doc_sources.append(metadata.get("source", ""))

        if not doc_texts:
            return []

        logger.info(f"重排 {len(doc_texts)} 条候选文档")
        reranked = self.rerank_model.rerank(query, doc_texts, doc_ids)

        doc_source_map = dict(zip(doc_ids, doc_sources))
        enriched = []
        for doc_id, content, score in reranked:
            source = doc_source_map.get(doc_id, "")
            enriched.append((doc_id, content, score, source))

        logger.info(
            f"重排完成，保留 {len(enriched)} 条 (阈值: {self.config.score_threshold})"
        )

        return enriched

    def process(
        self,
        query: str,
        retrieval_results: Dict[str, List[Tuple[str, float]]],
        documents: List[Dict[str, Any]],
    ) -> List[Tuple[str, str, float, str]]:
        vector_results = retrieval_results.get("vector", [])
        bm25_results = retrieval_results.get("bm25", [])

        vector_results = _normalize_scores(vector_results)
        bm25_results = _normalize_scores(bm25_results)

        fused = self.fuse_results(vector_results, bm25_results, documents)

        reranked = self.rerank(query, fused, documents)

        return reranked
