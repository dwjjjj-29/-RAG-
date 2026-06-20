"""
HybridRAG 全局配置模板
=====================
使用方式：
  1. 复制 .env.example 为 .env，填入你的配置
  2. 或在以下各个 Config 类中直接修改默认值
  3. .env 中的值优先级高于此文件的默认值
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    """优先读环境变量，其次读 .env"""
    return os.environ.get(key, default)


# ============================================================
# 数据目录
# ============================================================
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"               # 原始文档存放目录
PROCESSED_DATA_DIR = DATA_DIR / "processed"   # 处理后的数据缓存
INDEX_DIR = BASE_DIR / "indexes"
FAISS_INDEX_DIR = INDEX_DIR / "faiss"         # FAISS 向量索引目录
BM25_INDEX_DIR = INDEX_DIR / "bm25"           # BM25 索引目录

# ============================================================
# 1. 向量化模型配置  用途：将文本转为语义向量
# ============================================================
class EmbeddingConfig:
    # [必填] 向量化模型名称（HuggingFace 模型 ID）
    model_name: str = _env("EMBEDDING_MODEL_NAME", "")
    # [可选] 模型缓存目录
    model_cache_dir: str = _env("MODEL_CACHE_DIR", str(BASE_DIR / "models" / "cache"))
    # [可选] 推理设备 cpu / cuda
    device: str = _env("EMBEDDING_DEVICE", "cpu")
    # [可选] 批处理大小
    batch_size: int = int(_env("EMBEDDING_BATCH_SIZE", "32"))
    # [可选] 输出向量维度
    vector_dim: int = int(_env("EMBEDDING_VECTOR_DIM", "384"))
    # [可选] 最大序列长度
    max_seq_length: int = int(_env("EMBEDDING_MAX_SEQ_LENGTH", "512"))
    # [可选] 是否归一化向量
    normalize_embeddings: bool = _env("EMBEDDING_NORMALIZE", "true").lower() == "true"


# ============================================================
# 2. 文档分割配置  用途：控制文档如何切分
# ============================================================
class ChunkConfig:
    # [可选] 每个文档块的最大字符数
    chunk_size: int = int(_env("CHUNK_SIZE", "500"))
    # [可选] 相邻块之间的重叠字符数
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "80"))


# ============================================================
# 3. 向量检索配置  用途：语义向量搜索
# ============================================================
class VectorRetrievalConfig:
    # [可选] 向量检索返回的候选数量
    top_k: int = int(_env("VECTOR_TOP_K", "15"))
    # [可选] FAISS 索引类型（IndexFlatL2 / IndexIVFFlat）
    index_type: str = _env("VECTOR_INDEX_TYPE", "IndexFlatL2")


# ============================================================
# 4. 关键词检索配置  用途：BM25 关键词匹配
# ============================================================
class BM25Config:
    # [可选] BM25 检索返回的候选数量
    top_k: int = int(_env("BM25_TOP_K", "15"))


# ============================================================
# 5. 重排序配置  用途：对融合后的候选文档精排
# ============================================================
class RerankConfig:
    # [必填] 重排序模型名称（HuggingFace 交叉编码器模型 ID）
    model_name: str = _env("RERANK_MODEL_NAME", "")
    # [可选] 模型缓存目录
    model_cache_dir: str = _env("MODEL_CACHE_DIR", str(BASE_DIR / "models" / "cache"))
    # [可选] 推理设备
    device: str = _env("RERANK_DEVICE", "cpu")
    # [可选] 重排后保留的文档数
    top_k: int = int(_env("RERANK_TOP_K", "8"))
    # [可选] 最低相关度阈值（低于此分数的文档被丢弃）
    score_threshold: float = float(_env("RERANK_SCORE_THRESHOLD", "0.0"))
    # [可选] HuggingFace 访问令牌（避免限速）
    hf_token: str = _env("HF_TOKEN", "")


# ============================================================
# 6. 大语言模型配置  用途：基于检索上下文生成回答
# ============================================================
class LLMConfig:
    # [必填] API 密钥
    api_key: str = _env("DEEPSEEK_API_KEY", "")
    # [必填] API 地址
    base_url: str = _env("LLM_BASE_URL", "")
    # [必填] 模型名称
    model_name: str = _env("LLM_MODEL_NAME", "")
    # [可选] 生成温度（低=更确定，高=更随机）
    temperature: float = float(_env("LLM_TEMPERATURE", "0.3"))
    # [可选] 最大输出 token 数
    max_tokens: int = int(_env("LLM_MAX_TOKENS", "1500"))


# ============================================================
# 7. 应用配置  用途：Web 界面和系统行为
# ============================================================
class AppConfig:
    # [可选] 页面标题
    title: str = _env("APP_TITLE", "RAG 智能问答系统")
    # [可选] 服务监听地址
    host: str = _env("GRADIO_HOST", "127.0.0.1")
    # [可选] 服务端口
    port: int = int(_env("GRADIO_PORT", "7860"))
    # [可选] 对话历史保留轮数
    history_length: int = int(_env("APP_HISTORY_LENGTH", "10"))


# ============================================================
# 支持的文件格式
# ============================================================
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".md", ".txt", ".doc"]
