from typing import List, Tuple, Dict, Any
from openai import OpenAI

from config.config import LLMConfig
from utils.logging import logger
from utils.text_utils import format_reference


SYSTEM_PROMPT = """你是一个专业的智能问答助手。你的职责是基于提供的上下文信息，准确、简洁地回答用户的问题。

回答要求：
1. 仅基于提供的上下文信息回答问题，不要编造信息
2. 如果上下文中没有足够的信息来回答问题，请明确告知用户
3. 回答要简洁清晰，使用中文
4. 适当引用上下文中的具体信息
5. 涉及流程、规定等内容时，尽量提供详细的步骤说明"""


def _build_context(documents: List[Tuple[str, str, float, str]]) -> str:
    context_parts = []
    for i, (doc_id, content, score, source) in enumerate(documents, 1):
        context_parts.append(f"[文档 {i}]\n{content}\n")
    return "\n".join(context_parts)


def _build_messages(
    query: str,
    context: str,
    history: List[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if history:
        for msg in history[-6:]:
            messages.append(msg)

    user_content = f"""请基于以下上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{query}

请基于上述上下文给出回答。如果上下文信息不足以回答问题，请说明缺少哪些信息。"""

    messages.append({"role": "user", "content": user_content})
    return messages


class QAModule:
    def __init__(self, config: LLMConfig = LLMConfig()):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def generate_answer(
        self,
        query: str,
        documents: List[Tuple[str, str, float, str]],
        history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        context = _build_context(documents)
        messages = _build_messages(query, context, history)

        logger.info(f"调用大模型: {self.config.model_name}")

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )

            answer = response.choices[0].message.content.strip()

            references = []
            for doc_id, content, score, source in documents:
                ref = format_reference(content, score, source)
                references.append(ref)

            result = {
                "answer": answer,
                "references": references,
                "context_length": len(context),
                "model": self.config.model_name,
                "doc_count": len(documents),
            }

            logger.info(f"回答生成完成，长度: {len(answer)} 字符")
            return result

        except Exception as e:
            logger.error(f"大模型调用失败: {e}")
            return {
                "answer": f"抱歉，回答生成过程中出现错误: {str(e)}",
                "references": [],
                "context_length": len(context),
                "model": self.config.model_name,
                "doc_count": len(documents),
                "error": str(e),
            }

    def check_api_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.warning(f"API 不可用: {e}")
            return False
