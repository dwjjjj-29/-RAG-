import gradio as gr
from typing import List, Dict, Any

from config.config import AppConfig
from utils.logging import logger


def _format_answer(result: Dict[str, Any]) -> str:
    answer = result.get("answer", "")
    references = result.get("references", [])
    doc_count = result.get("doc_count", 0)
    model = result.get("model", "")

    output = answer + "\n\n"

    if references:
        output += "---\n"
        output += "### 📚 相关文档引用\n\n"
        for i, ref in enumerate(references, 1):
            output += f"**引用 {i}:**\n{ref}\n\n"

    return output


def _format_references(result: Dict[str, Any]) -> str:
    references = result.get("references", [])
    if not references:
        return "暂无相关文档引用"

    output = "## 📄 相关文档片段\n\n"
    for i, ref in enumerate(references, 1):
        output += f"### 文档 {i}\n{ref}\n---\n\n"
    return output


class GradioApp:
    def __init__(
        self,
        qa_chain_callable,
        system=None,
        config: AppConfig = AppConfig(),
    ):
        self.qa_chain = qa_chain_callable
        self.system = system
        self.config = config
        self.history: List[Dict[str, str]] = []

    def _respond(
        self, message: str, history: List[List[str]]
    ) -> str:
        if not message or not message.strip():
            return "请输入您的问题"

        try:
            result = self.qa_chain(
                query=message,
                history=self.history,
            )

            self.history.append({"role": "user", "content": message})
            self.history.append(
                {"role": "assistant", "content": result.get("answer", "")}
            )

            if len(self.history) > self.config.history_length * 2:
                self.history = self.history[-(self.config.history_length * 2):]

            return _format_answer(result)

        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            return f"处理您的请求时出现错误: {str(e)}"

    def _clear_history(self):
        self.history = []
        return [], ""

    def build(self) -> gr.Blocks:
        with gr.Blocks(
            title=self.config.title,
        ) as demo:
            gr.HTML(
                f"""
                <style>
                .gradio-container {{ max-width: 1200px !important; }}
                .header-text {{ text-align: center; margin-bottom: 20px; }}
                .header-text h1 {{ color: #1a73e8; font-size: 2em; margin-bottom: 5px; }}
                .header-text p {{ color: #666; font-size: 1.1em; }}
                </style>
                <div class="header-text">
                    <h1>🏫 {self.config.title}</h1>
                    <p>{self.config.description}</p>
                </div>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="对话",
                        height=500,
                        show_label=True,
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            label="输入您的问题",
                            placeholder="请输入您的问题...",
                            scale=4,
                            container=False,
                        )
                        submit_btn = gr.Button(
                            "发送", variant="primary", scale=1, min_width=80
                        )

                    with gr.Row():
                        clear_btn = gr.Button("清空对话", variant="secondary", size="sm")

                with gr.Column(scale=1):
                    status_box = gr.Markdown(
                        "### 系统状态\n\n🟢 系统就绪\n\n输入问题开始查询",
                        label="系统状态",
                    )
                    ref_box = gr.Markdown(
                        "### 相关文档\n\n提交问题后将显示相关文档引用",
                        label="相关文档",
                    )

            def respond_wrapper(message, chat_history):
                if not message or not message.strip():
                    return "", chat_history

                answer = self._respond(message, chat_history)
                # Gradio 6.0 Chatbot 使用 messages 格式
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": answer})

                ref_text = _format_references(
                    self.system.last_result
                    if self.system and hasattr(self.system, "last_result")
                    else {}
                )

                final_status = "### 系统状态\n\n🟢 回答完成\n\n可继续提问"

                return "", chat_history, final_status, ref_text

            def clear_wrapper():
                self._clear_history()

                status_text = "### 系统状态\n\n🟢 系统就绪\n\n输入问题开始查询"
                ref_text = "### 相关文档\n\n提交问题后将显示相关文档引用"
                return [], "", status_text, ref_text

            msg.submit(
                respond_wrapper,
                [msg, chatbot],
                [msg, chatbot, status_box, ref_box],
            )

            submit_btn.click(
                respond_wrapper,
                [msg, chatbot],
                [msg, chatbot, status_box, ref_box],
            )

            clear_btn.click(
                clear_wrapper,
                None,
                [chatbot, msg, status_box, ref_box],
            )

        return demo

    def launch(self, **kwargs):
        demo = self.build()
        demo.launch(
            theme=self.config.theme,
            **kwargs,
        )
