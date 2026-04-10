import functools
import logging
import logging.config
import os

import gradio as gr
import yaml
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()

from backend.app_context import build_app_context
from backend.chat_handlers import load_history, login, respond
from backend.config import load_config

_logging_cfg = here("logging.yaml")
_logging_cfg.parent.joinpath("logs").mkdir(exist_ok=True)
with open(_logging_cfg) as _f:
    logging.config.dictConfig(yaml.safe_load(_f))

logger = logging.getLogger(__name__)

ctx = build_app_context(load_config())

_login = functools.partial(login, ctx=ctx)
_load_history = functools.partial(load_history, ctx=ctx)
_respond = functools.partial(respond, ctx=ctx)

with gr.Blocks(title="ADgent") as demo:
    thread_id_state = gr.State(value=None)

    with gr.Column(visible=True) as login_panel:
        gr.Markdown("## ADgent — Alzheimer Imaging Assistant")
        username_input = gr.Textbox(label="Username", placeholder="Enter your name to start")
        login_btn = gr.Button("Start session")

    with gr.Column(visible=False) as chat_panel:
        chatbot = gr.Chatbot(label="ADgent")
        msg_input = gr.Textbox(label="Message", placeholder="Ask something...", show_label=False)
        send_btn = gr.Button("Send")

    login_btn.click(
        fn=_login,
        inputs=[username_input],
        outputs=[thread_id_state, login_panel, chat_panel],
    ).then(
        fn=_load_history,
        inputs=[thread_id_state],
        outputs=[chatbot],
    )

    send_btn.click(
        fn=_respond,
        inputs=[msg_input, chatbot, thread_id_state],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=_respond,
        inputs=[msg_input, chatbot, thread_id_state],
        outputs=[chatbot, msg_input],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )
