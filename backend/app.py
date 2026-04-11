import functools
import logging
import logging.config
import os

import gradio as gr
import yaml
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()

from backend.app_context import build_app_context, build_infrastructure
from backend.chat_handlers import load_history, login, respond, upload_file
from backend.config import load_config

_logging_cfg = here("logging.yaml")
_logging_cfg.parent.joinpath("logs").mkdir(exist_ok=True)
with open(_logging_cfg) as _f:
    logging.config.dictConfig(yaml.safe_load(_f))

logger = logging.getLogger(__name__)

_config = load_config()
ctx = build_app_context(_config, build_infrastructure(_config))

_login = functools.partial(login, ctx=ctx)
_load_history = functools.partial(load_history, ctx=ctx)
_respond = functools.partial(respond, ctx=ctx)
_upload_file = functools.partial(upload_file, ctx=ctx)

_CSS = """
#upload-btn, #send-btn {
    min-width: 44px !important;
    max-width: 44px !important;
    align-self: stretch !important;
}
#upload-btn button, #send-btn button {
    width: 100% !important;
    height: 100% !important;
    min-width: 44px !important;
    padding: 0 !important;
    border-radius: 8px !important;
    font-size: 26px !important;
    line-height: 1 !important;
}
"""

with gr.Blocks(title="ADgent", css=_CSS) as demo:
    thread_id_state = gr.State(value=None)

    with gr.Column(visible=True) as login_panel:
        gr.Markdown("## ADgent — Alzheimer Imaging Assistant")
        username_input = gr.Textbox(label="Username", placeholder="Enter your name to start")
        login_btn = gr.Button("Start session")

    with gr.Column(visible=False) as chat_panel:
        chatbot = gr.Chatbot(label="ADgent", height=600)
        with gr.Row(equal_height=True):
            msg_input = gr.Textbox(
                placeholder="Ask something...",
                show_label=False,
                scale=8,
                elem_id="msg-input",
            )
            upload_btn = gr.UploadButton(
                "📎",
                file_types=[".nii", ".gz"],
                file_count="single",
                size="sm",
                scale=1,
                elem_id="upload-btn",
            )
            send_btn = gr.Button("↑", scale=1, elem_id="send-btn")

    login_btn.click(
        fn=_login,
        inputs=[username_input],
        outputs=[thread_id_state, login_panel, chat_panel],
    ).then(
        fn=_load_history,
        inputs=[thread_id_state],
        outputs=[chatbot],
    )

    upload_btn.upload(
        fn=_upload_file,
        inputs=[upload_btn, chatbot, thread_id_state],
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
