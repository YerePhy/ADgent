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
#clear-upload button {
    position: absolute;
    top: 6px;
    right: 6px;
    z-index: 20;
    min-width: 30px !important;
    width: 30px !important;
    height: 30px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    font-size: 16px !important;
    line-height: 1 !important;
}
"""

# Move the clear button inside the file component's DOM so it sits visually
# within the drop zone. MutationObserver handles the initially-hidden chat panel.
_EMBED_CLEAR_BTN_JS = """
() => {
    const embed = () => {
        const upload = document.querySelector('#nifti-upload');
        const btn = document.querySelector('#clear-upload button');
        if (!upload || !btn || upload.contains(btn)) return false;
        upload.style.position = 'relative';
        upload.appendChild(btn);
        return true;
    };
    const obs = new MutationObserver(() => { if (embed()) obs.disconnect(); });
    obs.observe(document.body, { childList: true, subtree: true });
    embed();
}
"""

with gr.Blocks(title="ADgent", css=_CSS) as demo:
    thread_id_state = gr.State(value=None)

    with gr.Column(visible=True) as login_panel:
        gr.Markdown("## ADgent — Alzheimer Imaging Assistant")
        username_input = gr.Textbox(label="Username", placeholder="Enter your name to start")
        login_btn = gr.Button("Start session")

    with gr.Column(visible=False) as chat_panel:
        chatbot = gr.Chatbot(label="ADgent")
        with gr.Group():
            file_upload = gr.File(
                label="Upload NIfTI (.nii / .nii.gz)",
                file_types=[".nii", ".gz"],  # .gz lets .nii.gz pass browser validation
                file_count="single",
                elem_id="nifti-upload",
            )
            clear_btn = gr.Button("↺", elem_id="clear-upload", variant="secondary", size="sm")
        msg_input = gr.Textbox(label="Message", placeholder="Ask something...", show_label=False)
        send_btn = gr.Button("Send")

    demo.load(js=_EMBED_CLEAR_BTN_JS)

    login_btn.click(
        fn=_login,
        inputs=[username_input],
        outputs=[thread_id_state, login_panel, chat_panel],
    ).then(
        fn=_load_history,
        inputs=[thread_id_state],
        outputs=[chatbot],
    )

    file_upload.change(
        fn=_upload_file,
        inputs=[file_upload, chatbot, thread_id_state],
        outputs=[chatbot, file_upload],
    )

    clear_btn.click(fn=lambda: None, outputs=[file_upload])

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
