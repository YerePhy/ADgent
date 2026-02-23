"""Gradio chat UI backed by the FastAPI backend."""

import gradio as gr
import httpx

BACKEND_URL = "http://backend:8000"


def chat(message: str, history: list[list[str]]) -> str:
    """Send a message to the backend and return the response."""
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/chat",
            json={"message": message},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json().get("reply", "")
    except httpx.HTTPError as exc:
        return f"Error contacting backend: {exc}"


demo = gr.ChatInterface(
    fn=chat,
    title="LangChain Chat",
    description="Chat powered by LangChain + FastAPI",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
