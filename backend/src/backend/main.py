import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load .env variables (HUGGINGFACEHUB_API_TOKEN, HF_MODEL_REPO_ID, etc.)
load_dotenv()

app = FastAPI()

# 1. Create the underlying endpoint (handles API connection)
_endpoint = HuggingFaceEndpoint(
    repo_id=os.getenv("HF_MODEL_REPO_ID"),
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
)

# 2. Wrap it with ChatHuggingFace (handles chat message formatting)
llm = ChatHuggingFace(llm=_endpoint)

# System message — invisible instructions that shape the AI's behaviour.
SYSTEM_MSG = SystemMessage(content=(
    "You are a neuroradiology assistant specialized in PET Tau imaging. "
    "You help clinicians interpret SUVr values and compare them against "
    "normative reference data. Be concise and cite specific numbers. "
    "Flag any regions above the normative threshold of SUVr > 1.20. "
    "Never provide diagnoses — only data interpretation."
))


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/chat/default")
async def default_chat():
    """Test endpoint — sends a hardcoded question with fake analysis context."""

    # HumanMessage — this is what the user "says". We prepend context
    # (analysis results) that the user didn't type but the LLM needs.
    user_msg = HumanMessage(content=(
        "Here are the analysis results for this patient:\n"
        "- Left temporal SUVr: 1.42\n"
        "- Right temporal SUVr: 1.35\n"
        "- Left parietal SUVr: 1.38\n"
        "- Global cortical SUVr: 1.28\n\n"
        "Reference cohort (healthy controls, n=120):\n"
        "- Temporal mean SUVr: 1.05 (SD 0.08)\n"
        "- Parietal mean SUVr: 1.02 (SD 0.07)\n\n"
        "Which regions are above the normative threshold and how do they "
        "compare to the reference cohort?"
    ))

    # Send both messages to the LLM. It sees [system, user] and replies.
    response = llm.invoke([SYSTEM_MSG, user_msg])

    return {"reply": response}