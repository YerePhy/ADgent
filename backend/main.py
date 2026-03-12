import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from backend.agent import build_agent
from backend.chat_model_factory import create_chat_model
from backend.dataloaders import LocalDataLoader
from backend.prompts import load_system_message

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

data_loader = LocalDataLoader(
    data_dir=os.getenv("DATA_DIR", "./data"),
    registry=os.getenv("REGISTRY", "./registry.json"),
)
chat = create_chat_model()
agent = build_agent(chat)
system_message = load_system_message(data_loader)

messages: list = [system_message]
llm_calls: int = 0

while True:
    user_input = input("User: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    state = agent.invoke({
        "messages": messages + [HumanMessage(content=user_input)],
        "llm_calls": llm_calls,
    })
    messages = state["messages"]
    llm_calls = state["llm_calls"]
    print(f"AI: {messages[-1].content}")
