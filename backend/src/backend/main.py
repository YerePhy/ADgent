"""Simple LangChain CLI chat agent.

Run with:  python -m backend.main
"""

import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from backend.chat_model_factory import create_chat_model
from backend.prompts import load_system_message
from backend.agent import build_agent

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

chat = create_chat_model()
agent = build_agent(chat)
system_message = load_system_message()

messages: list = [system_message]
llm_calls = 0

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