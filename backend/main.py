import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from backend.agent import build_agent  # noqa: E402
from backend.chat_model_factory import create_chat_model  # noqa: E402
from backend.config import load_config  # noqa: E402
from backend.dataloaders import LocalDataLoader  # noqa: E402
from backend.prompts import load_system_message  # noqa: E402
from backend.tools import make_query_table_tool  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

config = load_config()

data_loader = LocalDataLoader(
    data_dir=config.data_dir,
    registry=config.registry,
)
tools = [make_query_table_tool(data_loader)]
chat = create_chat_model(
    provider=config.llm.provider,
    model_repo_id=config.llm.model_repo_id,
)
agent = build_agent(chat, tools, max_llm_calls=config.agent.max_llm_calls)
system_message = load_system_message(data_loader, config.system_prompt)

messages: list = [system_message]

while True:
    user_input = input("User: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    state = agent.invoke({
        "messages": messages + [HumanMessage(content=user_input)],
        "llm_calls": 0,
    })
    messages = state["messages"]
    content = messages[-1].content
    if isinstance(content, list):
        content = "\n".join(block["text"] for block in content if block.get("type") == "text")
    print(f"AI: {content}")
