"""Prompt loading utilities."""

import os

from langchain_core.messages import SystemMessage


def load_system_message() -> SystemMessage:
    """Load the system prompt from environment."""
    content = os.getenv("SYSTEM_PROMPT", "")
    return SystemMessage(content=content)
