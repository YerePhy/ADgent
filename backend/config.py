"""Application configuration loaded from config.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pyprojroot import here


@dataclass
class LLMConfig:
    """LLM provider configuration.

    Attributes:
        provider: Backend provider name (e.g. ``"huggingface"``).
        model_repo_id: Model repository identifier.
    """

    provider: str
    model_repo_id: str


@dataclass
class AgentConfig:
    """Agent execution configuration.

    Attributes:
        max_llm_calls: Maximum number of LLM invocations per conversation turn.
    """

    max_llm_calls: int


@dataclass
class Config:
    """Root application configuration.

    Attributes:
        env: Deployment environment (e.g. ``"development"``).
        data_dir: Path to the root data directory.
        registry: Path to the JSON asset registry file.
        llm: LLM provider settings.
        agent: Agent execution settings.
        system_prompt: Base behavioural prompt for the agent.
    """

    env: str
    data_dir: Path
    registry: Path
    llm: LLMConfig
    agent: AgentConfig
    system_prompt: str


def load_config() -> Config:
    """Load application configuration from ``config.yaml`` at the project root.

    Uses ``pyprojroot`` to locate the project root regardless of the working
    directory from which the application is launched.

    Returns:
        A fully populated ``Config`` instance.

    Raises:
        FileNotFoundError: If ``config.yaml`` is not found at the project root.
    """
    path: Path = here("config.yaml")
    with open(path) as f:
        raw: dict = yaml.safe_load(f)

    return Config(
        env=raw["env"],
        data_dir=Path(raw["data_dir"]),
        registry=Path(raw["registry"]),
        llm=LLMConfig(**raw["llm"]),
        agent=AgentConfig(**raw["agent"]),
        system_prompt=raw["system_prompt"],
    )
