"""Unit tests for backend.config — dataclasses and YAML loading."""

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.config import (
    AgentConfig,
    Config,
    EmbeddingsConfig,
    FileStoreConfig,
    LLMConfig,
    UploadStoreConfig,
    load_config,
)

_MINIMAL_YAML = """\
env: test
data_dir: ./data
registry: ./registry.json
llm:
  provider: anthropic
  model_repo_id: claude-test
embeddings:
  provider: huggingface
  model: some-model
agent:
  max_llm_calls: 5
  tools:
    - query_table
file_store:
  backend: local
  base_dir: ./data/uploads
upload_store:
  backend: sqlite
system_prompt: "You are a test assistant."
"""


def test_llm_config_stores_fields():
    cfg = LLMConfig(provider="anthropic", model_repo_id="claude-haiku")
    assert cfg.provider == "anthropic"
    assert cfg.model_repo_id == "claude-haiku"


def test_embeddings_config_stores_fields():
    cfg = EmbeddingsConfig(provider="huggingface", model="bert-base")
    assert cfg.provider == "huggingface"
    assert cfg.model == "bert-base"


def test_agent_config_stores_fields():
    cfg = AgentConfig(max_llm_calls=10, tools=["query_table", "search_documents"])
    assert cfg.max_llm_calls == 10
    assert cfg.tools == ["query_table", "search_documents"]


def test_file_store_config_defaults():
    cfg = FileStoreConfig(backend="local")
    assert cfg.base_dir == "./data/uploads"
    assert cfg.bucket is None


def test_upload_store_config_defaults():
    cfg = UploadStoreConfig()
    assert cfg.backend == "sqlite"
    assert cfg.dsn is None


def test_load_config_parses_yaml(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_MINIMAL_YAML)

    with patch("backend.config.here", return_value=config_file):
        cfg = load_config()

    assert cfg.env == "test"
    assert cfg.data_dir == Path("./data")
    assert cfg.registry == Path("./registry.json")
    assert cfg.llm == LLMConfig(provider="anthropic", model_repo_id="claude-test")
    assert cfg.embeddings == EmbeddingsConfig(provider="huggingface", model="some-model")
    assert cfg.agent == AgentConfig(max_llm_calls=5, tools=["query_table"])
    assert cfg.file_store == FileStoreConfig(backend="local", base_dir="./data/uploads")
    assert cfg.upload_store == UploadStoreConfig(backend="sqlite")
    assert cfg.system_prompt == "You are a test assistant."


def test_load_config_missing_file_raises(tmp_path: Path):
    with patch("backend.config.here", return_value=tmp_path / "nonexistent.yaml"):
        with pytest.raises(FileNotFoundError):
            load_config()


def test_load_config_uses_defaults_for_optional_sections(tmp_path: Path):
    yaml_without_stores = """\
env: dev
data_dir: ./data
registry: ./registry.json
llm:
  provider: anthropic
  model_repo_id: claude-test
embeddings:
  provider: huggingface
  model: some-model
agent:
  max_llm_calls: 3
  tools: []
system_prompt: "prompt"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_without_stores)

    with patch("backend.config.here", return_value=config_file):
        cfg = load_config()

    assert cfg.file_store == FileStoreConfig()
    assert cfg.upload_store == UploadStoreConfig()
