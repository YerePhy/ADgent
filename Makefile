.PHONY: help up down build purge rmi ingest run logs lint

IMAGE_NAME   := adgent-agent
MODEL_VOLUME := adgent_model_cache

help: ## Show this help message
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  %-12s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

up: ## Start all services detached — model-init runs first, then agent
	docker compose up -d

down: ## Stop and remove containers
	docker compose down

build: ## Build Docker images without starting services
	docker compose build

purge: ## Stop and remove all project containers and the model cache volume
	@echo "Removing containers for image '$(IMAGE_NAME)'..."
	@docker ps -a --filter "ancestor=$(IMAGE_NAME)" -q | xargs -r docker rm -f
	@echo "Removing model cache volume '$(MODEL_VOLUME)'..."
	@docker volume rm -f $(MODEL_VOLUME) 2>/dev/null || true
	@echo "Done."

rmi: ## Remove all Docker images for this project ($(IMAGE_NAME))
	@echo "Removing images for '$(IMAGE_NAME)'..."
	@docker images --filter "reference=$(IMAGE_NAME)" -q | xargs -r docker rmi -f
	@echo "Done."

ingest: ## Run the ingestion pipeline locally (builds the ChromaDB vectorstore)
	uv run python -m scripts.ingest

run: ## Launch the Gradio app locally via Python (no Docker)
	uv run python -m frontend.app

logs: ## Tail logs from the agent service only
	docker compose logs -f agent

lint: ## Run ruff lint, ruff format check, and mypy (mirrors CI)
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy backend/ scripts/
