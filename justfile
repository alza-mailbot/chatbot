# List available recipes
default:
    @just --list

# Install dependencies
install:
    uv sync

# Run the development server with auto-reload
run:
    uv run uvicorn chatbot.main:app --host 0.0.0.0 --port 8080 --reload

# Run tests (pass extra args to pytest)
test *ARGS:
    uv run pytest {{ARGS}}

# Run unit tests
test-unit:
    uv run pytest tests/unit

# Run integration tests
test-integration:
    uv run pytest tests/integration

# Format code
format:
    uv run ruff format .

# Lint code
lint:
    uv run ruff check .

# Lint code and auto-fix
lint-fix:
    uv run ruff check --fix .

# Type-check code
type-check:
    uv run ty check

# Lint + type-check
check: lint type-check

# Format + lint with auto-fix
fix: format lint-fix

# Install pre-commit hooks
install-hooks:
    uv run pre-commit install

# Run pre-commit checks on all files
pre-commit:
    uv run pre-commit run --all-files

# Remove caches
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache .ruff_cache

# Add a dependency
add PACKAGE:
    uv add {{PACKAGE}}

# Add a dev dependency
add-dev PACKAGE:
    uv add --dev {{PACKAGE}}

# Remove a dependency
remove PACKAGE:
    uv remove {{PACKAGE}}

# Sync environment exactly to the lockfile
sync-frozen:
    uv sync --frozen

# Upgrade dependencies and sync
update:
    uv sync --upgrade

# Build the Docker image
docker-build:
    docker build -t chatbot .

# Run the container locally (mounts gcloud ADC for Vertex AI access)
docker-run:
    docker run --rm -p 8080:8080 --env-file .env \
      -v ~/.config/gcloud/application_default_credentials.json:/adc.json:ro \
      -e GOOGLE_APPLICATION_CREDENTIALS=/adc.json \
      chatbot
