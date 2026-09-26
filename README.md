# Chatbot

> AI email chatbot service for the **alza-mailbot** system. Given an email (subject, body, attachments, thread history), it generates a context-aware reply using Gemini on Vertex AI.

It works alongside the `email-processor` service, which receives Gmail push notifications and calls this service to generate replies.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages Python 3.14 automatically)
- [just](https://github.com/casey/just) task runner
- gcloud CLI with Application Default Credentials (`gcloud auth application-default login`) for Vertex AI access

## Setup

1. Install dependencies:
   ```bash
   just install
   ```
   *Done when:* `uv run python -c "import chatbot"` prints no error.
2. Create your local configuration:
   ```bash
   cp .env.sample .env
   ```
3. Install git hooks:
   ```bash
   just install-hooks
   ```
   *Done when:* `just pre-commit` passes on all files.

## Run

```bash
just run
```

*Done when:* `curl localhost:8080/healthz` returns `{"status":"ok"}`.

## Development

| Command | Description |
| --- | --- |
| `just` | List all recipes |
| `just run` | Start the dev server (auto-reload, port 8080) |
| `just test` | Run all tests (`just test-unit`, `just test-integration` for subsets) |
| `just check` | Lint + type-check |
| `just fix` | Format + auto-fix lint issues |
| `just pre-commit` | Run all pre-commit checks manually |

Every commit runs ruff, ty and the full pytest suite via pre-commit hooks.

### Testing conventions

- Tests mirror the source tree and are split into `tests/unit/` and `tests/integration/`.
- TDD: tests are written before the implementation, covering both positive and negative cases.
