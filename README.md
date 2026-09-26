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

## API (contract v1, frozen)

### `POST /v1/chat`

Generates a reply to an email. Accepts `multipart/form-data`:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `body` | text | yes | Plain-text email body |
| `subject` | text | no | Email subject line |
| `thread` | text | no | Prior thread messages as a JSON array of `{"role": "user"\|"assistant", "text": "..."}`, oldest first |
| `files` | file(s) | no | Attachments: PDF, JPEG, PNG, MP3 or WAV, up to 15 MiB each |

```bash
curl -s localhost:8080/v1/chat \
  -F subject="Re: Warranty question" \
  -F body="And does the warranty cover the battery?" \
  -F thread='[{"role":"user","text":"How long is the warranty?"},{"role":"assistant","text":"Two years."}]' \
  -F files=@invoice.pdf
```

The service is stateless: thread history is supplied by the caller on every request. This contract is guarded by `TestContractV1` in the integration suite.

Response: `{"reply": "..."}`. Errors: `422` invalid input or unsupported/empty attachment (the `detail` message names the file and the supported types), `413` attachment over the size limit, `502` LLM failure, `500` unexpected error.

An unsupported attachment rejects the whole request rather than being silently ignored, so the caller can tell the sender which file could not be processed.

## Development

| Command | Description |
| --- | --- |
| `just` | List all recipes |
| `just run` | Start the dev server (auto-reload, port 8080) |
| `just test` | Run all tests (`just test-unit`, `just test-integration` for subsets) |
| `uv run pytest -m live` | Run live tests against real Vertex AI (needs ADC + `.env`) |
| `just check` | Lint + type-check |
| `just fix` | Format + auto-fix lint issues |
| `just pre-commit` | Run all pre-commit checks manually |

Every commit runs ruff, ty and the full pytest suite via pre-commit hooks.

### Testing conventions

- Tests mirror the source tree and are split into `tests/unit/` and `tests/integration/`.
- TDD: tests are written before the implementation, covering both positive and negative cases.
