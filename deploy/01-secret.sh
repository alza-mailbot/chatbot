#!/usr/bin/env bash
# Store the Brave API key in Secret Manager and let only the chatbot's
# service account read it. The key value is read from the local .env and
# never appears in code, git or the command history.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT_ID="alza-mailbot-509813"
SA_CHATBOT="sa-chatbot@${PROJECT_ID}.iam.gserviceaccount.com"

BRAVE_API_KEY="$(grep '^BRAVE_API_KEY=' .env | cut -d= -f2-)"
if [ -z "${BRAVE_API_KEY}" ]; then
  echo "BRAVE_API_KEY not found in .env" >&2
  exit 1
fi

if gcloud secrets describe brave-api-key --project="${PROJECT_ID}" >/dev/null 2>&1; then
  printf '%s' "${BRAVE_API_KEY}" | gcloud secrets versions add brave-api-key \
    --project="${PROJECT_ID}" --data-file=-
else
  printf '%s' "${BRAVE_API_KEY}" | gcloud secrets create brave-api-key \
    --project="${PROJECT_ID}" --replication-policy=automatic --data-file=-
fi
echo "Secret brave-api-key stored"

gcloud secrets add-iam-policy-binding brave-api-key \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${SA_CHATBOT}" \
  --role="roles/secretmanager.secretAccessor" --quiet >/dev/null
echo "sa-chatbot: secretAccessor on brave-api-key granted"
