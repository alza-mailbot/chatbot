#!/usr/bin/env bash
# Build from source and deploy the chatbot to Cloud Run.
# Access is locked by IAM (--no-allow-unauthenticated): only identities with
# run.invoker on this service can call it; the email-processor deploy grants
# that to its service account.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT_ID="alza-mailbot-509813"
REGION="europe-west3"
SA_CHATBOT="sa-chatbot@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud run deploy chatbot \
  --source . \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --service-account="${SA_CHATBOT}" \
  --no-allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},WEB_SEARCH_ENABLED=true" \
  --set-secrets="BRAVE_API_KEY=brave-api-key:latest" \
  --memory=1Gi \
  --max-instances=3

gcloud run services describe chatbot --project="${PROJECT_ID}" \
  --region="${REGION}" --format="value(status.url)"
