"""Test-wide environment setup.

Settings require GCP_PROJECT_ID; tests provide a dummy value up front so the
suite runs without a real GCP project or a local .env file.
"""

import os

os.environ.setdefault("GCP_PROJECT_ID", "test-project")
