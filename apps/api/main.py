"""Default Uvicorn entrypoint: ``uvicorn apps.api.main:app``."""

from agentqa.api import create_app

app = create_app()
