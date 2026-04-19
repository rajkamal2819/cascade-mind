# server/app.py — compatibility shim for openenv.yaml
# The real implementation lives in cascade_mind/server/app.py.
# This file exists so the openenv-base image can find `server.app:app`
# via openenv.yaml while all logic stays in the cascade_mind package.
from cascade_mind.server.app import app, main  # noqa: F401
