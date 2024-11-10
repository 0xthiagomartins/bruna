import os
from sqlmodel import SQLModel
from .models import *
from . import controllers, engine
from .. import settings

if settings.ENVIRONMENT in {"production", "development", "stage"}:
    print("dev", flush=True)
    SQLModel.metadata.create_all(engine.get())
    # seed_database()
elif settings.ENVIRONMENT == "test":
    print("test", flush=True)
    SQLModel.metadata.create_all(engine.get())
    # seed_database()
else:
    raise ValueError(f"Unsupported environment: {settings.ENVIRONMENT}")
