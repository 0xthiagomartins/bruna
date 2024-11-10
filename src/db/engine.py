from sqlmodel import create_engine
from sqlmodel.pool import StaticPool
from src import settings

production = create_engine(settings.SQL_URI)
test = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
development = create_engine(settings.SQL_URI, echo=True)


def get():
    match settings.ENVIRONMENT:
        case "production":
            return production
        case "development":
            return development
        case "test":
            return test
        case "stage":
            pass
