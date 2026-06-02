import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

_DEFAULT_DATABASE_URL = "postgresql://axuser:axpass@localhost:5432/axis"

DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL)


def _new_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=10)


engine = _new_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def reconfigure_from_env() -> None:
    """Rebind SessionLocal when DATABASE_URL is loaded after module import."""
    global DATABASE_URL, engine

    next_url = os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL)
    if next_url == DATABASE_URL:
        return
    engine.dispose()
    DATABASE_URL = next_url
    engine = _new_engine(DATABASE_URL)
    SessionLocal.configure(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
