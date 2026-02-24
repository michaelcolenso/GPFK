from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.config import get_settings
from src.db.models import Base


def get_engine():
    settings = get_settings()
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(settings.database_url, connect_args=connect_args)


def init_db():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    return engine


def get_session_factory():
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


SessionLocal = get_session_factory()


def get_db():
    """FastAPI dependency."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
