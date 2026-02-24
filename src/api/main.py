"""
HARBINGER API
Pre-announcement M&A signal detection platform.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.predictions import router as predict_router
from src.api.routes.predictions import router_watchlist
from src.api.routes.signals import router as signals_router
from src.db.session import init_db

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: initializing database")
    init_db()
    logger.info("startup: complete")
    yield
    logger.info("shutdown: complete")


app = FastAPI(
    title="HARBINGER",
    description="Pre-announcement M&A signal detection using public alternative data.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(router_watchlist)
app.include_router(signals_router)


@app.get("/", tags=["health"])
def root():
    return {
        "service": "HARBINGER",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
