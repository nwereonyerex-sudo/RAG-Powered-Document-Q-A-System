"""FastAPI entry for CV and job matching."""

from fastapi import FastAPI

from backend.api.routes import router

app = FastAPI(title="RAG-Powered Document Q&A System")
app.include_router(router)
