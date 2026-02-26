import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.api.app import app as api_app

# ─── Configuration ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf_unified_server")

# ─── Main HF Application ──────────────────────────────────────────────────
# We create a parent app that will host both the API and the static frontends
app = FastAPI(title="ClearAudit Unified Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Mount API ────────────────────────────────────────────────────────────
# All /api/ routes go to the existing FastAPI app from src/api/app.py
app.mount("/api", api_app)

# ─── Mount Frontends ──────────────────────────────────────────────────────
# Note: Order matters for catch-all routing.
# Specific routes first, then the root landing page.

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Technical Portfolio
tech_path = os.path.join(ROOT_DIR, "frontend", "technical_page")
if os.path.exists(tech_path):
    app.mount("/technical", StaticFiles(directory=tech_path, html=True), name="technical")
    logger.info("Mounted /technical to %s", tech_path)

# 2. Service Portal
service_path = os.path.join(ROOT_DIR, "frontend", "service_page")
if os.path.exists(service_path):
    app.mount("/service", StaticFiles(directory=service_path, html=True), name="service")
    logger.info("Mounted /service to %s", service_path)

# 3. Main Landing Page (Root)
main_path = os.path.join(ROOT_DIR, "frontend", "main_landing_page")
if os.path.exists(main_path):
    app.mount("/", StaticFiles(directory=main_path, html=True), name="main")
    logger.info("Mounted / (root) to %s", main_path)

@app.get("/health")
async def health():
    return {"status": "online", "message": "ClearAudit Unified Server is running"}

if __name__ == "__main__":
    import uvicorn
    # HF Spaces expects port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
