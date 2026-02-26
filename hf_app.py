import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from src.api.app import app as api_app, setup_app_services

# ─── Configuration ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf_unified_server")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Lifespan ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  ClearAudit Unified Server — Starting")
    logger.info("=" * 60)
    
    # Explicitly load models and initialize services (SHAP, Drift, etc.)
    # Mounted sub-app lifespans are not triggered in FastAPI automatically.
    try:
        setup_app_services()
        logger.info("Backend services initialized in Unified Server")
    except Exception as e:
        logger.error("CRITICAL: Failed to initialize backend: %s", e)

    yield
    logger.info("Unified Server Shutting Down.")

# ─── Main HF Application ──────────────────────────────────────────────────
app = FastAPI(title="ClearAudit Unified Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Middleware: Trailing Slash Enforcement ───────────────────────────────
@app.middleware("http")
async def ensure_trailing_slash(request: Request, call_next):
    path = request.url.path
    # Ensure /technical and /service end with a slash to keep relative assets aligned
    if path in ["/technical", "/service"]:
        return RedirectResponse(url=f"{path}/", status_code=301)
    return await call_next(request)

# ─── Mount API ────────────────────────────────────────────────────────────
# Mounted at /api. Endpoints like /score become /api/score
app.mount("/api", api_app)

# ─── Mount Frontends ──────────────────────────────────────────────────────

# 1. Technical Portfolio (.../technical/)
tech_path = os.path.join(ROOT_DIR, "frontend", "technical_page")
if os.path.exists(tech_path):
    app.mount("/technical", StaticFiles(directory=tech_path, html=True), name="technical")
    logger.info("Mounted /technical to %s", tech_path)

# 2. Service Portal (.../service/)
service_path = os.path.join(ROOT_DIR, "frontend", "service_page")
if os.path.exists(service_path):
    app.mount("/service", StaticFiles(directory=service_path, html=True), name="service")
    logger.info("Mounted /service to %s", service_path)

# 3. Main Landing Page (Root /)
# Instead of mounting the whole folder at root (which can be greedy), 
# we serve the home page explicitly and mount assets separately.
main_path = os.path.join(ROOT_DIR, "frontend", "main_landing_page")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(main_path, "index.html"))

# Mount assets for the landing page (css, js, images)
for folder in ["css", "js", "img", "assets"]:
    f_path = os.path.join(main_path, folder)
    if os.path.exists(f_path):
        app.mount(f"/{folder}", StaticFiles(directory=f_path), name=f"main_{folder}")

@app.get("/health")
async def health():
    return {"status": "online", "message": "ClearAudit Unified Server is running"}

if __name__ == "__main__":
    import uvicorn
    # HF Spaces expects port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
