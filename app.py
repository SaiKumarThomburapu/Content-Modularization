from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Form
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import asyncio
import shutil
import os
import uuid
import logging
import uvicorn
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.pipeline.content_pipeline import (
    process_text_classification,
    process_image_classification_with_ocr,
    process_video_analysis,
)
from src.exceptions import CustomException

# ----------------------------------------------------------------------------- #
#                               Logging Configuration                           #
# ----------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- #
#                       Global executors for parallel jobs                      #
# ----------------------------------------------------------------------------- #
process_executor: ProcessPoolExecutor | None = None
thread_executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan – create thread / process pools at startup,
    close them at shutdown.
    """
    global process_executor, thread_executor

    max_workers = min(4, os.cpu_count() or 1)
    process_executor = ProcessPoolExecutor(max_workers=max_workers)
    thread_executor = ThreadPoolExecutor(max_workers=max_workers * 2)

    logger.info(
        f"Executors started: {max_workers} process workers, {max_workers * 2} thread workers"
    )
    yield

    # --- Shutdown ---
    if process_executor:
        process_executor.shutdown(wait=True)
    if thread_executor:
        thread_executor.shutdown(wait=True)
    logger.info("Executors shut down")


# ----------------------------------------------------------------------------- #
#                                  FastAPI App                                  #
# ----------------------------------------------------------------------------- #
app = FastAPI(title="Content Modularization API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------- #
#                           Helper Async File Utilities                         #
# ----------------------------------------------------------------------------- #
async def save_file_async(file: UploadFile, temp_path: str) -> None:
    """Save an uploaded file asynchronously using thread pool."""
    def _save():
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    await run_in_threadpool(_save)


async def cleanup_file_async(file_path: str) -> None:
    """Delete a file asynchronously using thread pool."""
    def _cleanup():
        if os.path.exists(file_path):
            os.remove(file_path)

    await run_in_threadpool(_cleanup)


# ----------------------------------------------------------------------------- #
#                                    Routes                                     #
# ----------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return "<h1>Content Modularization API – Advanced Processing</h1>"


# 1. TEXT --------------------------------------------------------------------- #
@app.post("/classify-text/")
async def classify_text(text: str = Form(...)):
    """Classify raw text as toxic / non-toxic."""
    try:
        result = await run_in_threadpool(process_text_classification, text)
        return JSONResponse(content=result)
    except CustomException as ce:
        logger.error(f"CustomException: {ce}")
        raise HTTPException(500, str(ce))
    except Exception as ex:
        logger.error(f"Unexpected error: {ex}")
        raise HTTPException(500, f"Unexpected error: {ex}")


# 2. IMAGE + OCR -------------------------------------------------------------- #
@app.post("/classify-image/")
async def classify_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    • Classify the image for NSFW/SFW  
    • Extract any visible text via OCR  
    • Classify the extracted text for toxicity
    """
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"

    try:
        await save_file_async(file, temp_path)
        result = await run_in_threadpool(
            process_image_classification_with_ocr, temp_path
        )

        background_tasks.add_task(cleanup_file_async, temp_path)
        return JSONResponse(content=result)

    except CustomException as ce:
        logger.error(f"CustomException: {ce}")
        await cleanup_file_async(temp_path)
        raise HTTPException(500, str(ce))
    except Exception as ex:
        logger.error(f"Unexpected error: {ex}")
        await cleanup_file_async(temp_path)
        raise HTTPException(500, f"Unexpected error: {ex}")


# 3. VIDEO -------------------------------------------------------------------- #
@app.post("/classify-video/")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    • Extract frames → classify each frame for NSFW/SFW  
    • Extract audio → transcribe with Whisper  
    • Classify transcript for toxicity
    """
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"

    try:
        await save_file_async(file, temp_path)
        result = await run_in_threadpool(process_video_analysis, temp_path)

        background_tasks.add_task(cleanup_file_async, temp_path)
        return JSONResponse(content=result)

    except CustomException as ce:
        logger.error(f"CustomException: {ce}")
        await cleanup_file_async(temp_path)
        raise HTTPException(500, str(ce))
    except Exception as ex:
        logger.error(f"Unexpected error: {ex}")
        await cleanup_file_async(temp_path)
        raise HTTPException(500, f"Unexpected error: {ex}")


# 4. HEALTH ------------------------------------------------------------------- #
@app.get("/health")
async def health_check():
    """Simple health-check endpoint."""
    return {
        "status": "healthy",
        "cpu_count": os.cpu_count(),
        "thread_pool_active": bool(thread_executor),
        "process_pool_active": bool(process_executor),
    }


# ----------------------------------------------------------------------------- #
#                                  Entrypoint                                   #
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

