"""FastAPI application for defect detection."""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import numpy as np
import cv2
from typing import Optional

from src.vision.inference.onnx_infer import ONNXInferenceEngine
from src.core.config import get_settings
from src.core.logging import setup_logging, get_logger
from src.core.constants import ONNX_DIR
from src.core.exceptions import ModelInferenceError

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Digital Twin Factory - Defect Detection API",
    description="AI-powered defect detection for manufacturing",
    version="0.1.0",
)

# CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine: Optional[ONNXInferenceEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    global inference_engine

    logger.info("Starting up API server...")

    # Load ONNX model
    model_path = ONNX_DIR / "swin_defect.onnx"

    if not model_path.exists():
        logger.warning(f"ONNX model not found at {model_path}")
        logger.warning("Run: python scripts/export_onnx.py")
        logger.warning("API will start but /predict endpoint will fail")
    else:
        try:
            inference_engine = ONNXInferenceEngine(
                model_path=model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                num_classes=2,
                image_size=224,
            )
            logger.info("✓ Inference engine loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load inference engine: {e}")
            inference_engine = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")


def get_inference_engine() -> ONNXInferenceEngine:
    """Dependency to get inference engine."""
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not loaded. Please load ONNX model first.",
        )
    return inference_engine


def decode_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded image file.

    Args:
        file_bytes: Raw image bytes

    Returns:
        Image as numpy array (RGB)

    Raises:
        HTTPException: If image decoding fails
    """
    try:
        # Decode image
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {str(e)}",
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Digital Twin Factory - Defect Detection API",
        "version": "0.1.0",
        "status": "running",
        "model_loaded": inference_engine is not None,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
    }


@app.post("/api/v1/predict")
async def predict(
    file: UploadFile = File(...),
    engine: ONNXInferenceEngine = Depends(get_inference_engine),
):
    """
    Predict defects in uploaded image.

    Args:
        file: Uploaded image file (JPEG, PNG)

    Returns:
        Prediction results
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG/PNG allowed.",
        )

    try:
        # Read and decode image
        contents = await file.read()
        image = decode_image(contents)

        # Run inference
        result = engine.predict(image, return_probabilities=True)

        # Add image info
        result["image_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "shape": image.shape,
        }

        return JSONResponse(content=result)

    except ModelInferenceError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/v1/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    engine: ONNXInferenceEngine = Depends(get_inference_engine),
):
    """
    Predict defects in multiple images.

    Args:
        files: List of uploaded image files

    Returns:
        List of prediction results
    """
    if len(files) > 32:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum 32, got {len(files)}",
        )

    try:
        # Decode all images
        images = []
        filenames = []

        for file in files:
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.content_type}",
                )

            contents = await file.read()
            image = decode_image(contents)
            images.append(image)
            filenames.append(file.filename)

        # Run batch inference
        results = engine.predict_batch(images, return_probabilities=True)

        # Add filenames
        for result, filename in zip(results, filenames):
            result["filename"] = filename

        return JSONResponse(content={"predictions": results})

    except ModelInferenceError as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/v1/stats")
async def get_stats(engine: ONNXInferenceEngine = Depends(get_inference_engine)):
    """Get model statistics."""
    return {
        "model_path": str(engine.model_path),
        "num_classes": engine.num_classes,
        "image_size": engine.image_size,
        "input_name": engine.input_name,
        "output_name": engine.output_name,
    }


@app.get("/api/v1/benchmark")
async def benchmark(
    iterations: int = 100,
    batch_size: int = 1,
    engine: ONNXInferenceEngine = Depends(get_inference_engine),
):
    """
    Benchmark inference performance.

    Args:
        iterations: Number of benchmark iterations
        batch_size: Batch size

    Returns:
        Performance metrics
    """
    if iterations > 1000:
        raise HTTPException(
            status_code=400,
            detail="Too many iterations. Maximum 1000",
        )

    if batch_size > 32:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 32",
        )

    metrics = engine.benchmark(num_iterations=iterations, batch_size=batch_size)

    return JSONResponse(content=metrics)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
