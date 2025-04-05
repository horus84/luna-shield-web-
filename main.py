# main.py
from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import os
import uuid
import shutil
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any # More specific typing
import logging

# Initialize FastAPI app
app = FastAPI(title="Luna Shield Deepfake Detector")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define base directory ---
BASE_DIR = Path(__file__).resolve().parent

# Mount static files (CSS, JS, images, etc.) relative to BASE_DIR
# Ensure the 'static' directory exists at the same level as main.py
static_dir = BASE_DIR / "static"
if not static_dir.is_dir():
    logger.warning(f"Static directory not found at {static_dir}. Creating it.")
    static_dir.mkdir(parents=True, exist_ok=True) # Create if doesn't exist
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates for HTML rendering relative to BASE_DIR
# Ensure the 'templates' directory exists at the same level as main.py
templates_dir = BASE_DIR / "templates"
if not templates_dir.is_dir():
     logger.error(f"Templates directory not found at {templates_dir}. HTML rendering will fail.")
     # Consider exiting if templates are critical: sys.exit(1)
templates = Jinja2Templates(directory=templates_dir)

# Configuration
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results" # Not actively used for storage in this version
MODEL_PATH = BASE_DIR / "best_model.pth" # Assumes model is in the same directory
IMG_SIZE = 224
FRAMES_TO_ANALYZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# --- Model Definition and Loading ---
logger.info(f"Using device: {DEVICE}")
model = None # Initialize model as None

try:
    # Define model architecture
    model_instance = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if hasattr(models, 'EfficientNet_B0_Weights') else True)
    num_ftrs = model_instance.classifier[1].in_features
    model_instance.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 2) # 0: Real, 1: Fake
    )
    logger.info("Model architecture defined (EfficientNet-B0).")

    # Load trained weights
    if not MODEL_PATH.is_file():
         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure 'best_model.pth' exists.")

    model_instance.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
    logger.info(f"Model weights loaded successfully from {MODEL_PATH}.")

    # Set model to evaluation mode and move to device
    model = model_instance.to(DEVICE)
    model.eval()
    logger.info("Model set to evaluation mode and ready.")

except FileNotFoundError as e:
    logger.error(f"CRITICAL: {e}. Analysis endpoint will be unavailable.")
except Exception as e:
    logger.exception(f"CRITICAL: An unexpected error occurred during model loading: {e}. Analysis endpoint will be unavailable.")
    # Depending on requirements, you might want the app to exit here.
    # import sys
    # sys.exit(1)


# --- Define Transforms ---
data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
logger.info("Data transforms defined.")


# --- Analysis Function ---
def analyze_video(video_path: Path) -> Optional[Dict[str, Any]]:
    """Analyzes video, returns results dict or None on critical error."""
    if model is None:
        logger.error("Model is not loaded, cannot analyze video.")
        # Raise an exception to be caught by the endpoint handler
        raise RuntimeError("Model not available for analysis.")

    logger.info(f"Starting analysis for video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path)) # cv2 needs string path
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return None # Indicate failure to open

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames reported by OpenCV: {total_frames}")

    # Handle videos with zero or very few frames
    if total_frames < 1:
        logger.warning(f"Video {video_path.name} has zero or negative frames reported.")
        cap.release()
        # Return a specific structure indicating no frames processed
        return {
             "file_name": video_path.name,
             "total_frames": total_frames, # Report what OpenCV gave
             "frames_analyzed": 0,
             "real_frames": 0,
             "fake_frames": 0,
             "verdict": "UNKNOWN",
             "confidence": 0.0,
             "average_confidence": 0.0,
             "error_message": "Video contains no processable frames."
         }

    # Select frame indices evenly spaced
    num_frames_to_select = min(FRAMES_TO_ANALYZE, total_frames)
    # Ensure indices are within valid range [0, total_frames - 1]
    if total_frames > 1 :
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames_to_select, dtype=int)
    elif total_frames == 1:
        frame_indices = [0] # Only one frame to analyze
    else: # Should be caught above, but defensive check
        frame_indices = []

    logger.info(f"Attempting to analyze {len(frame_indices)} frames at indices: {frame_indices.tolist()}")

    predictions = []
    confidence_scores = []
    processed_frame_count = 0

    try:
        with torch.no_grad():
            for i, frame_index in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    processed_frame_count += 1
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        img_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)
                        output = model(img_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        pred = torch.argmax(probs, dim=1).item() # 0=Real, 1=Fake
                        confidence = probs[0, pred].item() * 100

                        predictions.append(pred)
                        confidence_scores.append(confidence)
                        # logger.debug(f"Frame {frame_index}: Prediction={pred}, Confidence={confidence:.2f}%")
                    except Exception as frame_proc_e:
                        logger.warning(f"Could not process frame index {frame_index} from {video_path.name}: {frame_proc_e}")
                        # Optionally skip this frame or handle differently
                else:
                    # This can happen if frame_index is invalid or near end of video
                    logger.warning(f"Could not read frame index {frame_index} from {video_path.name}. ret={ret}")

    except Exception as e:
        logger.exception(f"Error during model inference for {video_path.name}: {e}")
        # Raise an exception to be caught by the endpoint handler
        raise RuntimeError(f"Model inference failed: {e}") from e
    finally:
        cap.release()

    logger.info(f"Successfully processed {processed_frame_count} frames out of {len(frame_indices)} selected.")

    if not predictions: # No frames were successfully processed
        logger.warning(f"No frames were successfully processed for {video_path.name}")
        return {
             "file_name": video_path.name,
             "total_frames": total_frames,
             "frames_analyzed": 0,
             "real_frames": 0,
             "fake_frames": 0,
             "verdict": "UNKNOWN",
             "confidence": 0.0,
             "average_confidence": 0.0,
             "error_message": "Could not process any frames from the video."
         }

    # Calculate overall results
    real_count = predictions.count(0)
    fake_count = predictions.count(1)
    total_analyzed = len(predictions) # Use actual number processed
    avg_confidence = sum(confidence_scores) / total_analyzed if total_analyzed > 0 else 0

    # Determine overall verdict (0 = real, 1 = fake)
    overall_verdict_idx = 0 if real_count >= fake_count else 1
    overall_verdict_str = "REAL" if overall_verdict_idx == 0 else "FAKE"

    # Calculate confidence in the final verdict (proportion of frames matching)
    verdict_confidence = 0.0
    if total_analyzed > 0:
         if overall_verdict_idx == 0: # Verdict is REAL
             verdict_confidence = (real_count / total_analyzed) * 100
         else: # Verdict is FAKE
             verdict_confidence = (fake_count / total_analyzed) * 100

    logger.info(f"Analysis complete for {video_path.name}. Verdict: {overall_verdict_str}, Confidence: {verdict_confidence:.2f}%, Avg Frame Confidence: {avg_confidence:.2f}%")

    # Return comprehensive results, matching keys used by JS
    return {
        "file_name": video_path.name,
        "total_frames": total_frames, # Total frames reported by video
        "frames_analyzed": total_analyzed, # Actual number analyzed
        "real_frames": real_count,
        "fake_frames": fake_count,
        "verdict": overall_verdict_str,
        "confidence": round(verdict_confidence, 2), # Confidence in the final verdict
        "average_confidence": round(avg_confidence, 2), # Avg confidence of individual frames
        # Optionally include frame-level details if needed by frontend later
        # "frame_predictions": predictions,
        # "frame_confidences": [round(c, 2) for c in confidence_scores]
    }

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main index.html page."""
    logger.info("Serving index.html")
    try:
        # Ensure template file exists
        template_path = templates_dir / "index.html"
        if not template_path.is_file():
            logger.error("index.html template not found!")
            raise HTTPException(status_code=500, detail="Server configuration error: Missing main page template.")
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
         logger.exception("Error serving index.html")
         # Provide a generic error response
         raise HTTPException(status_code=500, detail="Internal Server Error retrieving page.")


# --- Serve other HTML pages (if they exist) ---
# You might need similar endpoints for analytics.html, about.html, contact.html
# Example for about.html:
@app.get("/about.html", response_class=HTMLResponse)
async def read_about(request: Request):
    """Serves the about.html page."""
    logger.info("Serving about.html")
    try:
        template_path = templates_dir / "about.html"
        if not template_path.is_file():
             logger.warning("about.html template not found!")
             # Return 404 if page doesn't exist
             raise HTTPException(status_code=404, detail="About page not found.")
        return templates.TemplateResponse("about.html", {"request": request})
    except HTTPException:
        raise # Re-raise HTTPException (like 404)
    except Exception as e:
        logger.exception("Error serving about.html")
        raise HTTPException(status_code=500, detail="Internal Server Error retrieving page.")

# Add similar endpoints for analytics.html and contact.html if needed


@app.post("/analyze", response_class=JSONResponse)
async def analyze_uploaded_video(file: UploadFile = File(...)):
    """Handles video upload, analysis, and returns JSON results."""
    if model is None:
        logger.error("Analysis request received, but model is not loaded.")
        # Use 503 Service Unavailable
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Analysis service is temporarily unavailable (Model not loaded). Please try again later or contact support."}
        )

    # Basic validation on server side as well
    file_ext = Path(file.filename).suffix.lower()
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if file_ext not in allowed_extensions:
         logger.warning(f"Unsupported file type rejected: {file.filename} (type: {file.content_type})")
         return JSONResponse(
             status_code=400, # Bad Request
             content={"success": False, "error": f"Unsupported file type ({file_ext}). Please upload MP4, AVI, MOV, or MKV."}
         )

    # Generate unique filename
    unique_id = uuid.uuid4()
    upload_path = UPLOAD_DIR / f"{unique_id}{file_ext}"
    logger.info(f"Receiving file: {file.filename} ({file.content_type}), saving to: {upload_path}")

    try:
        # Save uploaded file securely
        # Consider adding size limit check here too (FastAPI might have ways)
        start_time = datetime.now()
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        save_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"File saved successfully: {upload_path} (took {save_duration:.2f}s)")

        # Analyze the video
        start_analysis_time = datetime.now()
        results = analyze_video(upload_path)
        analysis_duration = (datetime.now() - start_analysis_time).total_seconds()
        logger.info(f"Analysis completed (took {analysis_duration:.2f}s)")


        if results is None:
             # Generic analysis failure (specifics logged in analyze_video)
             logger.error(f"Analysis function returned None for {upload_path.name}")
             return JSONResponse(
                 status_code=500,
                 content={"success": False, "error": "Video analysis failed due to an internal error during processing."}
             )

        # Check if analysis indicated specific errors (like no frames)
        if "error_message" in results:
             logger.warning(f"Analysis for {upload_path.name} completed with message: {results['error_message']}")
             # Return success=True, but include the error message in results
             return JSONResponse(content={"success": True, "results": results})


        # Prepare successful response
        response_data = {
            "success": True,
            "results": results # contains file_name, verdict, confidence etc.
        }
        return JSONResponse(content=response_data)

    except HTTPException:
         raise # Re-raise HTTPExceptions (like from read_root)
    except RuntimeError as e: # Catch specific errors raised from analysis
        logger.error(f"Runtime error during analysis for {file.filename}: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Analysis failed: {e}"}
        )
    except Exception as e:
        logger.exception(f"Unexpected error during upload/analysis for {file.filename}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "An unexpected internal server error occurred."}
        )
    finally:
        # Clean up uploaded file
        if upload_path.exists():
             try:
                 os.remove(upload_path)
                 logger.info(f"Cleaned up uploaded file: {upload_path}")
             except OSError as e:
                 logger.error(f"Error removing uploaded file {upload_path}: {e}")
        # Ensure file object is closed (FastAPI usually handles this with UploadFile context)
        await file.close()


# Placeholder for results retrieval (not implemented)
@app.get("/results/{result_id}")
async def get_analysis_results(result_id: str):
    logger.warning(f"Request for non-existent result ID: {result_id}")
    return JSONResponse(
        status_code=404,
        content={"error": "Result retrieval by ID is not implemented."}
        )

# --- Run the server ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Luna Shield Deepfake Detector API...")
    if model is None:
        logger.critical("MODEL NOT LOADED. ANALYSIS WILL FAIL.")
        # Decide whether to exit or run in a degraded state
        # import sys
        # sys.exit("Exiting due to model loading failure.")
    uvicorn.run(app, host="0.0.0.0", port=8000)