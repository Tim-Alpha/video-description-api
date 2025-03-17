from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import cv2
import numpy as np
import pytesseract

app = FastAPI()

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    return {"filename": file.filename, "file_path": file_path}

@app.get("/extract_text/")
def extract_text_from_video(video_filename: str):
    file_path = os.path.join(UPLOAD_DIR, video_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    cap = cv2.VideoCapture(file_path)
    text_data = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 30 == 0:  # Process every 30th frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            if text.strip():
                text_data.append(text.strip())
        
        frame_count += 1
    
    cap.release()
    return {"extracted_text": text_data}

@app.get("/video_info/")
def get_video_info(video_filename: str):
    file_path = os.path.join(UPLOAD_DIR, video_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open video file")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    return {
        "filename": video_filename,
        "frame_count": frame_count,
        "resolution": f"{frame_width}x{frame_height}",
        "fps": fps,
        "duration": duration
    }