import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from app.main import app
import os
import json

VIDEO_PATH = "Final1.mp4"

@pytest.mark.asyncio
async def test_analyze_video():
    print("\n\n" + "="*50)
    print("Starting test_analyze_video")
    print("="*50 + "\n")

    assert os.path.exists(VIDEO_PATH), f"Test video file not found: {VIDEO_PATH}"

    file_size = os.path.getsize(VIDEO_PATH)
    print(f"Test video file size: {file_size} bytes")

    with open(VIDEO_PATH, "rb") as video_file:
        video_content = video_file.read()

    print(f"Read {len(video_content)} bytes from video file")

    async with AsyncClient(app=app, base_url="http://test") as ac:
        print("Sending POST request to /api/v1/analyze_video")
        response = await ac.post(
            "/api/v1/analyze_video",
            files={"video": ("Final1.mp4", video_content, "video/mp4")}
        )

    print(f"Response status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response content: {response.content.decode()}")

    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}. Response: {response.text}"
    
    json_response = response.json()
    assert "description" in json_response, "Response is missing 'description' field"
    assert "metadata" in json_response, "Response is missing 'metadata' field"
    assert "audio_transcription" in json_response, "Response is missing 'audio_transcription' field"
    
    print("\nResponse:")
    print(json.dumps(json_response, indent=2))

    print("\n" + "="*50)
    print("test_analyze_video completed")
    print("="*50 + "\n")

def test_analyze_video_no_file():
    client = TestClient(app)
    response = client.post("/api/v1/analyze_video")
    assert response.status_code == 422  # Unprocessable Entity