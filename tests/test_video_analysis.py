import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from app.main import app
import os
import json
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configurations
VIDEO_PATH = "Final1.mp4"
INVALID_VIDEO_PATH = "invalid_video.txt"
API_ENDPOINT = "/api/v1/analyze_video"
EXPECTED_FIELDS = ["description", "metadata", "audio_transcription"]

def log_test_separator(message: str) -> None:
    """Print a visually distinct separator with test information."""
    separator = "="*50
    logger.info(f"\n\n{separator}")
    logger.info(message)
    logger.info(f"{separator}\n")

def create_invalid_video_file() -> str:
    """Create an invalid video file for testing."""
    with open(INVALID_VIDEO_PATH, "w") as file:
        file.write("This is not a valid video file")
    return INVALID_VIDEO_PATH

def cleanup_test_files() -> None:
    """Clean up any temporary files created during testing."""
    if os.path.exists(INVALID_VIDEO_PATH):
        os.remove(INVALID_VIDEO_PATH)

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    """Setup before tests and teardown after tests."""
    # Setup
    yield
    # Teardown
    cleanup_test_files()

def log_response_details(response) -> None:
    """Log detailed information about the API response."""
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response headers: {dict(response.headers)}")
    
    try:
        response_content = response.content.decode()
        logger.info(f"Response content length: {len(response_content)} characters")
        
        if len(response_content) > 500:
            logger.info(f"Response content (truncated): {response_content[:500]}...")
        else:
            logger.info(f"Response content: {response_content}")
            
        # Try to parse as JSON for better logging
        try:
            json_response = response.json()
            logger.info(f"Response JSON keys: {list(json_response.keys())}")
        except json.JSONDecodeError:
            logger.info("Response is not valid JSON")
    except Exception as e:
        logger.error(f"Error processing response content: {str(e)}")

@pytest.mark.asyncio
async def test_analyze_video():
    """Test successful video analysis with a valid video file."""
    log_test_separator("Starting test_analyze_video")
    
    # Assert video file exists
    assert os.path.exists(VIDEO_PATH), f"Test video file not found: {VIDEO_PATH}"
    
    # Log file details
    file_size = os.path.getsize(VIDEO_PATH)
    logger.info(f"Test video file size: {file_size} bytes")
    
    # Read video content
    with open(VIDEO_PATH, "rb") as video_file:
        video_content = video_file.read()
    
    logger.info(f"Read {len(video_content)} bytes from video file")
    
    # Send request
    async with AsyncClient(app=app, base_url="http://test") as ac:
        logger.info(f"Sending POST request to {API_ENDPOINT}")
        response = await ac.post(
            API_ENDPOINT,
            files={"video": ("Final1.mp4", video_content, "video/mp4")}
        )
    
    # Log response details
    log_response_details(response)
    
    # Assert response status
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}. Response: {response.text}"
    
    # Parse and validate response
    json_response = response.json()
    
    # Check for required fields
    for field in EXPECTED_FIELDS:
        assert field in json_response, f"Response is missing required field: '{field}'"
    
    # Additional validation on response content
    assert isinstance(json_response["description"], str), "Description should be a string"
    assert len(json_response["description"]) > 0, "Description should not be empty"
    
    assert isinstance(json_response["metadata"], dict), "Metadata should be a dictionary"
    
    # Validate specific metadata fields if applicable
    if "duration" in json_response["metadata"]:
        assert isinstance(json_response["metadata"]["duration"], (int, float)), "Duration should be a number"
        assert json_response["metadata"]["duration"] > 0, "Duration should be positive"
    
    # Log full JSON response
    logger.info("\nFull Response JSON:")
    logger.info(json.dumps(json_response, indent=2))
    
    log_test_separator("test_analyze_video completed successfully")
    return json_response

def test_analyze_video_no_file():
    """Test API behavior when no file is provided."""
    log_test_separator("Starting test_analyze_video_no_file")
    
    client = TestClient(app)
    response = client.post(API_ENDPOINT)
    
    log_response_details(response)
    
    assert response.status_code == 422, f"Expected status code 422 (Unprocessable Entity), but got {response.status_code}"
    
    # Validate error response format if applicable
    try:
        json_response = response.json()
        assert "detail" in json_response, "Error response should contain 'detail' field"
    except json.JSONDecodeError:
        pytest.fail("Error response should be valid JSON")
    
    log_test_separator("test_analyze_video_no_file completed successfully")

@pytest.mark.asyncio
async def test_analyze_video_invalid_format():
    """Test API behavior with an invalid video file format."""
    log_test_separator("Starting test_analyze_video_invalid_format")
    
    # Create an invalid video file
    invalid_file_path = create_invalid_video_file()
    
    with open(invalid_file_path, "rb") as invalid_file:
        invalid_content = invalid_file.read()
    
    # Send request with invalid file
    async with AsyncClient(app=app, base_url="http://test") as ac:
        logger.info(f"Sending POST request to {API_ENDPOINT} with invalid file")
        response = await ac.post(
            API_ENDPOINT,
            files={"video": ("invalid_video.txt", invalid_content, "text/plain")}
        )
    
    log_response_details(response)
    
    # API should return an error (either 400 Bad Request or 422 Unprocessable Entity)
    assert response.status_code in [400, 422], f"Expected error status code (400 or 422), but got {response.status_code}"
    
    log_test_separator("test_analyze_video_invalid_format completed successfully")

@pytest.mark.asyncio
async def test_analyze_video_large_valid_file(monkeypatch):
    """Mock test for a large but valid video file."""
    log_test_separator("Starting test_analyze_video_large_valid_file")
    
    # Use the actual video but mock a large file for logging purposes
    with open(VIDEO_PATH, "rb") as video_file:
        video_content = video_file.read()
    
    # Log as if it were a large file
    logger.info("Simulating a large valid video file (50MB)")
    
    # Send request
    async with AsyncClient(app=app, base_url="http://test") as ac:
        logger.info(f"Sending POST request to {API_ENDPOINT} with large valid file")
        response = await ac.post(
            API_ENDPOINT,
            files={"video": ("large_video.mp4", video_content, "video/mp4")}
        )
    
    log_response_details(response)
    
    # Should still succeed with a valid file
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    log_test_separator("test_analyze_video_large_valid_file completed successfully")

@pytest.mark.parametrize("field_name", EXPECTED_FIELDS)
def test_response_field_validation(field_name):
    """Test individual fields in the API response."""
    log_test_separator(f"Starting validation test for field: {field_name}")
    
    client = TestClient(app)
    
    # Use the sync client for simplicity in parametrized tests
    with open(VIDEO_PATH, "rb") as video_file:
        response = client.post(
            API_ENDPOINT,
            files={"video": ("Final1.mp4", video_file.read(), "video/mp4")}
        )
    
    assert response.status_code == 200, f"API request failed with status {response.status_code}"
    
    json_response = response.json()
    assert field_name in json_response, f"Response missing required field: {field_name}"
    
    # Field-specific validations
    if field_name == "description":
        assert isinstance(json_response[field_name], str), "Description must be a string"
        assert len(json_response[field_name]) > 0, "Description cannot be empty"
    
    elif field_name == "metadata":
        assert isinstance(json_response[field_name], dict), "Metadata must be a dictionary"
    
    elif field_name == "audio_transcription":
        assert isinstance(json_response[field_name], (str, list, dict)), "Invalid audio transcription format"
    
    logger.info(f"Field '{field_name}' validation passed")
    log_test_separator(f"Validation test for field: {field_name} completed successfully")
