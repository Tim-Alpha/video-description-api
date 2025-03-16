import os
import requests
import json
import time
from google.generativeai import configure, generate_content

# Configure Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=GEMINI_API_KEY)

MAX_RETRIES = 10
RETRY_DELAY = 5
API_URL = "http://13.92.184.232:8000/api/v1/analyze_video"

def start_analysis(video_url, identifier="custom_identifier"):
    """Send a video URL for processing and return task ID."""
    payload = {"url": video_url, "identifier": identifier}

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "success":
            print(f"Video submitted successfully: {data.get('message')}")
            return data.get("task_id")
        else:
            print(f"Unexpected response: {json.dumps(data, indent=2)}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None

def get_analysis_result(task_id):
    """Retrieve the video analysis result using the task ID."""
    result_url = f"http://13.92.184.232:8000/api/v1/analysis_result/{task_id}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(result_url, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "completed":
                return process_gemini_analysis(result)
            elif result.get("status") == "error":
                print(f"Analysis failed: {result.get('message')}")
                return None

            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Processing... Retrying in {RETRY_DELAY} seconds")
            time.sleep(RETRY_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")

    print("Max retries reached. No result available.")
    return None

def process_gemini_analysis(result):
    """Use Gemini API to refine and enhance the video description."""
    try:
        prompt = f"""
        Given the following video analysis data, provide a structured summary:

        Description: {result.get("description")}
        Keywords: {", ".join([kw["keyword"] for kw in result.get("keywords", [])])}
        Topics: {", ".join(result.get("topics", []))}
        Actions: {", ".join(result.get("actions", []))}
        Emotions: {", ".join(result.get("emotions", []))}
        Visual Elements: {", ".join(result.get("visual_elements", []))}
        Audio Elements: {", ".join(result.get("audio_elements", []))}
        Genre: {result.get("genre")}
        Target Audience: {", ".join(result.get("target_audience", []))}
        Duration Estimate: {result.get("duration_estimate")}
        Quality Indicators: {", ".join(result.get("quality_indicators", []))}
        Transcript: {result.get("transcript")}
        """

        response = generate_content(model="gemini-pro", prompt=prompt)
        return response.text if response else "No response from Gemini API."
    except Exception as e:
        print(f"Error processing Gemini API: {e}")
        return None

def main():
    video_url = "https://example.com/video.mp4"
    identifier = "custom_identifier"
    task_id = start_analysis(video_url, identifier)

    if task_id:
        result = get_analysis_result(task_id)
        if result:
            print("\nFinal Video Analysis Result:")
            print(result)

if __name__ == "__main__":
    main()
