import requests
import json
import time
import os

MAX_RETRIES = 20
RETRY_DELAY = 5

def start_analysis(video_path):
    # url = "http://127.0.0.1:8000/api/v1/analyze_video"
    url = "http://13.92.184.232:8000/api/v1/analyze_video"

    if video_path.startswith(('http://', 'https://')):
        params = {"file_url": video_path}
        response = requests.post(url, params=params)
    else:
        with open(video_path, "rb") as video_file:
            files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
            response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
        return response.json().get('task_id')
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    
def get_analysis_result(task_id):
    # url = f"http://127.0.0.1:8000/api/v1/analysis_result/{task_id}"
    url = f"http://13.92.184.232:8000/api/v1/analysis_result/{task_id}"
    for attempt in range(MAX_RETRIES):
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'completed':
                return result
            elif result.get('status') == 'error':
                print(f"Error in analysis: {result.get('message')}")
                return None
        print(f"Attempt {attempt + 1}: Result not ready. Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
    print("Max retries reached. Analysis result not available.")
    return None

def main():
    video_path = "https://video-cdn.socialverseapp.com/swapnil_e1994ef0-93e5-4c2e-9f53-1c7d63e1bb82.mp4"  # Replace with your video file path or url
    # video_path = "demo.mp4"  # Replace with your video file path or url
    task_id = start_analysis(video_path)
    if task_id:
        result = get_analysis_result(task_id)
        if result:
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()