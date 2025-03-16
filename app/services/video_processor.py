import cv2
import numpy as np
from PIL import Image
import io
import base64
from app.core.logging import logger
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import asyncio
from collections import defaultdict
import tempfile
import os
from app.core.task_tracker import task_tracker
import json
import google.generativeai as genai
from google.generativeai import GenerativeModel
from app.core.config import settings

# Task queue to store processing results
task_queue: Dict[str, Dict] = defaultdict(dict)

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)

async def split_video(video_content: bytes, task_id: str) -> List[bytes]:
    """
    Split video into parts based on duration.
    
    Args:
        video_content (bytes): Raw video content
        task_id (str): Unique task identifier
        
    Returns:
        List[bytes]: List of video chunks in memory
    """
    temp_input_file = None
    temp_output_file = None
    try:
        task_tracker.update_progress(task_id, "Saving video to temporary file", 6)
        # First, save the input video content to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_content)
            temp_input_file = temp_file.name
        
        task_tracker.update_progress(task_id, "Opening video file", 7)
        # Open video from the temporary file
        video = cv2.VideoCapture(temp_input_file)
        if not video.isOpened():
            raise ValueError(f"Could not open video content from {temp_input_file}")
        
        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        task_tracker.update_progress(task_id, "Calculating video parts", 8)
        # Calculate number of parts (max 5)
        minutes = duration / 60
        num_parts = min(5, max(1, int(minutes)))
        frames_per_part = total_frames // num_parts
        
        logger.info(f"Video properties: {total_frames} frames, {fps} FPS, Duration: {duration:.2f} seconds")
        logger.info(f"Splitting into {num_parts} parts, {frames_per_part} frames per part")
        
        # Split video into parts
        video_parts = []
        progress_per_part = 7 / num_parts  # 7% progress allocated for splitting
        
        for i in range(num_parts):
            task_tracker.update_progress(task_id, f"Processing video part {i+1}/{num_parts}", 8 + (i * progress_per_part))
            
            # Create temporary file for output chunk
            with tempfile.NamedTemporaryFile(suffix=f'_part_{i}.mp4', delete=False) as out_file:
                temp_output_file = out_file.name
            
            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                temp_output_file,
                fourcc,
                fps,
                (width, height)
            )
            
            start_frame = i * frames_per_part
            end_frame = start_frame + frames_per_part if i < num_parts - 1 else total_frames
            
            logger.info(f"Processing part {i+1}: frames {start_frame} to {end_frame}")
            
            for frame_idx in range(int(start_frame), int(end_frame)):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video.read()
                if ret:
                    writer.write(frame)
            
            writer.release()
            
            # Read the chunk back and append to list
            with open(temp_output_file, 'rb') as f:
                video_parts.append(f.read())
            
            # Clean up output file
            os.unlink(temp_output_file)
        
        task_tracker.update_progress(task_id, "Video splitting completed", 15)
        return video_parts
        
    except Exception as e:
        logger.error(f"Error in split_video: {str(e)}")
        raise
    
    finally:
        if 'video' in locals():
            video.release()
        # Clean up temporary files
        if temp_input_file and os.path.exists(temp_input_file):
            try:
                os.unlink(temp_input_file)
            except Exception as e:
                logger.error(f"Error deleting temporary input file: {str(e)}")
        if temp_output_file and os.path.exists(temp_output_file):
            try:
                os.unlink(temp_output_file)
            except Exception as e:
                logger.error(f"Error deleting temporary output file: {str(e)}")

async def extract_frames(video_chunk: bytes) -> str:
    """
    Extract frames from a video chunk and create a grid visualization.
    
    Args:
        video_chunk (bytes): Video chunk content
        
    Returns:
        str: Base64 encoded grid image
    """
    temp_file = None
    try:
        # Save video chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp:
            temp.write(video_chunk)
            temp_file = temp.name
        
        # Open video from temporary file
        video = cv2.VideoCapture(temp_file)
        if not video.isOpened():
            raise ValueError(f"Could not open video chunk from {temp_file}")
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError("Video chunk contains no frames")
            
        interval = max(1, total_frames // 16)
        logger.info(f"Extracting frames from chunk: {total_frames} total frames, interval {interval}")
        
        frames = []
        for i in range(16):
            frame_pos = min(i * interval, total_frames - 1)  # Ensure we don't exceed total frames
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = video.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                logger.warning(f"Failed to read frame at position {frame_pos}")
        
        if not frames:
            logger.warning("No frames were extracted from the video chunk")
            return None
            
        logger.info(f"Successfully extracted {len(frames)} frames")
        
        # Create grid image
        grid = Image.new('RGB', (frames[0].shape[1] * 4, frames[0].shape[0] * 4))
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            x_position = (i % 4) * frames[0].shape[1]
            y_position = (i // 4) * frames[0].shape[0]
            grid.paste(img, (x_position, y_position))
        
        # Convert to base64
        buffer = io.BytesIO()
        grid.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error in extract_frames: {str(e)}")
        return None
        
    finally:
        if 'video' in locals():
            video.release()
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")

async def check_content_moderation(base64_images: List[str]) -> Tuple[bool, List[str]]:
    """
    Ultra-strict content moderation using Gemini's content safety features.
    """
    try:
        all_warnings = []
        is_safe = True
        
        # Map for severity levels
        severity_map = {
            "HIGH": "CRITICAL",
            "MEDIUM": "HIGH",
            "LOW": "MEDIUM"
        }
        
        # Category mapping
        category_map = {
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "Sexual Content",
            "HARM_CATEGORY_HATE_SPEECH": "Hate Speech",
            "HARM_CATEGORY_HARASSMENT": "Harassment",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "Dangerous Content",
            "HARM_CATEGORY_UNSPECIFIED": "Unspecified Risk"
        }
        
        # Process each image individually
        for idx, base64_image in enumerate(base64_images):
            # Initialize the model
            model = GenerativeModel("gemini-1.5-pro-vision")
            
            # Prepare the image content
            image_part = {
                "mime_type": "image/png",
                "data": base64_image
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    [
                        "Perform a strict content safety analysis on this image. Identify any potential safety concerns.",
                        image_part
                    ],
                    safety_settings={
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM",
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM"
                    }
                )
            )
            
            # Check if the response was blocked
            if hasattr(response, "prompt_feedback") and response.prompt_feedback.block_reason:
                is_safe = False
                block_reason = response.prompt_feedback.block_reason
                all_warnings.append(f"CRITICAL RISK - Content blocked by moderation system: {block_reason}")
                continue
                
            # Check for safety attributes
            if hasattr(response, "safety_ratings"):
                for rating in response.safety_ratings:
                    if rating.probability and rating.probability not in ["NEGLIGIBLE", "LOW"]:
                        is_safe = False
                        category = category_map.get(rating.category, "Unknown Category")
                        severity = severity_map.get(rating.probability, "LOW")
                        all_warnings.append(f"{severity} RISK - {category} detected")
                        
            # Also analyze the content of the response for additional safety checks
            try:
                text_response = response.text.strip()
                if any(keyword in text_response.lower() for keyword in ["unsafe", "inappropriate", "explicit", "violence", "harmful"]):
                    is_safe = False
                    # Extract the reason from the response text
                    lines = text_response.split('\n')
                    for line in lines:
                        if any(keyword in line.lower() for keyword in ["unsafe", "inappropriate", "explicit", "violence", "harmful"]):
                            all_warnings.append(f"MEDIUM RISK - Potential issue detected: {line}")
            except Exception as text_error:
                logger.error(f"Error analyzing text response: {str(text_error)}")
        
        # Remove duplicates while preserving order
        seen = set()
        filtered_warnings = []
        for warning in all_warnings:
            if warning not in seen:
                seen.add(warning)
                filtered_warnings.append(warning)
        
        # Sort warnings by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "SYSTEM FLAG": 4}
        filtered_warnings.sort(key=lambda x: severity_order.get(x.split()[0], 5))
        
        # Add summary warnings for high-risk content
        if not is_safe:
            summary_warnings = []
            if any("CRITICAL" in w for w in filtered_warnings):
                summary_warnings.append("CRITICAL RISK - Severe content violations detected")
            if any("HIGH" in w for w in filtered_warnings):
                summary_warnings.append("HIGH RISK - Significant content concerns identified")
            filtered_warnings = summary_warnings + filtered_warnings
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filtered_warnings.insert(0, f"Content Moderation Timestamp: {timestamp}")
        
        # Log results
        logger.info(f"Content moderation complete - Safe: {is_safe}, Warnings: {len(filtered_warnings)}")
        for warning in filtered_warnings:
            logger.warning(f"Content Warning: {warning}")
        
        return is_safe, filtered_warnings
    
    except Exception as e:
        logger.error(f"Error in content moderation: {str(e)}")
        return False, ["CRITICAL RISK - Error in content moderation system"]

async def analyze_grid_images(base64_images: List[str], task_id: str = None) -> List[str]:
    """
    Analyze grid images using Gemini vision model.
    """
    try:
        descriptions = []
        total_images = len(base64_images)
        
        for idx, base64_image in enumerate(base64_images, 1):
            if task_id:
                progress = 65 + (idx * 5 / total_images)
                task_tracker.update_progress(task_id, f"Analyzing grid image {idx}/{total_images}", progress)
            
            try:
                # Initialize the model
                model = GenerativeModel("gemini-1.5-pro-vision")
                
                # Prepare the prompt
                system_prompt = "You are a video frame analysis expert. Describe the key visual elements, actions, and details in this frame grid."
                user_prompt = "Analyze this grid of video frames. Focus on: main subjects, actions, visual elements, text overlays, scene composition, and any notable details."
                
                # Prepare the image content
                image_part = {
                    "mime_type": "image/png",
                    "data": base64_image
                }
                
                # Make the API call
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(
                        [
                            system_prompt,
                            user_prompt,
                            image_part
                        ]
                    )
                )
                
                # Extract description
                description = response.text.strip()
                descriptions.append(description)
                
            except Exception as e:
                logger.error(f"Error analyzing grid image {idx}: {str(e)}")
                descriptions.append(f"Error analyzing frame grid {idx}")
        
        return descriptions
        
    except Exception as e:
        logger.error(f"Error in grid image analysis: {str(e)}")
        return ["Error analyzing frame grids"]

async def process_video(video_content: bytes, task_id: str) -> Tuple[bool, List[str], List[str]]:
    """
    Main video processing function that coordinates the entire workflow.
    """
    try:
        task_tracker.update_progress(task_id, "Starting video processing", 5)
        # Split video into parts
        video_parts = await split_video(video_content, task_id)
        task_tracker.update_progress(task_id, "Video split completed", 15)
        
        # Process each part in parallel
        tasks = [extract_frames(part) for part in video_parts]
        base64_grids = await asyncio.gather(*tasks)
        task_tracker.update_progress(task_id, "Frame extraction completed", 25)
        
        # Store grids in task queue
        task_queue[task_id]['grids'] = base64_grids
        
        # Filter out None values and check content moderation for all grids in one call
        valid_grids = [grid for grid in base64_grids if grid is not None]
        if valid_grids:
            task_tracker.update_progress(task_id, "Starting content moderation", 30)
            is_safe, warnings = await check_content_moderation(valid_grids)
            task_tracker.update_progress(task_id, "Content moderation completed", 35)
        else:
            is_safe, warnings = False, ["No valid frames extracted"]
            valid_grids = []
        
        print(f"\n{'='*30}\nIs Safe: {is_safe}\n{'='*30}")
        print(f"\n{'='*30}\nWarnings: {warnings}\n{'='*30}")
        
        # Store results in task queue
        task_queue[task_id]['is_safe'] = is_safe
        task_queue[task_id]['warnings'] = warnings
        
        return is_safe, warnings, valid_grids
    
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        task_queue[task_id]['error'] = str(e)
        return False, [f"Processing error: {str(e)}"], []
