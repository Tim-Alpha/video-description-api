import os
import time
from moviepy.editor import VideoFileClip
import google.generativeai as genai
import tempfile
from app.core.logging import logger
from datetime import datetime
from app.core.config import settings
from app.core.task_tracker import task_tracker
from pydub import AudioSegment
from typing import Tuple, List, Optional
import math
import re
import json
import asyncio

OUTPUT_FOLDER = os.path.abspath("video_analysis_output")

# Initialize Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)

MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24MB to stay safely under the 25MB limit
CHUNK_DURATION = 10 * 60 * 1000  # 10 minutes in milliseconds

# NSFW content detection patterns
NSFW_PATTERNS = [
    r'\b(?:sex|porn|xxx|adult|nude|naked|explicit|nsfw)\b',
    r'\b(?:masturbat(?:e|ion)|orgasm|erotic)\b',
    r'\b(?:breast|boob|tit|ass|penis|vagina|dick|cock|pussy)\b',
    r'\b(?:fuck|shit|bitch|cunt|whore|slut)\b',
    r'\b(?:strip(?:ping|per)|escort|prostitut(?:e|ion))\b',
    r'\b(?:hentai|rule34|onlyfans)\b'
]

async def process_audio(video_content: bytes, task_id: str = None) -> Tuple[List[dict], Optional[str]]:
    """
    Process audio from video content, handling large files by splitting into chunks.
    """
    temp_video_file = None
    temp_audio_file = None
    output_folder = "video_analysis_output"
    
    try:
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Output folder created/confirmed: {os.path.abspath(output_folder)}")
        
        # Save video content to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_content)
            temp_video_file = temp_file.name
        logger.info(f"Temporary video file created at: {temp_video_file}")
        
        if task_id:
            task_tracker.update_progress(task_id, "Video file saved", 10)
        
        # Load video and extract audio
        video = AudioSegment.from_file(temp_video_file)
        if task_id:
            task_tracker.update_progress(task_id, "Video loaded for audio extraction", 15)
        
        # Save extracted audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = os.path.join(output_folder, f"extracted_audio_{timestamp}.wav")
        video.export(audio_filename, format="wav")
        logger.info(f"Saved extracted audio locally: {audio_filename}")
        
        if task_id:
            task_tracker.update_progress(task_id, "Audio extracted and saved", 25)
        
        # Get file info
        file_size = os.path.getsize(audio_filename)
        logger.info(f"Audio file size: {file_size} bytes")
        
        # Process audio in chunks if necessary
        if task_id:
            task_tracker.update_progress(task_id, "Starting audio transcription", 30)
        
        logger.info("Transcribing audio using Gemini API...")
        
        # Calculate number of chunks needed
        audio_length = len(video)
        num_chunks = math.ceil(audio_length / CHUNK_DURATION)
        logger.info(f"Audio length: {audio_length}ms, splitting into {num_chunks} chunks")
        
        # Process audio in chunks
        transcriptions = []
        chunk_files = []
        
        try:
            for i in range(num_chunks):
                start_time = i * CHUNK_DURATION
                end_time = min((i + 1) * CHUNK_DURATION, audio_length)
                
                # Extract chunk
                chunk = video[start_time:end_time]
                
                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_chunk:
                    chunk.export(temp_chunk.name, format="wav")
                    chunk_files.append(temp_chunk.name)
                    
                    # Check chunk size
                    chunk_size = os.path.getsize(temp_chunk.name)
                    logger.info(f"Chunk {i+1}/{num_chunks} size: {chunk_size} bytes")
                    
                    if chunk_size > MAX_CHUNK_SIZE:
                        raise ValueError(f"Chunk {i+1} size ({chunk_size} bytes) exceeds maximum allowed size ({MAX_CHUNK_SIZE} bytes)")
                    
                    # Transcribe chunk using Gemini API
                    transcription = await transcribe_audio_with_gemini(temp_chunk.name)
                    transcriptions.append(transcription)
                        
                    logger.info(f"Chunk {i+1}/{num_chunks} transcribed successfully")
                    
                    if task_id:
                        progress = 30 + (i + 1) * (35 - 30) / num_chunks
                        task_tracker.update_progress(task_id, f"Transcribed chunk {i+1}/{num_chunks}", progress)
            
            # Combine all transcriptions
            combined_text = " ".join(transcriptions)
            result = [{"text": combined_text}]
            
            if task_id:
                task_tracker.update_progress(task_id, "Audio transcription completed", 35)
                task_tracker.update_progress(task_id, "Audio processing completed", 40)
            
            return result, audio_filename
            
        finally:
            # Clean up chunk files
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    try:
                        os.unlink(chunk_file)
                        logger.info(f"Cleaned up chunk file: {chunk_file}")
                    except Exception as e:
                        logger.error(f"Error cleaning up chunk file: {str(e)}")
        
    except Exception as e:
        error_msg = f"Error in audio processing: {str(e)}"
        logger.error(error_msg)
        if task_id:
            task_tracker.update_progress(task_id, f"Error: {error_msg}", 35)
        return [{"error": error_msg}], None
        
    finally:
        # Clean up temporary files
        if temp_video_file and os.path.exists(temp_video_file):
            try:
                os.unlink(temp_video_file)
                logger.info(f"Cleaned up temporary video file: {temp_video_file}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary video file: {str(e)}")

async def transcribe_audio_with_gemini(audio_file_path: str) -> str:
    """
    Transcribe audio using Gemini API.
    """
    try:
        # Create a coroutine to run the synchronous Gemini API call in a separate thread
        loop = asyncio.get_event_loop()
        
        def sync_transcribe():
            # Open the audio file
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Use Gemini's Audio model for transcription
            # Note: This is a placeholder for the actual Gemini API call
            # You'll need to replace this with the actual Gemini audio transcription method
            model = genai.GenerativeModel('gemini-pro-vision')
            
            # We're using the vision model to analyze audio since Gemini doesn't have a dedicated audio API
            # We'll send a prompt asking it to transcribe the audio
            response = model.generate_content(
                [
                    "Please transcribe the following audio file in full detail:",
                    {"mime_type": "audio/wav", "data": audio_data}
                ]
            )
            
            # Extract the transcription from the response
            return response.text
        
        # Run the synchronous function in a thread pool
        transcription = await loop.run_in_executor(None, sync_transcribe)
        return transcription
    
    except Exception as e:
        logger.error(f"Error transcribing audio with Gemini: {str(e)}")
        return f"Transcription error: {str(e)}"

async def check_content_safety(text: str) -> Tuple[bool, List[str]]:
    """
    Check if the content is safe by analyzing the text for NSFW content.
    Returns a tuple of (is_safe, warnings).
    """
    try:
        # First check for explicit patterns
        warnings = []
        for pattern in NSFW_PATTERNS:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                warnings.append(f"Detected inappropriate content: {match.group()}")
        
        if warnings:
            return False, warnings

        # Use Gemini to check for more subtle NSFW content
        try:
            safety_prompt = f"""
            Task: Analyze the following text for inappropriate content.
            Be extremely conservative - if there's any doubt, mark it as inappropriate.
            Return a JSON object with this exact format:
            {{
                "is_safe": boolean,
                "warnings": [list of specific warnings],
                "reason": "detailed explanation"
            }}
            
            Text to analyze:
            {text}
            """
            
            # Create a coroutine to run the synchronous Gemini API call in a separate thread
            loop = asyncio.get_event_loop()
            
            def sync_safety_check():
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(
                    safety_prompt,
                    generation_config={"temperature": 0.1}
                )
                return response.text
            
            # Run the synchronous function in a thread pool
            safety_result_text = await loop.run_in_executor(None, sync_safety_check)
            
            # Extract the JSON part of the response
            # Look for JSON between curly braces
            json_match = re.search(r'\{.*\}', safety_result_text, re.DOTALL)
            if json_match:
                safety_result_json = json_match.group(0)
                result = json.loads(safety_result_json)
            else:
                # If no JSON found, try to parse the entire response
                result = json.loads(safety_result_text)
            
            if not result.get("is_safe", False):
                warnings.extend(result.get("warnings", []))
                reason = result.get("reason", "Content flagged as inappropriate")
                if reason and reason not in warnings:
                    warnings.append(reason)
                return False, warnings
            
            return True, []
            
        except asyncio.TimeoutError:
            logger.error("Timeout during content safety check")
            return False, ["Content safety check timed out"]
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return False, ["Unable to verify content safety"]
        except Exception as e:
            logger.error(f"Error in Gemini content check: {str(e)}")
            return False, ["Error checking content safety"]
        
    except Exception as e:
        logger.error(f"Error in content safety check: {str(e)}")
        return False, ["Error in content safety check"]
