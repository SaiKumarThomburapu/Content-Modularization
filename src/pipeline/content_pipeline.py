from src.components.content_modularization import ContentModularization
from src.exceptions import CustomException
from src.logger import logging
import sys
import os
import tempfile
import shutil

def process_text_classification(text: str) -> dict:
    """Process text classification and return results."""
    try:
        logging.info(f"Starting text classification for: {text[:50]}...")
        content_obj = ContentModularization()
        artifact = content_obj.classify_text(text)
        return artifact.classification_result
    except Exception as e:
        logging.error(f"Error in text classification pipeline: {e}")
        raise CustomException(e, sys) from e

def process_image_classification_with_ocr(image_path: str) -> dict:
    """Process image classification AND OCR text extraction + classification."""
    try:
        logging.info(f"Starting combined image processing for: {image_path}")
        content_obj = ContentModularization()
        
        # Step 1: Classify the image
        image_artifact = content_obj.classify_image(image_path)
        
        # Step 2: Extract text using OCR
        ocr_artifact = content_obj.extract_text_from_image(image_path)
        
        # Combine results
        combined_result = {
            "image_classification": image_artifact.classification_result,
            "ocr_extracted_text": ocr_artifact.extracted_text,
            "ocr_text_classification": ocr_artifact.text_classification
        }
        
        return combined_result
        
    except Exception as e:
        logging.error(f"Error in combined image processing pipeline: {e}")
        raise CustomException(e, sys) from e

def process_video_analysis(video_path: str) -> dict:
    """Process video analysis: classify frames for NSFW and transcribe audio for text classification."""
    try:
        logging.info(f"Starting video analysis for: {video_path}")
        content_obj = ContentModularization()
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        try:
            # Extract and classify frames (only image classification - NO OCR)
            frame_paths = content_obj.extract_frames(video_path, frames_dir)
            frame_results = []
            for frame_path in frame_paths:
                # Only image classification for frames
                image_artifact = content_obj.classify_image(frame_path)
                frame_results.append(image_artifact.classification_result)
            
            # Extract and transcribe audio, then classify the transcript
            content_obj.extract_audio(video_path, audio_path)
            audio_artifact = content_obj.transcribe_audio(audio_path)
            
            return {
                "frame_classifications": frame_results,  # Only image classification results
                "audio_transcript": audio_artifact.transcript,
                "transcript_classification": audio_artifact.transcript_classification
            }
            
        finally:
            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logging.error(f"Error in video analysis pipeline: {e}")
        raise CustomException(e, sys) from e


