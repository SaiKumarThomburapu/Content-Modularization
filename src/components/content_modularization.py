import os
import subprocess
from PIL import Image
import torch
import timm
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import easyocr

from src.exceptions import CustomException
from src.logger import logging
from src.entity.config import ContentConfig, TextConfig, ImageConfig, OCRConfig, AudioConfig, VideoConfig
from src.entity.artifact import TextArtifact, ImageArtifact, OCRArtifact, AudioArtifact, VideoArtifact
from src.utils import generate_unique_filename, create_directories
import sys

class ContentModularization:
    def __init__(self):
        try:
            self.content_config = ContentConfig()
            self.text_config = TextConfig(content_config=self.content_config)
            self.image_config = ImageConfig(content_config=self.content_config)
            self.ocr_config = OCRConfig(content_config=self.content_config)
            self.audio_config = AudioConfig(content_config=self.content_config)
            self.video_config = VideoConfig(content_config=self.content_config)
            
            # Initialize models
            self._initialize_models()
            
            # Create necessary directories
            create_directories([
                self.text_config.results_dir,
                self.image_config.results_dir,
                self.ocr_config.results_dir,
                self.audio_config.results_dir,
                self.video_config.results_dir
            ])
            
        except Exception as e:
            raise CustomException(e, sys)

    def _initialize_models(self):
        """Initialize all ML models"""
        try:
            # Text classifier
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_config.model_name)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(self.text_config.model_name).eval()
            
            # Image classifier
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.img_model = timm.create_model(self.image_config.model_name, pretrained=True).eval().to(self.device)
            cfg = timm.data.resolve_model_data_config(self.img_model)
            self.img_transforms = timm.data.create_transform(**cfg, is_training=False)
            self.img_labels = self.img_model.pretrained_cfg["label_names"]
            
            # ASR model
            self.asr = whisper.load_model(self.audio_config.model_name)
            
            # OCR reader
            self.ocr_reader = easyocr.Reader(self.ocr_config.languages, gpu=torch.cuda.is_available())
            
            logging.info("All models initialized successfully")
            
        except Exception as e:
            raise CustomException(e, sys)

    def classify_text(self, text: str) -> TextArtifact:
        try:
            logging.info(f"Classifying text: {text[:50]}...")
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = self.text_model(**inputs).logits[0]
            
            score = torch.sigmoid(logits[1]).item()
            label = "toxic" if score > self.text_config.threshold else "non-toxic"
            
            result = {
                "text": text,
                "label": label,
                "score": round(score, 4),
                "is_toxic": score > self.text_config.threshold
            }
            
            logging.info(f"Text classification completed: {label}")
            return TextArtifact(classification_result=result)
            
        except Exception as e:
            logging.error(f"Error in text classification: {e}")
            raise CustomException(e, sys)

    def classify_image(self, image_path: str) -> ImageArtifact:
        try:
            logging.info(f"Classifying image: {image_path}")
            
            img = Image.open(image_path).convert("RGB")
            tensor = self.img_transforms(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                probs = self.img_model(tensor).softmax(dim=-1)[0].cpu()
            
            scores = {lbl: float(probs[i]) for i, lbl in enumerate(self.img_labels)}
            top = max(scores, key=scores.get)
            
            result = {
                "image_path": image_path,
                "label": top,
                "score": round(scores[top], 4),
                "full_probs": {k: round(v, 4) for k, v in scores.items()},
                "is_nsfw": scores.get("NSFW", 0) > self.image_config.threshold
            }
            
            logging.info(f"Image classification completed: {top}")
            return ImageArtifact(classification_result=result)
            
        except Exception as e:
            logging.error(f"Error in image classification: {e}")
            raise CustomException(e, sys)

    def extract_text_from_image(self, image_path: str) -> OCRArtifact:
        try:
            logging.info(f"Extracting text from image: {image_path}")
            
            results = self.ocr_reader.readtext(image_path, detail=0)
            extracted_text = " ".join(results)
            
            # Classify extracted text
            text_artifact = self.classify_text(extracted_text)
            
            logging.info(f"OCR extraction completed. Text: {extracted_text[:50]}...")
            return OCRArtifact(
                extracted_text=extracted_text,
                text_classification=text_artifact.classification_result
            )
            
        except Exception as e:
            logging.error(f"Error in OCR extraction: {e}")
            raise CustomException(e, sys)

    def extract_audio(self, video_path: str, output_audio_path: str) -> str:
        try:
            logging.info(f"Extracting audio from video: {video_path}")
            
            cmd = [
                "ffmpeg", "-i", video_path,
                "-q:a", "0", "-map", "a", output_audio_path,
                "-y", "-loglevel", "error"
            ]
            
            subprocess.run(cmd, check=True)
            logging.info(f"Audio extracted to: {output_audio_path}")
            return output_audio_path
            
        except Exception as e:
            logging.error(f"Error in audio extraction: {e}")
            raise CustomException(e, sys)

    def extract_frames(self, video_path: str, output_dir: str) -> list:
        try:
            logging.info(f"Extracting frames from video: {video_path}")
            
            os.makedirs(output_dir, exist_ok=True)
            pattern = os.path.join(output_dir, "frame_%04d.jpg")
            
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps={self.video_config.fps}",
                pattern, "-hide_banner", "-loglevel", "error"
            ]
            
            subprocess.run(cmd, check=True)
            frame_paths = sorted([
                os.path.join(output_dir, f) 
                for f in os.listdir(output_dir) 
                if f.endswith(".jpg")
            ])
            
            logging.info(f"Extracted {len(frame_paths)} frames")
            return frame_paths
            
        except Exception as e:
            logging.error(f"Error in frame extraction: {e}")
            raise CustomException(e, sys)

    def transcribe_audio(self, audio_path: str) -> AudioArtifact:
        try:
            logging.info(f"Transcribing audio: {audio_path}")
            
            result = self.asr.transcribe(audio_path)
            transcript = result["text"]
            
            # Classify transcript
            text_artifact = self.classify_text(transcript)
            
            logging.info(f"Audio transcription completed. Text: {transcript[:50]}...")
            return AudioArtifact(
                transcript=transcript,
                transcript_classification=text_artifact.classification_result
            )
            
        except Exception as e:
            logging.error(f"Error in audio transcription: {e}")
            raise CustomException(e, sys)
