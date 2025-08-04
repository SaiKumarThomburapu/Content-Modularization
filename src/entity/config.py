from src.exceptions import CustomException
from src.logger import logging
import sys

from src.constants import *

class ContentConfig:
    def __init__(self):
        self.text_results_dir = TEXT_RESULTS_DIR
        self.image_results_dir = IMAGE_RESULTS_DIR
        self.ocr_results_dir = OCR_RESULTS_DIR
        self.audio_results_dir = AUDIO_RESULTS_DIR
        self.video_results_dir = VIDEO_RESULTS_DIR
        self.text_model_name = TEXT_MODEL_NAME
        self.image_model_name = IMAGE_MODEL_NAME
        self.whisper_model_name = WHISPER_MODEL_NAME
        self.text_threshold = TEXT_TOXICITY_THRESHOLD
        self.image_threshold = IMAGE_NSFW_THRESHOLD
        self.ocr_languages = OCR_LANGUAGES
        self.default_fps = DEFAULT_FPS

class TextConfig:
    def __init__(self, content_config: ContentConfig):
        self.model_name = content_config.text_model_name
        self.threshold = content_config.text_threshold
        self.results_dir = content_config.text_results_dir

class ImageConfig:
    def __init__(self, content_config: ContentConfig):
        self.model_name = content_config.image_model_name
        self.threshold = content_config.image_threshold
        self.results_dir = content_config.image_results_dir

class OCRConfig:
    def __init__(self, content_config: ContentConfig):
        self.languages = content_config.ocr_languages
        self.results_dir = content_config.ocr_results_dir

class AudioConfig:
    def __init__(self, content_config: ContentConfig):
        self.model_name = content_config.whisper_model_name
        self.results_dir = content_config.audio_results_dir

class VideoConfig:
    def __init__(self, content_config: ContentConfig):
        self.fps = content_config.default_fps
        self.results_dir = content_config.video_results_dir
