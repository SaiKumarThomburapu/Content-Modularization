import os

# Root artifact directory
ARTIFACTS_DIR = "artifacts"

# Subdirectories
TEXT_RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "text_classifications")
IMAGE_RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "image_classifications")
OCR_RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "ocr_extractions")
AUDIO_RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "audio_transcriptions")
VIDEO_RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "video_frame_extractions")

# Model names
TEXT_MODEL_NAME = "textdetox/xlmr-large-toxicity-classifier-v2"
IMAGE_MODEL_NAME = "hf_hub:Marqo/nsfw-image-detection-384"
WHISPER_MODEL_NAME = "base"

# Thresholds
TEXT_TOXICITY_THRESHOLD = 0.5
IMAGE_NSFW_THRESHOLD = 0.5

# OCR settings
OCR_LANGUAGES = ["en"]

# File size limits
MAX_FILE_SIZE = 50_000_000
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Default processing parameters
DEFAULT_FPS = 1
