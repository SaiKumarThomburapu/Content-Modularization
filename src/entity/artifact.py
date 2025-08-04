from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TextArtifact:
    classification_result: Dict[str, Any]

@dataclass
class ImageArtifact:
    classification_result: Dict[str, Any]

@dataclass
class OCRArtifact:
    extracted_text: str
    text_classification: Dict[str, Any]

@dataclass
class AudioArtifact:
    transcript: str
    transcript_classification: Dict[str, Any]

@dataclass
class VideoArtifact:
    frame_classifications: List[Dict[str, Any]]
    audio_transcript: str
    transcript_classification: Dict[str, Any]
