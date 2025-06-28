import cv2
from PIL import Image
from pathlib import Path
import logging
from typing import List


class VideoProcessor:
    """Process videos and GIFs for captioning."""
    
    def __init__(self, captioner, max_frames: int = 5):
        self.captioner = captioner
        self.max_frames = max_frames
        self.logger = logging.getLogger(__name__)
    
    def process(self, file_path: Path) -> str:
        """Process a video file by captioning key frames."""
        try:
            frames = self._extract_frames(file_path)
            if not frames:
                return "No frames extracted from video"
            
            captions = []
            for i, frame in enumerate(frames):
                caption = self.captioner.caption(frame)
                captions.append(caption)
                self.logger.debug(f"Frame {i+1}: {caption}")
            
            return " | ".join(captions)
            
        except Exception as e:
            self.logger.error(f"Failed to process video {file_path}: {e}")
            raise
    
    def _extract_frames(self, file_path: Path) -> List[Image.Image]:
        """Extract key frames from video."""

        video = cv2.VideoCapture(str(file_path))
        
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {file_path}")
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            video.release()
            return []
        
        if total_frames <= self.max_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // self.max_frames
            frame_indices = [i * step for i in range(self.max_frames)]
        
        frames = []
        for frame_index in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        
        video.release()
        return frames