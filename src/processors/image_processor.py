from PIL import Image
from pathlib import Path
import logging


class ImageProcessor:
    """Process images for captioning."""
    
    def __init__(self, captioner):
        self.captioner = captioner
        self.logger = logging.getLogger(__name__)
    
    def process(self, file_path: Path) -> str:
        """Process a single image file."""
        try:
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                return self.captioner.caption(img)
                
        except Exception as e:
            self.logger.error(f"Failed to process image {file_path}: {e}")
            raise