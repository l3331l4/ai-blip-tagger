from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import logging


class ImageCaptioner:
    """BLIP model wrapper for image captioning."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", use_fast: bool = False):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading BLIP model...")
        
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=use_fast)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        self.device = self._select_device()
        self.model.to(self.device)
        
        self.logger.info(f"Model loaded on {self.device}")
    
    def _select_device(self) -> str:
        """Select the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def caption(self, image: Image.Image) -> str:
        """Generate caption for a PIL Image."""
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=5, repetition_penalty=1.2, no_repeat_ngram_size=2)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption.strip()
            
        except Exception as e:
            self.logger.error(f"Caption generation failed: {e}")
            return "Caption generation failed"