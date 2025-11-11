from PIL import Image
from typing import List, Dict
from app.encoder import Encoder
from app.utils import logger
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import json

class FeatureExtractor:
    def __init__(self):
        self.encoder = Encoder()
        # load VQA model (BLIP2 small recommended for lower VRAM)
        try:
            self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-small")
            self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-small").to(self.encoder.device)
            logger.info("Loaded VQA model on %s", self.encoder.device)
        except Exception as e:
            logger.warning("VQA load failed: %s — disabling VQA", e)
            self.vqa_processor = None
            self.vqa_model = None

    def load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            logger.error("Open image fail %s: %s", path, e)
            return None

    def vqa_answer(self, image, question, max_new_tokens=24):
        if self.vqa_model is None:
            return ""
        try:
            inputs = self.vqa_processor(images=image, text=question, return_tensors="pt").to(self.encoder.device)
            with torch.no_grad():
                out = self.vqa_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
            ans = self.vqa_processor.decode(out[0], skip_special_tokens=True).strip().lower()
            return ans
        except Exception as e:
            logger.warning("VQA failed: %s", e)
            return ""

    def extract_metadata(self, pil_image, questions: List[str]):
        meta = {}
        for q in questions:
            ans = self.vqa_answer(pil_image, q)
            meta[q] = ans
        return meta

    def encode_image(self, pil_images: List):
        return self.encoder.encode_images(pil_images)

    def encode_texts(self, texts: List[str]):
        return self.encoder.encode_text(texts)

extractor = FeatureExtractor()
