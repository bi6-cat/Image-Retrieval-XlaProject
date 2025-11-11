from app.config import settings
from app.utils import l2norm_np, logger
import numpy as np
import torch

def load_hf_clip(device=None):
    from transformers import CLIPProcessor, CLIPModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

def load_st_model(name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name)
    return model

class Encoder:
    def __init__(self, backend=None):
        self.backend = backend or settings.ENCODER_BACKEND
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Encoder backend=%s device=%s", self.backend, self.device)
        if self.backend == "sentence_transformers":
            # adjust model id as needed
            self.model = load_st_model("clip-ViT-B-32")
        else:
            self.model, self.processor, self.device = load_hf_clip(self.device)

    def encode_text(self, texts):
        if self.backend == "sentence_transformers":
            emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return l2norm_np(np.array(emb))
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model.get_text_features(**inputs)
            emb = out.cpu().numpy()
            return l2norm_np(np.array(emb))

    def encode_images(self, pil_images):
        # pil_images: list of PIL.Image
        if self.backend == "sentence_transformers":
            try:
                emb = self.model.encode(pil_images, convert_to_numpy=True, show_progress_bar=False)
                return l2norm_np(np.array(emb))
            except Exception as e:
                logger.warning("ST encode images failed: %s; falling back to HF CLIP", e)
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
        emb = out.cpu().numpy()
        return l2norm_np(np.array(emb))
