import numpy as np
import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModel,
    BlipProcessor, BlipForConditionalGeneration
)

# ===================== CLIP / SigLIP =====================
def load_clip_model(provider: str, model_id: str):
    """
    Загрузка CLIP/SigLIP модели.
    """
    try:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        return model, processor
    except Exception:
        model = AutoModel.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor


def encode_texts(model, processor, texts):
    """
    Текст → эмбеддинги CLIP/SigLIP
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    return out.cpu().numpy()


def encode_images(model, processor, images):
    """
    Картинки → эмбеддинги CLIP/SigLIP
    """
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    return out.cpu().numpy()

# ===================== BLIP / BLIP-2 =====================
def load_blip_model(provider: str, model_id: str):
    """
    Загрузка BLIP или BLIP-2 модели.
    """
    if "blip2" in model_id.lower():
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            model = Blip2ForConditionalGeneration.from_pretrained(model_id)
            processor = Blip2Processor.from_pretrained(model_id)
        except Exception:
            raise ImportError("Для BLIP-2 нужна версия transformers >= 4.39")
    else:
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        processor = BlipProcessor.from_pretrained(model_id)
    return model, processor


def generate_caption(model, processor, image):
    """
    Сгенерировать подпись к картинке через BLIP/BLIP-2
    """
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

# ===================== Similarity =====================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Косинусное сходство для двух векторов
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Косинусное сходство для матриц эмбеддингов (батч).
    a: (N, D), b: (M, D)
    return: (N, M)
    """
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.matmul(a_norm, b_norm.T)
