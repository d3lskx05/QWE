"""
Модуль для мультимодальных моделей (текст + изображение).
Поддержка: CLIP, SigLIP (text↔image), BLIP, BLIP-2 (image→caption).
Источник: HuggingFace или Google Drive.
"""

import streamlit as st
import torch
from typing import List
from PIL import Image
import numpy as np
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq
)
from utils import download_file_from_gdrive


# ================== Вспомогательная функция ==================
def _resolve_model_path(source: str, identifier: str) -> str:
    """
    Возвращает путь или идентификатор модели для from_pretrained:
      - huggingface -> строка (ID)
      - google_drive -> локальный путь к извлеченной папке
    """
    if source == "huggingface":
        return identifier
    elif source == "google_drive":
        return download_file_from_gdrive(identifier)
    else:
        raise ValueError(f"Неизвестный источник модели: {source}")


# ================== CLIP / SigLIP ==================
@st.cache_resource(show_spinner=False)
def load_clip_model(source: str = "huggingface", model_id: str = "openai/clip-vit-base-patch32"):
    """
    Загрузка CLIP или SigLIP модели.
    """
    model_path = _resolve_model_path(source, model_id)
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor


def encode_texts(model, processor, texts: List[str], batch_size: int = 16) -> np.ndarray:
    """Эмбеддинги текстов CLIP/SigLIP с batching."""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        embs.append(out.cpu().numpy())
    return np.vstack(embs)


def encode_images(model, processor, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
    """Эмбеддинги изображений CLIP/SigLIP с batching."""
    embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        embs.append(out.cpu().numpy())
    return np.vstack(embs)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def check_text_image_pair(model, processor, text: str, image: Image.Image) -> float:
    """Сходство между текстом и изображением через CLIP/SigLIP."""
    text_emb = encode_texts(model, processor, [text])[0]
    image_emb = encode_images(model, processor, [image])[0]
    return cosine_similarity(text_emb, image_emb)


# ================== BLIP / BLIP-2 ==================
@st.cache_resource(show_spinner=False)
def load_blip_model(source: str = "huggingface", model_id: str = "Salesforce/blip-image-captioning-base"):
    """
    Загрузка BLIP или BLIP-2.
    """
    model_path = _resolve_model_path(source, model_id)

    if "blip2" in model_id.lower():
        # BLIP-2 (vision-to-seq)
        model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.float32)
        processor = AutoProcessor.from_pretrained(model_path)
    else:
        # Классический BLIP
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        processor = BlipProcessor.from_pretrained(model_path)

    return model, processor


def generate_caption(model, processor, image: Image.Image, max_length: int = 30) -> str:
    """Генерация описания картинки через BLIP или BLIP-2."""
    if isinstance(processor, BlipProcessor):
        inputs = processor(image, return_tensors="pt")
    else:
        # BLIP-2 (AutoProcessor)
        inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_length)

    return processor.decode(out[0], skip_special_tokens=True)
