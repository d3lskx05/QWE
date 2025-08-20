"""
Модуль для мультимодальных моделей (текст + изображение).
Поддержка: CLIP (text↔image), BLIP (image→caption).
Источник: HuggingFace или Google Drive.
"""

import streamlit as st
import torch
from typing import List
from PIL import Image
import numpy as np
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
from utils import download_file_from_gdrive

# Определяем устройство (GPU, если доступно)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== Вспомогательная функция ==================
def _resolve_model_path(source: str, identifier: str) -> str:
    """Возвращает путь или идентификатор модели для from_pretrained"""
    if source == "huggingface":
        return identifier
    elif source == "google_drive":
        return download_file_from_gdrive(identifier)
    else:
        raise ValueError(f"Неизвестный источник модели: {source}")


# ================== CLIP ==================
@st.cache_resource(show_spinner=False)
def load_clip_model(source: str = "huggingface", model_id: str = "openai/clip-vit-base-patch32"):
    """Загрузка CLIP модели с выбором источника"""
    model_path = _resolve_model_path(source, model_id)
    model = CLIPModel.from_pretrained(model_path).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor


def encode_texts(model, processor, texts: List[str]) -> np.ndarray:
    """Эмбеддинги текстов CLIP."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_embs = model.get_text_features(**inputs)
    return text_embs.cpu().numpy()


def encode_images(model, processor, images: List[Image.Image]) -> np.ndarray:
    """Эмбеддинги изображений CLIP."""
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_embs = model.get_image_features(**inputs)
    return image_embs.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def check_text_image_pair(model, processor, text: str, image: Image.Image) -> float:
    """Сходство между текстом и изображением через CLIP."""
    text_emb = encode_texts(model, processor, [text])[0]
    image_emb = encode_images(model, processor, [image])[0]
    return cosine_similarity(text_emb, image_emb)


# ================== BLIP ==================
@st.cache_resource(show_spinner=False)
def load_blip_model(source: str = "huggingface", model_id: str = "Salesforce/blip-image-captioning-base"):
    """Загрузка BLIP для генерации описаний картинок."""
    model_path = _resolve_model_path(source, model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
    processor = BlipProcessor.from_pretrained(model_path)
    return model, processor


def generate_caption(model, processor, image: Image.Image, max_length: int = 30) -> str:
    """Генерация описания картинки через BLIP."""
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_length)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
