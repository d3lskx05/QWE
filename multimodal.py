"""
Модуль для мультимодальных моделей (текст + изображение).
Поддержка: CLIP (text↔image), BLIP (image→caption).
Теперь добавлена возможность выбирать источник (HF или Google Drive).
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


# ================== CLIP ==================
@st.cache_resource(show_spinner=False)
def load_clip_model(source: str = "huggingface", model_id: str = "openai/clip-vit-base-patch32"):
    """
    Загрузка CLIP модели с выбором источника.
    source: huggingface | google_drive
    model_id: ID модели или File ID на Google Drive
    """
    model_path = _resolve_model_path(source, model_id)
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor


def encode_texts(model, processor, texts: List[str]) -> np.ndarray:
    """Эмбеддинги текстов CLIP."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_embs = model.get_text_features(**inputs)
    return text_embs.cpu().numpy()


def encode_images(model, processor, images: List[Image.Image]) -> np.ndarray:
    """Эмбеддинги изображений CLIP."""
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        image_embs = model.get_image_features(**inputs)
    return image_embs.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def check_text_image_pair(model, processor, text: str, image: Image.Image) -> float:
    """Сходство между текстом и изображением через CLIP."""
    text_emb = encode_texts(model, processor, [text])[0]
    image_emb = encode_images(model, processor, [image])[0]
    return cosine_similarity(text_emb, image_emb)


# ================== BLIP ==================
@st.cache_resource(show_spinner=False)
def load_blip_model(source: str = "huggingface", model_id: str = "Salesforce/blip-image-captioning-base"):
    """
    Загрузка BLIP для генерации описаний картинок.
    source: huggingface | google_drive
    model_id: ID модели или File ID на Google Drive
    """
    model_path = _resolve_model_path(source, model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_path)
    processor = BlipProcessor.from_pretrained(model_path)
    return model, processor


def generate_caption(model, processor, image: Image.Image, max_length: int = 30) -> str:
    """Генерация описания картинки через BLIP."""
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_length)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
