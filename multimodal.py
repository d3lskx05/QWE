"""
Модуль для мультимодальных моделей (текст + изображение).
Поддержка бесплатных open-source моделей: CLIP, BLIP.
"""

import streamlit as st
import torch
from typing import List, Tuple
from PIL import Image
import numpy as np

from transformers import CLIPProcessor, CLIPModel

# ================== Загрузка модели ==================

@st.cache_resource(show_spinner=False)
def load_clip_model(model_id: str = "openai/clip-vit-base-patch32"):
    """
    Загружает CLIP-модель (по умолчанию openai/clip-vit-base-patch32).
    Кэшируем через Streamlit.
    """
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

# ================== Энкодинг ==================

def encode_texts(model, processor, texts: List[str]) -> np.ndarray:
    """
    Кодирует список текстов в эмбеддинги CLIP.
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_embs = model.get_text_features(**inputs)
    return text_embs.cpu().numpy()

def encode_images(model, processor, images: List[Image.Image]) -> np.ndarray:
    """
    Кодирует список изображений в эмбеддинги CLIP.
    """
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        image_embs = model.get_image_features(**inputs)
    return image_embs.cpu().numpy()

# ================== Сходство ==================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

# ================== Утилиты ==================

def check_text_image_pair(model, processor, text: str, image: Image.Image) -> float:
    """
    Вычисляет косинусное сходство между текстом и изображением.
    """
    text_emb = encode_texts(model, processor, [text])[0]
    image_emb = encode_images(model, processor, [image])[0]
    return cosine_similarity(text_emb, image_emb)
