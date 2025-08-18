"""
Тонкий слой для работы с моделями.
Вся реализация (кэш, загрузка, батч-энкод) лежит в utils.py, чтобы избежать дублирования.
Этот модуль оставлен для удобной будущей расширяемости (например, multimodal.py).
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# реэкспортируем функции из utils
from utils import load_model_from_source, encode_texts_in_batches

__all__ = [
    "load_model_from_source",
    "encode_texts_in_batches",
    "SentenceTransformer",
]
