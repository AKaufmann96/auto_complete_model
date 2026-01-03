import pandas as pd
import re
import os
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Пути к данным
RAW_DATA_PATH = "data/raw_dataset.csv"
PROCESSED_DATA_PATH = "data/dataset_processed.csv"
TRAIN_DATA_PATH = "data/train.csv"
VAL_DATA_PATH = "data/val.csv"
TEST_DATA_PATH = "data/test.csv"

def clean_text(text: str) -> str:
    """
    Очищает текст: приводит к нижнему регистру, удаляет спецсимволы, оставляет только кириллицу, латиницу и пробелы.
    """
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r'[^а-яёa-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_clean_data() -> pd.DataFrame:
    """
    Загружает сырой датасет, очищает текст и возвращает DataFrame.
    """
    print("Загрузка и очистка данных...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Предполагаем, что в датасете есть колонка 'text' с текстами
    if 'text' not in df.columns:
        raise ValueError("В датасете отсутствует колонка 'text'. Проверьте структуру raw_dataset.csv.")
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 0]  # Удаляем пустые тексты
    df = df[['cleaned_text']].rename(columns={'cleaned_text': 'text'})
    
    print(f"Обработано {len(df)} текстов.")
    return df

def tokenize_and_save(df: pd.DataFrame):
    """
    Токенизирует тексты (простое разбиение по пробелам) и сохраняет обработанный датасет.
    """
    print("Токенизация текстов...")
    df['tokens'] = df['text'].apply(lambda x: x.split())
    df = df[['text']]  # Сохраняем только очищенный текст
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Обработанный датасет сохранён в {PROCESSED_DATA_PATH}")

def split_and_save_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.5):
    """
    Разбивает данные на train, val, test и сохраняет в соответствующие файлы.
    """
    print("Разбиение на train/val/test...")
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    val, test = train_test_split(test, test_size=val_size, random_state=42)
    
    train_val.to_csv(TRAIN_DATA_PATH, index=False)
    val.to_csv(VAL_DATA_PATH, index=False)
    test.to_csv(TEST_DATA_PATH, index=False)
    
    print(f"Размер обучающей выборки: {len(train_val)}")
    print(f"Размер валидационной выборки: {len(val)}")
    print(f"Размер тестовой выборки: {len(test)}")
    print("Разбиение завершено.")

def main():
    """
    Основная функция для выполнения этапа подготовки данных.
    """
    if not os.path.exists("data"):
        raise FileNotFoundError("Папка data не найдена. Убедитесь, что структура проекта создана корректно.")
    
    df = load_and_clean_data()
    tokenize_and_save(df)
    split_and_save_data(df)

if __name__ == "__main__":
    main()