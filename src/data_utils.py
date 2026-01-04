import pandas as pd
import re
import os
from typing import Tuple
from sklearn.model_selection import train_test_split

# Пути к данным
RAW_DATA_PATH = "data/raw_dataset.csv"
PROCESSED_DATA_PATH = "data/dataset_processed.csv"
TRAIN_DATA_PATH = "data/train.csv"
VAL_DATA_PATH = "data/val.csv"
TEST_DATA_PATH = "data/test.csv"

# Параметры обработки
MAX_LEN = 128  # Максимальная длина текста в токенах


def clean_text(text: str) -> str:
    """
    Очищает и нормализует англоязычный текст твита:
    - Приводит к нижнему регистру.
    - Удаляет URL и email.
    - Сохраняет @mentions и #hashtags как токены (без символов @ и #).
    - Сохраняет апострофы в сокращениях (например, "don't", "it's").
    - Оставляет только буквенно-цифровые символы, пробелы и апострофы.
    - Удаляет лишние пробелы.
    - Обрезает до MAX_LEN токенов.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Удаление URL и email
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URL
    text = re.sub(r'\S+@\S+', '', text)                # Email

    # Сохранение хэштегов и упоминаний (оставляем только имя)
    text = re.sub(r'#(\w+)', r'\1', text)   # #hello → hello
    text = re.sub(r'@(\w+)', r'\1', text)   # @user → user

    # Оставляем только допустимые символы: буквы, цифры, пробелы, апостроф
    text = re.sub(r"[^a-z0-9\s']", '', text)

    # Замена множественных пробелов на один
    text = re.sub(r'\s+', ' ', text).strip()

    # Обрезка до максимальной длины
    tokens = text.split()[:MAX_LEN]
    text = ' '.join(tokens)

    return text


def load_and_clean_data() -> pd.DataFrame:
    """
    Загружает сырой датасет, очищает текст и возвращает DataFrame.
    """
    print("Загрузка и очистка данных...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Файл {RAW_DATA_PATH} не найден. Убедитесь, что датасет загружен.")

    df = pd.read_csv(RAW_DATA_PATH)

    if 'text' not in df.columns:
        raise ValueError("В датасете отсутствует колонка 'text'. Проверьте структуру raw_dataset.csv.")

    # Очистка текста
    df['text'] = df['text'].apply(clean_text)

    # Удаление пустых строк
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    print(f"Обработано и сохранено {len(df)} текстов после очистки.")

    return df


def tokenize_and_save(df: pd.DataFrame):
    """
    Выполняет токенизацию (уже частично сделана в clean_text) и сохраняет очищенные тексты.
    """
    print("Сохранение обработанного датасета...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Обработанный датасет сохранён в {PROCESSED_DATA_PATH}")


def split_and_save_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.5):
    """
    Разбивает данные на train, val, test в пропорции:
    train: 80% * (1 - test_size) = 80%
    val: 20% * val_size = 10%
    test: 20% * (1 - val_size) = 10%
    При test_size=0.2, val_size=0.5 → 80%/10%/10%
    """
    print("Разбиение на train/val/test...")

    train, temp = train_test_split(df, test_size=test_size, random_state=42)
    val, test = train_test_split(temp, test_size=val_size, random_state=42)

    train.to_csv(TRAIN_DATA_PATH, index=False)
    val.to_csv(VAL_DATA_PATH, index=False)
    test.to_csv(TEST_DATA_PATH, index=False)

    print(f"Размер обучающей выборки: {len(train)}")
    print(f"Размер валидационной выборки: {len(val)}")
    print(f"Размер тестовой выборки: {len(test)}")
    print("Разбиение завершено.")


def main():
    """
    Основная функция для выполнения этапа подготовки данных.
    """
    if not os.path.exists("data"):
        raise FileNotFoundError("Папка 'data' не найдена. Убедитесь, что структура проекта создана корректно.")

    df = load_and_clean_data()
    tokenize_and_save(df)
    split_and_save_data(df)


if __name__ == "__main__":
    main()