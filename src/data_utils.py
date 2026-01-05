# Импорт необходимых библиотек
import torch
import torch.nn as nn
import re
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Пути к данным
RAW_DATA_PATH = "data/raw_dataset.csv"
PROCESSED_DATA_PATH = "data/dataset_processed.csv"
TRAIN_DATA_PATH = "data/train.csv"
VAL_DATA_PATH = "data/val.csv"
TEST_DATA_PATH = "data/test.csv"

# Создание директории, если её нет
os.makedirs("data", exist_ok=True)

# Функция очистки текста твитов
def clean_string(text):
    """
    Очищает текст твита:
    - удаляет @упоминания, #хештеги, ссылки;
    - оставляет только латинские буквы, цифры и пробелы;
    - нормализует пробелы.
    """
    text = text.lower()
    # Удаление упоминаний (@username)
    text = re.sub(r'@\w+', '', text)
    # Удаление хештегов целиком (#example → удалить)
    text = re.sub(r'#\w+', '', text)
    # Удаление URL (http://, https://, www.)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Удаление всех символов, кроме латинских букв, цифр и пробелов
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Нормализация пробелов: сжатие нескольких пробелов в один, обрезка краёв
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Длина последовательности (3 токена до, [MASK], 3 токена после)
seq_len = 7

# Загрузка датасета из CSV
print("Загрузка датасета из файла:", RAW_DATA_PATH)
try:
    df = pd.read_csv(RAW_DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Файл не найден по пути: {RAW_DATA_PATH}")
except Exception as e:
    raise Exception(f"Ошибка при чтении CSV-файла: {e}")

# Проверка наличия колонки 'text'
if 'text' not in df.columns:
    raise ValueError(f"Ожидалась колонка 'text', но в файле присутствуют: {list(df.columns)}")

# Извлечение и очистка текстов
texts = df['text'].dropna().tolist()
print(f"Загружено текстов: {len(texts)}")

# Очистка текстов
print("Очистка текстов от упоминаний, хештегов и ссылок...")
cleaned_texts = [clean_string(text) for text in texts]

# Фильтрация пустых и слишком коротких текстов
cleaned_texts = [text for text in cleaned_texts if len(text.split()) >= seq_len]
print(f"Текстов после очистки и фильтрации: {len(cleaned_texts)}")

# Ограничение объёма данных (по желанию)
max_texts_count = 7000
cleaned_texts = cleaned_texts[:max_texts_count]
print(f"Используется первых {len(cleaned_texts)} текстов.")

# Сохранение всех очищенных текстов
print("Сохранение обработанных данных...")
df_processed = pd.DataFrame(cleaned_texts, columns=["text"])
df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Обработанные данные сохранены в: {PROCESSED_DATA_PATH}")

# Разделение: сначала train (80%), затем val и test (по 10% от исходного)
print("Разделение на train/val/test (80%/10%/10%)...")

# Сначала выделим test (10%) из исходных данных
train_val_texts, test_texts = train_test_split(
    cleaned_texts,
    test_size=0.1,
    random_state=42
)

# Затем разделим оставшиеся на train (80%) и val (10% от общего)
train_texts, val_texts = train_test_split(
    train_val_texts,
    test_size=0.111,  # ~10% от 90% = 10% от общего
    random_state=42
)

# Сохранение выборок
def save_split(texts, path):
    df_split = pd.DataFrame(texts, columns=["text"])
    df_split.to_csv(path, index=False)
    print(f"Сохранено {len(texts)} текстов в {path}")

save_split(train_texts, TRAIN_DATA_PATH)
save_split(val_texts, VAL_DATA_PATH)
save_split(test_texts, TEST_DATA_PATH)

print(f"Размеры выборок: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

# Класс датасета для маскированного предсказания
class MaskedBertDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=7):
        self.samples = []
        for line in texts:
            # Токенизация без специальных токенов
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=512, truncation=True)
            if len(token_ids) < seq_len:
                continue
            # Генерация примеров: окно вокруг каждого токена
            for i in range(1, len(token_ids) - 1):
                start_idx = max(0, i - seq_len // 2)
                end_idx = i + 1 + seq_len // 2
                context = (
                    token_ids[start_idx:i] +
                    [tokenizer.mask_token_id] +
                    token_ids[i + 1:end_idx]
                )
                if len(context) != seq_len:
                    continue  # Пропуск, если длина не совпадает
                target = token_ids[i]
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# Загрузка токенизатора
print("Загрузка BERT-токенизатора...")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Создание датасетов
print("Формирование обучающего, валидационного и тестового датасетов...")
train_dataset = MaskedBertDataset(train_texts, tokenizer, seq_len=seq_len)
val_dataset = MaskedBertDataset(val_texts, tokenizer, seq_len=seq_len)
test_dataset = MaskedBertDataset(test_texts, tokenizer, seq_len=seq_len)

# Создание даталоадеров
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Вывод итоговой статистики
print(f"Размер обучающего датасета: {len(train_dataset)}")
print(f"Размер валидационного датасета: {len(val_dataset)}")
print(f"Размер тестового датасета: {len(test_dataset)}")
print("Подготовка данных завершена.")