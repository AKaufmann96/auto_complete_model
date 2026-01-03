import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from collections import Counter
import pickle
from typing import List, Tuple

# Пути к данным
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"
VOCAB_PATH = "models/vocab.pkl"
MAX_VOCAB_SIZE = 10000
MIN_FREQ = 2

class NextTokenDataset(Dataset):
    """
    Кастомный Dataset для задачи предсказания следующего токена.
    Каждый элемент — (последовательность индексов токенов, индекс следующего токена).
    """

    def __init__(self, texts: List[str], vocab=None, max_length: int = 128, create_vocab: bool = False):
        """
        :param texts: Список текстов (очищенных).
        :param vocab: Словарь (слово -> индекс). Если None и create_vocab=True — строится новый.
        :param max_length: Максимальная длина входной последовательности.
        :param create_vocab: Флаг — нужно ли строить словарь.
        """
        self.texts = texts
        self.max_length = max_length
        self.tokenized_texts = [text.split() for text in texts]

        if create_vocab:
            self.vocab = self._build_vocab(self.tokenized_texts)
            self._save_vocab(self.vocab)
        else:
            self.vocab = vocab if vocab is not None else self._load_vocab()

        self.pad_idx = self.vocab["<PAD>"]
        self.unk_idx = self.vocab["<UNK>"]

    def _build_vocab(self, tokenized_texts: List[List[str]]) -> dict:
        """
        Строит словарь по частоте встречаемости токенов.
        """
        print("Построение словаря...")
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        # Сортируем по частоте
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, freq in counter.most_common():
            if freq < MIN_FREQ:
                continue
            if len(vocab) >= MAX_VOCAB_SIZE:
                break
            vocab[word] = len(vocab)

        print(f"Размер словаря: {len(vocab)}")
        return vocab

    def _save_vocab(self, vocab: dict):
        """Сохраняет словарь в файл."""
        os.makedirs("models", exist_ok=True)
        with open(VOCAB_PATH, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Словарь сохранён в {VOCAB_PATH}")

    def _load_vocab(self) -> dict:
        """Загружает словарь из файла."""
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(f"Файл словаря не найден: {VOCAB_PATH}. Постройте словарь с create_vocab=True.")
        with open(VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)
        print(f"Словарь загружен из {VOCAB_PATH}")
        return vocab

    def __len__(self) -> int:
        return len(self.tokenized_texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает:
            input_ids: тензор индексов длины max_length (обрезанной или дополненной)
            target_id: индекс следующего токена (целевого)
        """
        tokens = self.tokenized_texts[idx]

        # Обрезаем, если слишком длинно
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Если текст слишком короткий — пропускаем
        if len(tokens) < 2:
            # Возвращаем пустую последовательность и PAD как таргет
            input_ids = torch.full((self.max_length,), self.pad_idx, dtype=torch.long)
            target_id = torch.tensor(self.pad_idx, dtype=torch.long)
            return input_ids, target_id

        # Вход: все токены, кроме последнего
        input_tokens = tokens[:-1]
        # Цель: последний токен
        target_token = tokens[-1]

        # Преобразуем в индексы
        input_ids = [self.vocab.get(token, self.unk_idx) for token in input_tokens]

        # Дополняем до max_length
        padding = [self.pad_idx] * (self.max_length - len(input_ids))
        input_ids = padding + input_ids  # Паддинг слева

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_id = torch.tensor(self.vocab.get(target_token, self.unk_idx), dtype=torch.long)

        return input_ids, target_id


def get_dataloader(split: str = "train", batch_size: int = 64, max_length: int = 128, num_workers: int = 0):
    """
    Возвращает DataLoader для указанной выборки.
    """
    assert split in ["train", "val", "test"], "split должен быть 'train', 'val' или 'test'"

    path = {
        "train": TRAIN_PATH,
        "val": VAL_PATH,
        "test": TEST_PATH
    }[split]

    df = pd.read_csv(path)
    texts = df["text"].tolist()

    vocab = None
    create_vocab = False

    if split == "train":
        create_vocab = True  # Только при создании train-датасета строим словарь
    else:
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, "rb") as f:
                vocab = pickle.load(f)

    dataset = NextTokenDataset(
        texts=texts,
        vocab=vocab,
        max_length=max_length,
        create_vocab=create_vocab
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=(split == "train")  # Убираем неполные батчи для стабильности обучения
    )

    print(f"Даталоадер для {split} создан: {len(dataset)} примеров, батч={batch_size}")
    return dataloader