import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Optional

# Пути к данным
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"
VOCAB_PATH = "models/vocab.pkl"

# Параметры
MAX_VOCAB_SIZE = 10000
MIN_FREQ = 2
MAX_LENGTH = 128  # Максимальная длина последовательности (вход + сдвиг)


class NextTokenDataset(Dataset):
    """
    Кастомный Dataset для задачи языковой модели: предсказание следующего токена.
    Каждый элемент — (входная последовательность, целевая последовательность),
    где целевая — это сдвиг входной на один токен вперёд.
    Подходит для обучения LSTM и оценки через ROUGE при генерации.
    """

    def __init__(
        self,
        texts: List[str],
        vocab: Dict[str, int] = None,
        max_length: int = MAX_LENGTH,
        create_vocab: bool = False
    ):
        """
        :param texts: Список очищенных текстов.
        :param vocab: Словарь (слово -> индекс). Если None и create_vocab=True — строится новый.
        :param max_length: Максимальная длина последовательности (до обрезки).
        :param create_vocab: Флаг — нужно ли построить и сохранить словарь.
        """
        self.max_length = max_length
        self.tokenized_texts = []

        # Предварительная токенизация и фильтрация коротких текстов
        for text in texts:
            tokens = text.strip().split()
            if len(tokens) >= 2:  # Требуем минимум 2 токена для пары (x, y)
                self.tokenized_texts.append(tokens[:max_length + 1])  # +1 для target

        if create_vocab:
            self.vocab = self._build_vocab([t for t in self.tokenized_texts if len(t) >= 2])
            self._save_vocab(self.vocab)
        else:
            self.vocab = vocab if vocab is not None else self._load_vocab()

        self.pad_idx = self.vocab["<PAD>"]
        self.unk_idx = self.vocab["<UNK>"]

    def _build_vocab(self, tokenized_texts: List[List[str]]) -> Dict[str, int]:
        """Строит словарь по частоте."""
        print("Построение словаря...")
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, freq in counter.most_common():
            if freq < MIN_FREQ:
                continue
            if len(vocab) >= MAX_VOCAB_SIZE:
                break
            vocab[word] = len(vocab)

        print(f"Размер словаря: {len(vocab)}")
        return vocab

    def _save_vocab(self, vocab: Dict[str, int]):
        """Сохраняет словарь."""
        os.makedirs("models", exist_ok=True)
        with open(VOCAB_PATH, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Словарь сохранён в {VOCAB_PATH}")

    def _load_vocab(self) -> Dict[str, int]:
        """Загружает словарь."""
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(
                f"Файл словаря {VOCAB_PATH} не найден. "
                "Постройте словарь, запустив датасет с create_vocab=True на train."
            )
        with open(VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)
        print(f"Словарь загружен из {VOCAB_PATH}")
        return vocab

    def __len__(self) -> int:
        return len(self.tokenized_texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает:
            input_ids: тензор [T], входная последовательность (x_1, ..., x_{T-1})
            target_ids: тензор [T], целевая последовательность (x_2, ..., x_T)
        Длина T = min(len(tokens), max_length)
        """
        tokens = self.tokenized_texts[idx]

        # Вход: все токены, кроме последнего
        input_tokens = tokens[:-1]
        # Цель: все токены, начиная со второго (сдвиг на 1)
        target_tokens = tokens[1:]

        # Преобразуем в индексы
        input_ids = [self.vocab.get(token, self.unk_idx) for token in input_tokens]
        target_ids = [self.vocab.get(token, self.unk_idx) for token in target_tokens]

        # Паддинг справа до max_length
        seq_len = len(input_ids)
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids += [self.pad_idx] * pad_len
            target_ids += [self.pad_idx] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            target_ids = target_ids[:self.max_length]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )


def collate_fn(batch):
    """
    Динамическое паддинг (не требуется при фиксированном паддинге справа, но оставлен для гибкости).
    Сейчас используется фиксированный размер, так что можно без collate_fn.
    """
    return torch.utils.data.default_collate(batch)


def get_dataloader(
    split: str = "train",
    batch_size: int = 64,
    max_length: int = MAX_LENGTH,
    num_workers: int = 0
) -> DataLoader:
    """
    Возвращает DataLoader для указанной выборки.
    При первом запуске (train) строится словарь. В последующих — загружается.
    """
    assert split in ["train", "val", "test"], "split должен быть 'train', 'val' или 'test'"

    path = {
        "train": TRAIN_PATH,
        "val": VAL_PATH,
        "test": TEST_PATH
    }[split]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_csv(path)
    texts = df["text"].tolist()

    vocab = None
    create_vocab = False

    if split == "train":
        create_vocab = True
    else:
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, "rb") as f:
                vocab = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Словарь не найден при загрузке {split}-данных. "
                "Убедитесь, что сначала был запущен train-даталоадер для построения словаря."
            )

    dataset = NextTokenDataset(
        texts=texts,
        vocab=vocab,
        max_length=max_length,
        create_vocab=create_vocab
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=(split == "train"),
        collate_fn=collate_fn
    )

    print(f"Даталоадер для {split} создан: {len(dataset)} примеров, батч={batch_size}")
    return dataloader