import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np
import os


class LSTMModel(nn.Module):
    """
    LSTM-модель для предсказания следующего токена.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        """
        :param vocab_size: Размер словаря.
        :param embedding_dim: Размерность эмбеддингов.
        :param hidden_dim: Размерность скрытого состояния LSTM.
        :param num_layers: Количество слоёв LSTM.
        :param dropout: Вероятность дропаута.
        :param pad_idx: Индекс паддинга (для корректной работы с масками).
        """
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        """Инициализация весов."""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход.
        :param input_ids: Тензор формы (batch_size, seq_len)
        :return: Логиты формы (batch_size, vocab_size)
        """
        # Создаём маску для паддинга
        mask = (input_ids != self.pad_idx).any(dim=1)  # (batch_size,) — True, если не весь паддинг

        # Эмбеддинги
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, emb_dim)

        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)  # hidden: (num_layers, batch_size, hidden_dim)

        # Используем последнее скрытое состояние
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Применяем маску: если вся последовательность — паддинг, возвращаем нули
        last_hidden = last_hidden * mask.unsqueeze(1).float()  # обнуляем, если маска False

        # Дропаут и полносвязный слой
        output = self.dropout(last_hidden)
        logits = self.fc(output)  # (batch_size, vocab_size)

        return logits

    def generate(
        self,
        start_text: str,
        vocab: dict,
        reverse_vocab: dict,
        max_length: int = 50,
        temperature: float = 1.0,
        device: str = "cpu"
    ) -> str:
        """
        Генерация продолжения текста.
        
        :param start_text: Начало текста.
        :param vocab: Словарь (слово -> индекс).
        :param reverse_vocab: Обратный словарь (индекс -> слово).
        :param max_length: Максимальное число генерируемых токенов.
        :param temperature: Температура для сэмплирования.
        :param device: Устройство ('cpu' или 'cuda').
        :return: Сгенерированное продолжение (только новая часть).
        """
        self.eval()
        with torch.no_grad():
            # Подготавливаем входной текст
            tokens = start_text.lower().split()
            if not tokens:
                return ""

            # Преобразуем в индексы
            input_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
            
            seq_len = 128

            # Ограничиваем длину входа, чтобы влезло место для генерации
            if len(input_ids) >= seq_len:
                input_ids = input_ids[-(seq_len - 1):]  # оставляем место для новых токенов

            # Список для всех токенов (вход + продолжение)
            generated_tokens = tokens.copy()

            for _ in range(max_length):
                # Берём последние seq_len токенов
                current_input = input_ids[-seq_len:]
                input_tensor = torch.tensor([current_input], dtype=torch.long).to(device)

                # Прямой проход
                logits = self.forward(input_tensor)  # (1, vocab_size)

                # Температура и вероятности
                logits = logits / max(temperature, 1e-9)
                probs = F.softmax(logits, dim=-1)[0]  # (vocab_size,)

                # Блокируем <PAD> и <UNK>
                probs[vocab["<PAD>"]] = 0
                if vocab.get("<UNK>") is not None:
                    probs[vocab["<UNK>"]] = 0

                # Проверка на валидные токены
                if probs.sum() <= 1e-8:
                    break  # нет куда генерировать

                probs = probs / probs.sum()

                # Сэмплируем следующий токен
                next_token_idx = torch.multinomial(probs, 1).item()

                # Прерываем, если <PAD>
                if next_token_idx == vocab["<PAD>"]:
                    break

                # Добавляем в последовательность
                input_ids.append(next_token_idx)
                next_token = reverse_vocab.get(next_token_idx, "<UNK>")
                generated_tokens.append(next_token)

            # Возвращаем ТОЛЬКО продолжение
            prefix_length = len(tokens)
            completion = " ".join(generated_tokens[prefix_length:])
            return completion.strip()

    def save(self, path: str):
        """Сохраняет модель."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Модель сохранена в {path}")

    def load(self, path: str, device: str = "cpu"):
        """Загружает модель."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Модель загружена из {path}")