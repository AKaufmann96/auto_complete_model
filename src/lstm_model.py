"""
lstm_model.py

LSTM-модель для задачи языкового моделирования: предсказывает следующий токен.
Поддерживает обучение (teacher forcing) и авторегрессивную генерацию с кэшированием состояний.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import os


class LSTMModel(nn.Module):
    """
    LSTM-модель для языковой модели: предсказывает следующий токен на каждом шаге.
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
        :param vocab_size: размер словаря
        :param embedding_dim: размерность эмбеддингов
        :param hidden_dim: размер скрытого состояния LSTM
        :param num_layers: количество слоёв LSTM
        :param dropout: вероятность дропаута
        :param pad_idx: индекс токена <PAD>
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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Инициализация весов."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход.
        :param input_ids: (batch_size, seq_len)
        :return: логиты (batch_size, seq_len, vocab_size)
        """
        embedded = self.embedding(input_ids)  # (B, T, E)
        lstm_out, _ = self.lstm(embedded)    # (B, T, H)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)           # (B, T, V)
        return logits

    def generate(
        self,
        start_text: str,
        vocab: Dict[str, int],
        reverse_vocab: Dict[int, str],
        max_gen_tokens: int = 15,
        context_length: int = 50,
        temperature: float = 1.0,
        device: str = "cpu"
    ) -> str:
        """
        Генерация продолжения текста с кэшированием скрытых состояний LSTM.

        :param start_text: начальный текст (первые 3/4)
        :param vocab: словарь (токен → индекс)
        :param reverse_vocab: обратный словарь (индекс → токен)
        :param max_gen_tokens: максимальное число токенов для генерации
        :param context_length: сколько токенов из контекста использовать (берутся последние)
        :param temperature: температура для сэмплирования
        :param device: устройство ('cpu' или 'cuda')
        :return: сгенерированное продолжение
        """
        self.eval()
        with torch.no_grad():
            tokens = start_text.strip().lower().split()
            if not tokens:
                return ""

            # Преобразуем в индексы
            input_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
            input_ids = input_ids[-context_length:]  # ограничиваем контекст

            # Инициализация скрытых состояний
            h_0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=device)
            c_0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=device)

            # Прогоняем контекст (без генерации)
            context_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            embedded = self.embedding(context_tensor)
            _, (h_n, c_n) = self.lstm(embedded, (h_0, c_0))  # (num_layers, 1, hidden_dim)

            # Текущий токен для генерации — последний из контекста
            current_token_idx = input_ids[-1]
            generated_tokens = []

            for _ in range(max_gen_tokens):
                # Вводим только последний токен
                current_tensor = torch.tensor([[current_token_idx]], dtype=torch.long).to(device)
                embedded = self.embedding(current_tensor)  # (1, 1, E)

                # Один шаг LSTM
                lstm_out, (h_n, c_n) = self.lstm(embedded, (h_n, c_n))  # (1, 1, H)

                # Прогоняем через FC
                logits = self.fc(self.dropout(lstm_out.squeeze(1)))  # (1, V)
                logits = logits[0] / max(temperature, 1e-9)

                # Блокируем <PAD> и <UNK>
                logits[vocab["<PAD>"]] = float('-inf')
                if "<UNK>" in vocab:
                    logits[vocab["<UNK>"]] = float('-inf')

                # Софтмакс и сэмплирование
                probs = F.softmax(logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1).item()

                # Проверка на <PAD>
                if next_token_idx == vocab["<PAD>"]:
                    break

                # Добавляем в результат
                next_token = reverse_vocab.get(next_token_idx, "<UNK>")
                generated_tokens.append(next_token)

                # Обновляем текущий токен
                current_token_idx = next_token_idx

            return " ".join(generated_tokens)

    def save(self, path: str):
        """Сохраняет состояние модели."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Модель сохранена в {path}")

    def load(self, path: str, device: str = "cpu"):
        """Загружает состояние модели."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Модель загружена из {path}")