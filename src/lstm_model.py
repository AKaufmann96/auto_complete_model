import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np
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
        pad_idx: int = 0,
        max_length: int = 128
    ):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.max_length = max_length

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
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) and module.padding_idx is not None:
            nn.init.xavier_uniform_(module.weight)
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
        max_length: int = 15,
        temperature: float = 1.0,
        device: str = "cpu"
    ) -> str:
        """
        Генерация продолжения текста пошагово.
        """
        self.eval()
        with torch.no_grad():
            tokens = start_text.strip().lower().split()
            if not tokens:
                return ""

            input_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
            seq_len = self.max_length

            generated_tokens = tokens.copy()

            for _ in range(max_length):
                # Берём последние max_length токенов
                current_input = input_ids[-seq_len:]
                # Паддинг слева
                padded_input = [self.pad_idx] * (seq_len - len(current_input)) + current_input
                input_tensor = torch.tensor([padded_input], dtype=torch.long).to(device)

                # Логиты для всех шагов
                logits = self.forward(input_tensor)  # (1, T, V)
                # Берём логиты последнего непустого токена
                last_logits = logits[0, len(current_input)-1]  # (V,)

                # Температура
                last_logits = last_logits / max(temperature, 1e-9)
                probs = F.softmax(last_logits, dim=-1)  # (V,)

                # Блокируем <PAD> и <UNK>
                probs[vocab["<PAD>"]] = 0
                probs[vocab.get("<UNK>", -1)] = 0
                if probs.sum() < 1e-8:
                    break
                probs = probs / probs.sum()

                # Сэмплирование
                next_token_idx = torch.multinomial(probs, 1).item()
                if next_token_idx == vocab["<PAD>"]:
                    break

                input_ids.append(next_token_idx)
                next_token = reverse_vocab.get(next_token_idx, "<UNK>")
                generated_tokens.append(next_token)

            return " ".join(generated_tokens[len(tokens):]).strip()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Модель сохранена в {path}")

    def load(self, path: str, device: str = "cpu"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Модель загружена из {path}")