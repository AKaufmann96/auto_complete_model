import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm

from src.lstm_model import LSTMModel
from src.next_token_dataset import get_dataloader
from src.eval_lstm import evaluate_model, generate_examples

# Пути
MODEL_SAVE_PATH = "models/lstm_model.pth"
os.makedirs("models", exist_ok=True)

# Гиперпараметры
BATCH_SIZE = 256
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_LENGTH = 10
TEMPERATURE = 0.8


def train_epoch(
    model: LSTMModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
):
    """
    Обучение модели на одной эпохе.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for input_ids, target_ids in progress_bar:
        input_ids = input_ids.to(DEVICE)  # (batch_size, seq_len)
        target_ids = target_ids.to(DEVICE)  # (batch_size,)

        optimizer.zero_grad()
        logits = model(input_ids)  # (batch_size, vocab_size)
        loss = criterion(logits, target_ids)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == target_ids).sum().item()
        total += target_ids.size(0)

        progress_bar.set_postfix({"loss": loss.item(), "acc": f"{correct/total:.3f}"})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model():
    """
    Основная функция обучения модели.
    """
    print(f"Используется устройство: {DEVICE}")

    # Загрузка данных
    train_loader = get_dataloader("train", batch_size=BATCH_SIZE)
    val_loader = get_dataloader("val", batch_size=BATCH_SIZE)
    vocab = train_loader.dataset.vocab
    vocab_size = len(vocab)
    pad_idx = vocab["<PAD>"]

    # Инициализация модели
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=pad_idx,
    ).to(DEVICE)

    # Оптимизатор и лосс
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Обучение
    best_val_loss = float("inf")
    reverse_vocab = {idx: token for token, idx in vocab.items()}

    for epoch in range(NUM_EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{NUM_EPOCHS}")

        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Валидация
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, DEVICE)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Примеры генерации
        print("Примеры автодополнения:")
        sample_texts = [
            "hello how are",
            "the weather today",
            "i want to tell",
            "in the bank there was",
        ]
        generate_examples(
            model,
            sample_texts,
            vocab,
            reverse_vocab,
            DEVICE,
            max_length=MAX_GEN_LENGTH,
            temperature=TEMPERATURE,
        )

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(MODEL_SAVE_PATH)
            print(f"Модель сохранена: {MODEL_SAVE_PATH}")

    print("Обучение завершено.")


if __name__ == "__main__":
    train_model()
