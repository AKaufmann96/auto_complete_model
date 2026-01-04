import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# Исправленные импорты
from src.lstm_model import LSTMModel
from src.next_token_dataset import get_dataloader
from src.eval_lstm import evaluate_model, generate_examples

# Пути
MODEL_SAVE_PATH = "models/lstm_model.pth"
os.makedirs("models", exist_ok=True)


def train_epoch(
    model: LSTMModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    pad_idx: int,
    device: str
):
    """
    Обучение модели на одной эпохе.

    Аргументы:
        model: обучаемая модель
        dataloader: загрузчик данных
        optimizer: оптимизатор
        criterion: функция потерь
        pad_idx: индекс паддинга для маски
        device: устройство ('cuda' или 'cpu')
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for input_ids, target_ids in progress_bar:
        # Переносим данные на устройство
        input_ids = input_ids.to(device)      # (batch_size, seq_len)
        target_ids = target_ids.to(device)    # (batch_size, seq_len)

        optimizer.zero_grad()

        # Прямой проход: (B, T, V)
        logits = model(input_ids)

        # Reshape для CrossEntropyLoss
        loss = criterion(
            logits.view(-1, logits.size(-1)),   # (B*T, V)
            target_ids.view(-1)                 # (B*T,)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Подсчёт accuracy с маской
        preds = torch.argmax(logits, dim=-1)   # (B, T)
        mask = target_ids != pad_idx
        n_correct = ((preds == target_ids) & mask).sum().item()
        n_total = mask.sum().item()

        correct += n_correct
        total += n_total

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{n_correct/n_total:.3f}" if n_total > 0 else 0.0
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train_model(
    num_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.001,
    hidden_size: int = 256,
    embedding_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    max_gen_length: int = 15,
    temperature: float = 0.8,
    max_length: int = 128,
    device: str = None
):
    """
    Основная функция обучения LSTM-модели для автодополнения текста.

    Аргументы:
        num_epochs: количество эпох обучения
        batch_size: размер батча
        lr: скорость обучения для Adam
        hidden_size: размер скрытого состояния LSTM
        embedding_dim: размер векторов эмбеддингов
        num_layers: количество слоёв LSTM
        dropout: вероятность дропаута в модели
        max_gen_length: максимальная длина генерации при примерах
        temperature: температура для сэмплирования (сглаживание логитов)
        max_length: максимальная длина последовательности в модели
        device: устройство ('cuda' или 'cpu'). Если None — автоматически.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Используется устройство: {device}")

    # Загрузка данных
    train_loader = get_dataloader("train", batch_size=batch_size)
    val_loader = get_dataloader("val", batch_size=batch_size)

    vocab = train_loader.dataset.vocab
    vocab_size = len(vocab)
    pad_idx = vocab["<PAD>"]
    reverse_vocab = {idx: token for token, idx in vocab.items()}

    # Инициализация модели
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx,
        max_length=max_length
    ).to(device)

    # Оптимизатор и лосс
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Обучение
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")

        # Обучение
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            pad_idx=pad_idx,
            device=device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Валидация
        val_loss, val_acc = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            pad_idx=pad_idx
        )
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
            model=model,
            sample_texts=sample_texts,
            vocab=vocab,
            reverse_vocab=reverse_vocab,
            device=device,
            max_length=max_gen_length,
            temperature=temperature,
        )

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(MODEL_SAVE_PATH)
            print(f"Модель сохранена: {MODEL_SAVE_PATH}")

    print("Обучение завершено.")


if __name__ == "__main__":
    # Запуск с параметрами по умолчанию
    train_model()