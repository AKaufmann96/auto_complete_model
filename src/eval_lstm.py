# src/eval_lstm.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer
from typing import Dict, List, Tuple
import pandas as pd
import os

from src.next_token_dataset import get_dataloader
from src.lstm_model import LSTMModel


def evaluate_model(
    model: LSTMModel,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    Оценивает модель на выборке: возвращает средний loss и accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = criterion(logits, target_ids)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == target_ids).sum().item()
            total += target_ids.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def generate_completion(
    model: LSTMModel,
    input_text: str,
    vocab: dict,
    reverse_vocab: dict,
    device: str = "cpu",
    max_gen_length: int = 15,  # Уменьшено для скорости
    temperature: float = 0.7   # Снижено для менее шумной генерации
) -> str:
    """
    Генерирует продолжение текста.
    :param input_text: Полный текст.
    :return: context, target, generated
    """
    model.eval()
    with torch.no_grad():
        tokens = input_text.strip().lower().split()
        if len(tokens) < 2:
            return "", "", ""

        # Берём 50–75% как контекст (но минимум 1 токен)
        input_length = max(1, min(len(tokens) - 1, len(tokens) // 2))
        context = " ".join(tokens[:input_length])
        target = " ".join(tokens[input_length:])

        # Генерируем продолжение
        generated = model.generate(
            start_text=context,
            vocab=vocab,
            reverse_vocab=reverse_vocab,
            max_length=max_gen_length,
            temperature=temperature,
            device=device
        )
        return context, target, generated


def compute_rouge_scores(
    references: List[str],
    candidates: List[str],
    use_stemmer: bool = True
) -> Dict[str, float]:
    """
    Вычисляет усреднённые ROUGE-метрики (R-1, R-2, R-L).
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=use_stemmer
    )

    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }

    for ref, cand in zip(references, candidates):
        if not ref.strip() or not cand.strip():
            continue
        score = scorer.score(ref, cand)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)

    avg_scores = {
        k: np.mean(v) if v else 0.0 for k, v in scores.items()
    }
    return avg_scores


def generate_examples(
    model: LSTMModel,
    sample_texts: List[str],
    vocab: dict,
    reverse_vocab: dict,
    device: str,
    max_length: int = 10,
    temperature: float = 0.8
):
    """
    Выводит примеры генерации модели.
    """
    model.eval()
    with torch.no_grad():
        for text in sample_texts:
            completion = model.generate(
                start_text=text,
                vocab=vocab,
                reverse_vocab=reverse_vocab,
                max_length=max_length,
                temperature=temperature,
                device=device
            )
            print(f"  '{text}' → '{completion}'")
    print()


def evaluate_on_dataset(
    model: LSTMModel,
    split: str = "val",
    batch_size: int = 64,
    device: str = "cpu",
    max_samples: int = 500,        # Ограничено
    max_gen_length: int = 15       # Короткая генерация
):
    """
    Полная оценка модели на датасете: ROUGE-метрики.
    """
    print(f"Оценка модели на {split} выборке...")
    model.eval()
    torch.set_grad_enabled(False)

    # Загружаем датасет и извлекаем тексты
    dataloader = get_dataloader(split=split, batch_size=batch_size, num_workers=0)
    dataset = dataloader.dataset
    vocab = dataset.vocab
    reverse_vocab = {idx: token for token, idx in vocab.items()}

    # Извлекаем тексты из CSV (надёжнее, чем хранение в dataset.texts)
    path = {
        "train": "data/train.csv",
        "val": "data/val.csv",
        "test": "data/test.csv"
    }[split]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_csv(path)
    texts = df["text"].tolist()

    references = []
    candidates = []
    processed = 0

    # Используем tqdm для прогресса
    pbar = tqdm(total=max_samples, desc="Генерация", unit="текст")

    for i, text in enumerate(texts):
        if processed >= max_samples:
            break

        text = text.strip()
        if not text:
            continue

        # Генерация
        try:
            _, target, generated = generate_completion(
                model=model,
                input_text=text,
                vocab=vocab,
                reverse_vocab=reverse_vocab,
                device=device,
                max_gen_length=max_gen_length
            )

            if target.strip() and generated.strip():
                references.append(target)
                candidates.append(generated)
                processed += 1
                pbar.update(1)

        except Exception as e:
            # Пропускаем проблемные примеры
            continue

    pbar.close()

    if not references:
        print("⚠️ Не удалось сгенерировать ни одного примера.")
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

    # Вычисляем ROUGE
    rouge_scores = compute_rouge_scores(references, candidates)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    return {
        "rouge-1": float(rouge_scores['rouge1']),
        "rouge-2": float(rouge_scores['rouge2']),
        "rouge-l": float(rouge_scores['rougeL'])
    }