import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer
from typing import Dict, List, Tuple
import pandas as pd

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
    max_gen_length: int = 50,
    temperature: float = 1.0
) -> str:
    """
    Генерирует продолжение текста.
    :param input_text: Полный текст.
    :return: Сгенерированное продолжение.
    """
    model.eval()
    with torch.no_grad():
        tokens = input_text.lower().split()
        if len(tokens) < 2:
            return "", "", ""

        # Берём первые 3/4 как вход
        input_length = max(1, int(len(tokens) * 0.75))
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
    max_samples: int = 1000,
    max_gen_length: int = 50
):
    """
    Полная оценка модели на датасете: ROUGE-метрики.
    """
    print(f"Оценка модели на {split} выборке...")

    dataloader = get_dataloader(split, batch_size=batch_size)
    vocab = dataloader.dataset.vocab
    reverse_vocab = {idx: token for token, idx in vocab.items()}

    references = []
    candidates = []

    # Ограничиваем число примеров для ускорения
    num_samples = 0
    for batch in tqdm(dataloader, desc="Generating completions"):
        input_ids, target_ids = batch
        texts = dataloader.dataset.texts  # Предполагаем, что dataset хранит оригинальные тексты

        batch_texts = texts[num_samples:num_samples + len(input_ids)]
        num_samples += len(input_ids)

        for text in batch_texts:
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

        if len(references) >= max_samples:
            break

    # Вычисляем ROUGE
    rouge_scores = compute_rouge_scores(references, candidates)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    return rouge_scores