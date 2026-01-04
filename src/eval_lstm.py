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
    device: str = "cpu",
    pad_idx: int = 0
) -> Tuple[float, float]:
    """
    Оценивает модель: loss и accuracy с маской на <PAD>.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)  # (B, T, V)

            # Reshape для loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            total_loss += loss.item()

            # Accuracy с маской
            preds = torch.argmax(logits, dim=-1)  # (B, T)
            mask = target_ids != pad_idx
            n_correct = ((preds == target_ids) & mask).sum().item()
            n_total = mask.sum().item()

            correct += n_correct
            total += n_total

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def generate_completion(
    model: LSTMModel,
    input_text: str,
    vocab: dict,
    reverse_vocab: dict,
    device: str = "cpu",
    max_gen_length: int = 15,
    temperature: float = 0.7
) -> Tuple[str, str, str]:
    """
    Генерирует продолжение на основе первых 3/4 текста.
    :return: (context, target, generated)
    """
    model.eval()
    with torch.no_grad():
        tokens = input_text.strip().lower().split()
        if len(tokens) < 2:
            return "", "", ""

        # Берём 3/4 как контекст
        input_length = max(1, min(len(tokens) - 1, 3 * len(tokens) // 4))
        context = " ".join(tokens[:input_length])
        target = " ".join(tokens[input_length:])

        # Генерация
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
    Вычисляет усреднённые ROUGE-метрики.
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=use_stemmer
    )

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for ref, cand in zip(references, candidates):
        if not ref.strip() or not cand.strip():
            continue
        score = scorer.score(ref, cand)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)

    avg_scores = {k: np.mean(v) if v else 0.0 for k, v in scores.items()}
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
    Выводит примеры генерации.
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
    max_samples: int = 500,
    max_gen_length: int = 15
):
    """
    Оценка модели по ROUGE на реальных данных.
    """
    print(f"Оценка модели на {split} выборке...")
    model.eval()
    torch.set_grad_enabled(False)

    path = {
        "train": "data/train.csv",
        "val": "data/val.csv",
        "test": "data/test.csv"
    }[split]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_csv(path)
    texts = df["text"].tolist()

    vocab = model.vocab if hasattr(model, 'vocab') else None
    if vocab is None:
        # Если нет — загружаем из датасета
        temp_loader = get_dataloader(split="train", batch_size=1)
        vocab = temp_loader.dataset.vocab

    reverse_vocab = {idx: token for token, idx in vocab.items()}
    references = []
    candidates = []
    processed = 0

    pbar = tqdm(total=max_samples, desc="Генерация", unit="текст")

    for text in texts:
        if processed >= max_samples:
            break
        text = text.strip()
        if not text:
            continue

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
            continue

    pbar.close()

    if not references:
        print("⚠️ Не удалось сгенерировать ни одного примера.")
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

    rouge_scores = compute_rouge_scores(references, candidates)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    return {
        "rouge-1": float(rouge_scores['rouge1']),
        "rouge-2": float(rouge_scores['rouge2']),
        "rouge-l": float(rouge_scores['rougeL'])
    }