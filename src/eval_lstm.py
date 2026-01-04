import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer
from typing import Dict, List, Tuple
import pandas as pd
import os
import pickle

from src.lstm_model import LSTMModel


def evaluate_model(
    model: LSTMModel,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str = "cpu",
    pad_idx: int = 0
) -> Tuple[float, float]:
    """
    Оценивает модель по loss и accuracy с маской на <PAD>.
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
        try:
            generated = model.generate(
                start_text=context,
                vocab=vocab,
                reverse_vocab=reverse_vocab,
                max_length=max_gen_length,
                temperature=temperature,
                device=device
            )
            # Если model.generate вернул None или не строку
            if not isinstance(generated, str):
                generated = ""
        except Exception as e:
            print(f"[ERROR] Ошибка в model.generate для '{context}': {e}")
            generated = ""

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
    Выводит примеры генерации с защитой от ошибок.
    """
    model.eval()
    with torch.no_grad():
        for text in sample_texts:
            print(f"[LSTM] Обработка: '{text}'")  # 🔧 Отладка
            try:
                completion = model.generate(
                    start_text=text,
                    vocab=vocab,
                    reverse_vocab=reverse_vocab,
                    max_length=max_length,
                    temperature=temperature,
                    device=device
                )
                if not isinstance(completion, str):
                    completion = "<не строка>"
                print(f"  '{text}' → '{completion}'")
            except Exception as e:
                print(f"  ❌ Ошибка при генерации для '{text}': {type(e).__name__}: {e}")
    print()


def evaluate_on_dataset(
    model: LSTMModel,
    split: str = "val",
    batch_size: int = 64,
    device: str = "cpu",
    max_samples: int = 500,
    max_gen_length: int = 15
) -> Dict[str, float]:
    """
    Оценка модели LSTM по ROUGE-метрикам на реальных данных.
    """
    print(f"Оценка модели на {split} выборке...")

    path = {
        "train": "data/train.csv",
        "val": "data/val.csv",
        "test": "data/test.csv"
    }.get(split)

    if path is None:
        raise ValueError("split должен быть 'train', 'val' или 'test'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_csv(path, dtype=str).fillna("")
    texts = df["text"].tolist()

    # 🔧 Загрузка vocab
    vocab = getattr(model, 'vocab', None)
    if vocab is None:
        vocab_path = "models/vocab.pkl"
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Файл словаря не найден: {vocab_path}")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"[VOCAB] Загружен словарь из {vocab_path}, размер: {len(vocab)}")
    else:
        print("[VOCAB] Используется vocab из модели")

    reverse_vocab = {idx: token for token, idx in vocab.items()}
    references = []
    candidates = []
    processed = 0

    model.eval()
    torch.set_grad_enabled(False)

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

            # 🔧 Отладка
            if not target.strip():
                print(f"[DEBUG] Пропуск: пустой target для '{text}'")
            if not generated.strip():
                print(f"[DEBUG] Пропуск: пустая генерация")

            if target.strip() and generated.strip():
                references.append(target)
                candidates.append(generated)
                processed += 1
                pbar.update(1)

        except Exception as e:
            print(f"[ERROR] Исключение для '{text}': {e}")
            continue

    pbar.close()

    if not references:
        print("⚠️ Не удалось сгенерировать ни одного примера.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge_scores = compute_rouge_scores(references, candidates, use_stemmer=True)

    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    return {
        "rouge1": float(rouge_scores['rouge1']),
        "rouge2": float(rouge_scores['rouge2']),
        "rougeL": float(rouge_scores['rougeL'])
    }