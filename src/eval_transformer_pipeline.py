import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, List
import os


# === ВНУТРЕННЯЯ ФУНКЦИЯ: calculate_rouge_batch ===
def calculate_rouge_batch(
    hypotheses: List[str],
    references: List[str],
    use_stemmer: bool = True
) -> Dict[str, float]:
    """
    Вычисляет усреднённые ROUGE-метрики по списку гипотез и эталонов.

    Аргументы:
        hypotheses: список сгенерированных продолжений
        references: список ожидаемых продолжений
        use_stemmer: использовать ли стеммер (рекомендуется для русского)

    Возвращает:
        Словарь с метриками: {'rouge1': ..., 'rouge2': ..., 'rougeL': ...}
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("Установите пакет: pip install rouge-score")

    if len(hypotheses) == 0 or len(references) == 0:
        raise ValueError("Списки гипотез и эталонов не должны быть пустыми.")
    if len(hypotheses) != len(references):
        raise ValueError("Количество гипотез и эталонов должно совпадать.")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    scores = {key: [] for key in ['rouge1', 'rouge2', 'rougeL']}

    for hyp, ref in zip(hypotheses, references):
        if not hyp.strip() or not ref.strip():
            continue
        score = scorer.score(ref, hyp)
        for key in scores:
            scores[key].append(score[key].fmeasure)

    return {key: np.mean(vals) if vals else 0.0 for key, vals in scores.items()}
# ==============================================


def evaluate_transformer(
    model_name: str = "distilgpt2",
    split: str = "val",
    max_samples: int = 500,
    max_length: int = 30,
    device: str = None,
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Оценка предобученной языковой модели (например, DistilGPT-2) на задаче автодополнения.

    Модель генерирует продолжение текста, которое сравнивается с реальным "хвостом" исходного текста.

    Аргументы:
        model_name: имя модели из Hugging Face Hub
        split: 'train', 'val' или 'test'
        max_samples: максимальное число примеров для оценки
        max_length: максимальное число новых токенов для генерации
        device: устройство ('cuda' или 'cpu')
        batch_size: размер батча (не используется напрямую)

    Возвращает:
        Словарь с усреднёнными ROUGE-метриками:
        {'rouge1': float, 'rouge2': float, 'rougeL': float}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Загрузка токенизатора и модели: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Установка pad_token, если отсутствует
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("⚠️ pad_token не задан. Используется eos_token как pad_token.")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Загрузка оригинальных текстов
    data_path = f"data/{split}.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл данных не найден: {data_path}")

    df = pd.read_csv(data_path, dtype=str).fillna("")
    texts = df["text"].tolist()
    texts = [t.strip() for t in texts if t.strip()]
    texts = texts[:max_samples]

    all_hypotheses = []  # сгенерированные продолжения
    all_references = []  # реальные продолжения

    print(f"Оценка на {split} (max_samples={max_samples})...")

    with torch.no_grad():
        for i, text in enumerate(tqdm(texts, desc="Генерация", unit="текст")):
            if not text:
                continue

            # Полная токенизация текста
            full_encoding = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
            if len(full_encoding) <= 1:
                continue  # слишком короткий — нет места для продолжения

            # Определяем длину контекста: min(50, len-1), но не менее 1
            context_len = min(50, len(full_encoding) - 1)
            input_ids = full_encoding[:context_len].unsqueeze(0).to(device)  # (1, T)

            # Эталонное продолжение — оставшиеся токены
            target_token_ids = full_encoding[context_len:]
            reference_suffix = tokenizer.decode(target_token_ids, skip_special_tokens=True)

            # Создаём attention_mask
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            # Генерация новых токенов
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_k=50,
                top_p=0.95
            )

            # Декодируем вход
            prefix_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

            # Сгенерированный текст
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Извлечение продолжения
            if generated_text.startswith(prefix_text):
                hypothesis_suffix = generated_text[len(prefix_text):].strip()
            else:
                # Пытаемся найти префикс в сгенерированном тексте
                idx = generated_text.lower().find(prefix_text.lower())
                if idx != -1:
                    end_idx = idx + len(prefix_text)
                    hypothesis_suffix = generated_text[end_idx:].strip()
                else:
                    hypothesis_suffix = generated_text.strip()

            all_hypotheses.append(hypothesis_suffix)
            all_references.append(reference_suffix)

    # Фильтрация пустых пар
    non_empty_pairs = [
        (h, r) for h, r in zip(all_hypotheses, all_references)
        if h.strip() and r.strip()
    ]

    if len(non_empty_pairs) == 0:
        print("⚠️ Не найдено ни одной пары с непустыми гипотезой и эталоном.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # Расчёт итоговых метрик
    hypotheses, references = zip(*non_empty_pairs)
    rouge_scores = calculate_rouge_batch(hypotheses, references)

    print(f"✅ Использовано для оценки: {len(non_empty_pairs)} примеров")
    return rouge_scores


def generate_transformer_examples(
    model,
    tokenizer,
    sample_texts: List[str],
    max_gen_length: int = 10,
    device: str = None
):
    """
    Генерирует и выводит примеры автодополнения для трансформерной модели.

    Аргументы:
        model: предобученная модель (например, DistilGPT-2)
        tokenizer: токенизатор
        sample_texts: список начальных фраз
        max_gen_length: максимальное число новых токенов
        device: устройство ('cuda' или 'cpu')
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    print(f"Генерация с max_new_tokens={max_gen_length}...\n")

    with torch.no_grad():
        for text in sample_texts:
            # Токенизация
            full_encoding = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
            if len(full_encoding) <= 1:
                print(f"  '{text}' → [текст слишком короткий]")
                continue

            context_len = min(50, len(full_encoding))
            input_ids = full_encoding[:context_len].unsqueeze(0).to(device)

            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_k=50,
                top_p=0.95
            )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prefix_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

            if generated.startswith(prefix_text):
                completion = generated[len(prefix_text):].strip()
            else:
                idx = generated.lower().find(prefix_text.lower())
                if idx != -1:
                    end_idx = idx + len(prefix_text)
                    completion = generated[end_idx:].strip()
                else:
                    completion = generated.strip()

            print(f"  '{text}' → '{completion}'")
    print()


def main():
    """
    Точка входа для запуска из командной строки.
    Пример: python -m src.eval_transformer_pipeline --model_name distilgpt2 --split val --max_samples 100
    """
    import argparse

    parser = argparse.ArgumentParser(description="Оценка языковой модели на задаче автодополнения.")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Модель из Hugging Face")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val", help="Сплит для оценки")
    parser.add_argument("--max_samples", type=int, default=500, help="Макс. число примеров")
    parser.add_argument("--max_length", type=int, default=30, help="Макс. длина генерации (токенов)")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--device", type=str, default=None, help="Устройство: 'cuda' или 'cpu'")

    args = parser.parse_args()

    scores = evaluate_transformer(
        model_name=args.model_name,
        split=args.split,
        max_samples=args.max_samples,
        max_length=args.max_length,
        device=args.device,
        batch_size=args.batch_size
    )

    print("\n✅ Оценка завершена:")
    for metric, value in scores.items():
        print(f"{metric.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()