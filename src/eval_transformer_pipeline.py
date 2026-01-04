# src/eval_transformer_pipeline.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
import os

# Импорты (без использования dataloader для текстов)
from src.next_token_dataset import get_dataloader  # только для совместимости
from src.eval_lstm import compute_rouge_scores, generate_examples

# Пути
DATA_PATHS = {
    "train": "data/train.csv",
    "val": "data/val.csv",
    "test": "data/test.csv"
}
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Устройство
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_distilgpt2(model_name: str = "distilgpt2"):
    """
    Загружает DistilGPT-2 и токенизатор.
    """
    print(f"Загрузка модели {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Установка pad_token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(DEVICE)
    model.eval()

    print(f"Модель {model_name} загружена.")
    return model, tokenizer


def generate_completion_transformer(
    model,
    tokenizer,
    input_text: str,
    device: str = "cpu",
    max_input_fraction: float = 0.75,
    max_gen_length: int = 50,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 0.8,
) -> Tuple[str, str, str]:
    """
    Генерирует продолжение текста.
    :return: (context, target, completion)
    """
    tokens = input_text.strip().split()
    if len(tokens) < 2:
        return "", "", ""

    # Контекст = первые 75%
    input_length = max(1, min(len(tokens) - 1, int(len(tokens) * max_input_fraction)))
    context = " ".join(tokens[:input_length])
    target = " ".join(tokens[input_length:])

    # Токенизация контекста
    inputs = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length - max_gen_length,
        padding=False,
    ).to(device)

    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_gen_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1,
        )

    # Декодирование
    generated_tokens = outputs[0].tolist()
    context_tokens = tokenizer.encode(context, add_special_tokens=False)

    # Удаляем контекст по токенам
    if generated_tokens[:len(context_tokens)] == context_tokens:
        completion_tokens = generated_tokens[len(context_tokens):]
    else:
        completion_tokens = generated_tokens

    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()

    return context, target, completion


def evaluate_transformer_on_dataset(
    model,
    tokenizer,
    split: str = "val",
    max_samples: int = 500,
    max_gen_length: int = 50,
    save_results: bool = True,
) -> Dict[str, float]:
    """
    Оценка DistilGPT-2 по ROUGE.
    """
    print(f"Оценка DistilGPT-2 на {split} выборке...")

    if split not in DATA_PATHS:
        raise ValueError(f"split должен быть одним из {list(DATA_PATHS.keys())}")

    path = DATA_PATHS[split]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_csv(path)
    texts = df["text"].tolist()

    references = []
    candidates = []
    contexts_list = []

    pbar = tqdm(total=min(max_samples, len(texts)), desc="Генерация", unit="текст")

    for text in texts:
        if len(references) >= max_samples:
            break
        text = text.strip()
        if not text:
            continue

        try:
            context, target, generated = generate_completion_transformer(
                model=model,
                tokenizer=tokenizer,
                input_text=text,
                device=DEVICE,
                max_gen_length=max_gen_length
            )
            if target.strip() and generated.strip():
                references.append(target)
                candidates.append(generated)
                contexts_list.append(context)
                pbar.update(1)
        except Exception as e:
            print(f"Пропущен текст из-за ошибки: {e}")
            continue

    pbar.close()

    if not references:
        print("⚠️ Не удалось сгенерировать ни одного примера.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # Вычисление ROUGE
    rouge_scores = compute_rouge_scores(references, candidates)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    # Сохранение
    if save_results:
        results_df = pd.DataFrame({
            "context": contexts_list,
            "target": references,
            "generated": candidates
        })
        results_path = os.path.join(RESULTS_DIR, f"transformer_{split}_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Результаты сохранены: {results_path}")

    return rouge_scores


def generate_transformer_examples(
    model,
    tokenizer,
    sample_texts: List[str],
    max_gen_length: int = 10
):
    """
    Выводит примеры генерации.
    """
    print("Примеры автодополнения (DistilGPT-2):")
    for text in sample_texts:
        _, _, generated = generate_completion_transformer(
            model=model,
            tokenizer=tokenizer,
            input_text=text,
            device=DEVICE,
            max_gen_length=max_gen_length,
            temperature=0.8
        )
        print(f"  Вход: '{text}'")
        print(f"  → '{generated}'")
    print()


def main():
    """
    Основная функция.
    """
    model, tokenizer = load_distilgpt2("distilgpt2")

    sample_texts = [
        "hello how are",
        "the weather today",
        "i want to tell",
        "in the bank there was",
    ]

    generate_transformer_examples(model, tokenizer, sample_texts, max_gen_length=10)

    rouge_scores = evaluate_transformer_on_dataset(
        model=model,
        tokenizer=tokenizer,
        split="val",
        max_samples=500,
        max_gen_length=50
    )

    return rouge_scores


if __name__ == "__main__":
    main()