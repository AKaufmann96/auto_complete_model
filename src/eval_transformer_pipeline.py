import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
import os

from src.next_token_dataset import get_dataloader
from src.eval_lstm import compute_rouge_scores, generate_examples


# Пути
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Выбор устройства
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_distilgpt2(model_name: str = "distilgpt2"):
    """
    Загружает модель и токенизатор DistilGPT-2.
    """
    print(f"Загрузка модели {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Устанавливаем pad_token (у GPT-2 нет pad_token по умолчанию)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(DEVICE)
    model.eval()  # режим инференса

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
    Генерирует продолжение текста с помощью трансформера.
    :param input_text: Полный текст.
    :return: (контекст, цель, сгенерированное продолжение)
    """
    tokens = input_text.strip().split()
    if len(tokens) < 2:
        return "", "", ""

    # Берём первые max_input_fraction как вход
    input_length = max(1, int(len(tokens) * max_input_fraction))
    context = " ".join(tokens[:input_length])
    target = " ".join(tokens[input_length:])

    # Токенизация
    inputs = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length - max_gen_length,
        padding=False
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
            num_return_sequences=1
        )

    # Декодирование
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Убираем контекст из сгенерированного текста
    if generated_text.startswith(context):
        completion = generated_text[len(context):].strip()
    else:
        # Если не начинается — возвращаем всё после контекста
        completion = generated_text.strip()

    return context, target, completion


def evaluate_transformer_on_dataset(
    model,
    tokenizer,
    split: str = "val",
    batch_size: int = 16,  # меньше, чем для LSTM — из-за размера модели
    max_samples: int = 1000,
    max_gen_length: int = 50,
    save_results: bool = True
) -> Dict[str, float]:
    """
    Оценка модели трансформера на выборке: ROUGE-метрики.
    """
    print(f"Оценка DistilGPT-2 на {split} выборке...")

    dataloader = get_dataloader(split, batch_size=1, num_workers=0)  # берём по одному тексту
    references = []
    candidates = []
    contexts_list = []

    num_samples = 0
    for batch in tqdm(dataloader, desc="Генерация продолжений"):
        _, _ = batch  # не используем таргеты — работаем с текстами напрямую
        text = dataloader.dataset.texts[num_samples]
        num_samples += 1

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
        except Exception as e:
            print(f"Ошибка при обработке текста: {e}")
            continue

        if len(references) >= max_samples:
            break

    # Вычисляем ROUGE
    rouge_scores = compute_rouge_scores(references, candidates)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    # Сохраняем примеры
    if save_results:
        results_df = pd.DataFrame({
            "context": contexts_list,
            "target": references,
            "generated": candidates
        })
        results_path = os.path.join(RESULTS_DIR, f"transformer_{split}_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Результаты сохранены в {results_path}")

    return rouge_scores


def generate_transformer_examples(
    model,
    tokenizer,
    sample_texts: List[str],
    max_gen_length: int = 10
):
    """
    Выводит примеры генерации трансформера.
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
        print(f"  '{text}' → '{generated}'")
    print()


def main():
    """
    Основная функция: загрузка модели, генерация примеров, оценка.
    """
    # Загружаем модель и токенизатор
    model, tokenizer = load_distilgpt2("distilgpt2")

    # Примеры генерации
    sample_texts = [
        "привет как",
        "сегодня погода",
        "я хочу рассказать",
        "в банке произошла"
    ]
    generate_transformer_examples(model, tokenizer, sample_texts, max_gen_length=10)

    # Оценка на валидации
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