# 参考
# [huggingface/transformers の日本語BERTで文書分類器を作成する] https://qiita.com/nekoumei/items/7b911c61324f16c43e7e
#

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

if "JUPYTER_NOTEBOOK" in os.environ:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# ハイパーパラメータ
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5


# 評価指標
def flat_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)


def flat_precision_recall_fscore(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_recall_fscore_support(labels_flat, preds_flat, average="macro")


# トレーニング関数
def train(
    device,
    model,
    optimizer,
    scheduler,
    tokenizer,
    train_dataloader,
    validation_dataloader,
    test_dataloader,
    output_dir,
):
    # ベストなモデルを保存
    best_validation_loss = float("inf")
    best_model = None

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 10)

        # トレーニング
        total_train_loss = 0
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        # 平均トレーニング損失
        avg_train_loss = total_train_loss / len(train_dataloader)

        # 検証
        total_validation_loss = 0
        model.eval()

        for batch in validation_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss
                total_validation_loss += loss.item()

        # 平均検証損失
        avg_validation_loss = total_validation_loss / len(validation_dataloader)
        print(f"Average validation loss: {avg_validation_loss}")

        # 評価
        model.eval()
        predictions, true_labels = [], []

        for batch in test_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            logits = outputs.logits

            predictions.append(logits.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())

        # 精度、適合率、再現率、F1スコア
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        accuracy = flat_accuracy(predictions, true_labels)
        precision, recall, f1, _ = flat_precision_recall_fscore(
            predictions, true_labels
        )

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")

        # 訓練済みモデルを保存する
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


def load_dataset(dataset_name):
    with open(dataset_name) as f:
        lines = f.readlines()
    labels = []
    texts = []
    for i, line in enumerate(lines):
        try:
            label, text = line.strip().split("\t")
        except Exception as e:
            print(e)
            print(f"{i}: {line}")
            raise
        labels.append(label)
        texts.append(text)
    return labels, texts


def main(train_dataset, val_dataset, test_dataset, output_dir: Path):
    # トレーニング、検証、テストセットの読み込み & ラベルと本文を取得
    train_labels, train_texts = load_dataset(train_dataset)
    val_labels, val_texts = load_dataset(val_dataset)
    test_labels, test_texts = load_dataset(test_dataset)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ラベルを数値に変換(常に同じ順番にするためlabel setを事前にソート)
    label2id = {l: i for i, l in enumerate(sorted(set(train_labels)))}
    if (output_dir / "label2id.json").exists():
        raise Exception(f"{str(output_dir / 'label2id.json')} already exits.")

    print(f"{label2id=}")
    (output_dir / "label2id.json").write_text(json.dumps(label2id))
    train_labels = [label2id[l] for l in train_labels]
    val_labels = [label2id[l] for l in val_labels]
    test_labels = [label2id[l] for l in test_labels]

    # トークン化とエンコード
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    TOKEN_MAX_LEN = 512
    train_encodings = tokenizer(
        train_texts, max_length=TOKEN_MAX_LEN, truncation=True, padding=True
    )
    val_encodings = tokenizer(
        val_texts, max_length=TOKEN_MAX_LEN, truncation=True, padding=True
    )
    test_encodings = tokenizer(
        test_texts, max_length=TOKEN_MAX_LEN, truncation=True, padding=True
    )

    # データセットの作成
    train_dataset = TensorDataset(
        torch.tensor(train_encodings["input_ids"]),
        torch.tensor(train_encodings["attention_mask"]),
        torch.tensor(train_labels),
    )

    val_dataset = TensorDataset(
        torch.tensor(val_encodings["input_ids"]),
        torch.tensor(val_encodings["attention_mask"]),
        torch.tensor(val_labels),
    )

    test_dataset = TensorDataset(
        torch.tensor(test_encodings["input_ids"]),
        torch.tensor(test_encodings["attention_mask"]),
        torch.tensor(test_labels),
    )

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    batch_size = BATCH_SIZE
    if torch.cuda.is_available():
        device_idx = 0
        device_props = torch.cuda.get_device_properties(device_idx)
        gpu_memory = device_props.total_memory
        batch_size = gpu_memory // (1000 * 1000 * 1000)
        print(f"GPU Memory: {gpu_memory / 1024 / 1024:.2f} MB, {batch_size=}")

    # データローダーの作成
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size
    )

    # モデルの定義
    model = BertForSequenceClassification.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking",
        num_labels=len(label2id),
        output_attentions=False,
        output_hidden_states=False,
    )

    # GPUへの転送
    model.to(device)

    # optimizerの定義
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

    # エポック数の計算
    total_steps = len(train_dataloader) * NUM_EPOCHS

    # スケジューラーの定義
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    train(
        device,
        model,
        optimizer,
        scheduler,
        tokenizer,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        output_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train-txt", help="training dataset", required=True)
    parser.add_argument("--val-txt", help="validation dataset", required=True)
    parser.add_argument("--test-txt", help="test dataset", required=True)
    parser.add_argument("--output-dir", help="output dir", required=True)

    args = parser.parse_args()
    main(args.train_txt, args.val_txt, args.test_txt, Path(args.output_dir))
