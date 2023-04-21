import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# トレーニング、検証、テストセットの読み込み
with open("livedoor_news_corpus/text/ldcc_train.txt") as f:
    train_lines = f.readlines()

with open("livedoor_news_corpus/text/ldcc_val.txt") as f:
    val_lines = f.readlines()

with open("livedoor_news_corpus/text/ldcc_test.txt") as f:
    test_lines = f.readlines()

# ラベルと本文を取得
train_labels, train_texts = [], []
for line in train_lines:
    label, text = line.strip().split("\t")
    train_labels.append(label)
    train_texts.append(text)

val_labels, val_texts = [], []
for line in val_lines:
    label, text = line.strip().split("\t")
    val_labels.append(label)
    val_texts.append(text)

test_labels, test_texts = [], []
for line in test_lines:
    label, text = line.strip().split("\t")
    test_labels.append(label)
    test_texts.append(text)

# ラベルを数値に変換
label2id = {l: i for i, l in enumerate(set(train_labels))}
train_labels = [label2id[l] for l in train_labels]
val_labels = [label2id[l] for l in val_labels]
test_labels = [label2id[l] for l in test_labels]

# トークン化とエンコード
tokenizer = BertTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

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

# ハイパーパラメータ
batch_size = 16
learning_rate = 2e-5
num_epochs = 5

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

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

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
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

# エポック数の計算
total_steps = len(train_dataloader) * num_epochs

# スケジューラーの定義
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

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
def train(model, train_dataloader, validation_dataloader):
    # ベストなモデルを保存
    best_validation_loss = float("inf")
    best_model = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # トレーニング
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
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
        output_dir = "./model_save/"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
