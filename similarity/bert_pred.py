import os
import torch
import random
import pickle
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_MAX_LEN = 512

# GPUを使用する場合は、deviceをcudaに設定する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 日本語のBERTモデルを使用する
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking").to(
    device
)


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


# トレーニングデータ
_, training_data = load_dataset("train.txt")

with open("training_vectors.pkl", "rb") as f:
    training_vectors = pickle.load(f)

# テストデータ
with open("test.txt") as f:
    test_dataset = f.read().split("\n")
_, test_data = random.choice(test_dataset).split("\t")

print(f"{'*'*50}test{'*'*50}")
print(test_data)

# テストデータのベクトルを作成する
input_ids = torch.tensor(
    [
        tokenizer.encode(
            test_data,
            max_length=TOKEN_MAX_LEN,
            truncation=True,
            padding=True,
            add_special_tokens=True,
        )
    ]
).to(device)
with torch.no_grad():
    outputs = model(input_ids)
    test_vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# トレーニングデータとテストデータの類似度を計算する
similarities = cosine_similarity([test_vector], training_vectors)

print(similarities)
# 類似度が高い順にトレーニングデータの文章を表示する
ranking = similarities.argsort()[0][::-1]
for i in ranking[:5]:
    print(f"{'*'*50}{i}{'*'*50}")
    print(training_data[i])
