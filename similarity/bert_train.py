import os
import torch
import pickle
from transformers import BertJapaneseTokenizer, BertModel

if "JUPYTER_NOTEBOOK" in os.environ:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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

# トレーニングデータのベクトルを作成する
training_vectors = []
for text in tqdm(training_data):
    # 文章をトークン化して、BERTの入力に変換する
    input_ids = torch.tensor(
        [
            tokenizer.encode(
                text,
                max_length=TOKEN_MAX_LEN,
                truncation=True,
                padding=True,
                add_special_tokens=True,
            )
        ]
    ).to(device)

    # BERTモデルで処理して、ベクトルを作成する
    with torch.no_grad():
        outputs = model(input_ids)
        vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        training_vectors.append(vector)

# pickleファイルにデータを書き込む
with open("training_vectors.pkl", "wb") as f:
    pickle.dump(training_vectors, f)
