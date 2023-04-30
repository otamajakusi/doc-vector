# 【日本語モデル付き】2020年に自然言語処理をする人にお勧めしたい文ベクトルモデル
# https://qiita.com/sonoisa/items/1df94d0a98cd4f209051

import json
from transformers import BertJapaneseTokenizer, BertModel
import torch
import pickle


class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx : batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch, padding="longest", truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            ).to("cpu")

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)


MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
SENTENCE_VECTORS_NAME = "irasuto_items_part.pkl"
model = SentenceBertJapanese(MODEL_NAME)
with open("irasuto_items_part.json") as f:
    items = json.loads(f.read())

sentences = [item["title"] for item in items]
sentence_vectors = model.encode(sentences, batch_size=8)

## inference

import scipy.spatial
import copy

queries = ["暴走したAI", "暴走した人工知能", "いらすとやさんに感謝", "つづく"]
query_embeddings = model.encode(queries).numpy()

closest_n = 5
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist(
        [query_embedding], sentence_vectors, metric="cosine"
    )[0]

    # FIXME: zipをlistに変換しないと何故か途中printしたときにうまく動作しなかった
    results = list(zip(range(len(distances)), distances))
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(sentences[idx].strip(), "(Score: %.4f)" % (distance / 2))
