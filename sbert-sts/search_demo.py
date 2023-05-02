# https://zenn.dev/skwbc/articles/sentence_bert_implementation
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from baseline_experiment import BaselineModel
from sentence_bert_experiment import SentenceBert, encode_single_sentences

USE_CUDA = True
BATCH_SIZE = 64
MAX_LENGTH = 512
MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"

ROOTDIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = ROOTDIR / "JGLUE" / "datasets" / "jsts-v1.1" / "train-v1.1.json"
VALID_DATA_PATH = ROOTDIR / "JGLUE" / "datasets" / "jsts-v1.1" / "valid-v1.1.json"


def load_test_texts():
    df_test = pd.read_json(VALID_DATA_PATH, lines=True).set_index("sentence_pair_id")
    test_texts = pd.concat([df_test["sentence1"], df_test["sentence2"]])
    test_texts = test_texts.drop_duplicates()
    return test_texts


class SentenceBertSearcher:
    def __init__(self, texts):
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.sentence_bert = SentenceBert()
        self.sentence_bert.load_state_dict(torch.load("best_sentence_bert_model.bin"))
        if USE_CUDA:
            self.sentence_bert = self.sentence_bert.cuda()
        self.sentence_bert.eval()
        self.sentence_bert_vectors = self.make_sentence_vectors(texts)

    def search(self, query, data_size=None):
        if data_size is None:
            data_size = len(self.texts)

        query_vector = self.make_sentence_vectors(pd.Series([query]))
        similarities = cosine_similarity(
            query_vector, self.sentence_bert_vectors[:data_size]
        )
        df_result = pd.DataFrame(
            {"text": self.texts.iloc[:data_size], "similarity": similarities[0]}
        )
        return df_result.sort_values("similarity", ascending=False)

    def make_sentence_vectors(self, texts):
        encoded = encode_single_sentences(texts, self.tokenizer)
        dataset_for_loader = [
            {k: v[i] for k, v in encoded.items()} for i in range(len(texts))
        ]
        sentence_vectors = []
        for batch in DataLoader(dataset_for_loader, batch_size=BATCH_SIZE):
            if USE_CUDA:
                batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                bert_output = self.sentence_bert.bert(**batch)
                sentence_vector = self.sentence_bert._mean_pooling(
                    bert_output, batch["attention_mask"]
                )
                sentence_vectors.append(sentence_vector.cpu().detach().numpy())
        sentence_vectors = np.vstack(sentence_vectors)
        return sentence_vectors


def result_to_markdown_table(df_result):
    result = "\n".join(
        f"|{r.text}|{round(r.similarity, 3)}|" for _, r in df_result.iterrows()
    )
    return f"|text|similarity|\n|---|---|\n{result}"


def search_and_show_results(query, top_k=10):
    # baseline_result = baseline_seacher.search(query).head(top_k)
    sentence_bert_result = sentence_bert_seacher.search(query).head(top_k)
    # bert_result = bert_seacher.search(query).head(top_k)

    print(f"## クエリ: {query}")

    # print("### baseline")
    # print(result_to_markdown_table(baseline_result))
    # print()

    print("### SBERT")
    print(result_to_markdown_table(sentence_bert_result))
    print()

    # print("### BERT")
    # print(result_to_markdown_table(bert_result))
    # print()


if __name__ == "__main__":
    # テストデータを読み込んで検索データとして使う
    test_texts = load_test_texts()

    # 各手法のSearcherを作成
    sentence_bert_seacher = SentenceBertSearcher(test_texts)

    # 検索実行
    search_and_show_results("湖の側の草むらで、３頭の馬が草を食べています。")
    search_and_show_results("海の近くの牧場で、牛たちが何かを食べています。")  # 似たような文章だが単語が違う
    search_and_show_results("湖の側の草むらで、３頭の馬が草を食べていません。")  # 否定形
