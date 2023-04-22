from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch.nn.functional as F
import json
from pathlib import Path
import random
import torch


def predict(model_dir, test_txt):

    label2id_path = model_dir / "label2id.json"
    label2id = json.loads(label2id_path.read_text())
    id2label = {value: key for key, value in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    with open(test_txt) as f:
        test_dataset = f.read().split("\n")

    test_label, test_text = random.choice(test_dataset).split("\t")
    print(f"{test_label}:\n{test_text}")

    encoded_input = tokenizer(
        test_text, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    output = model(
        input_ids=encoded_input["input_ids"],
        attention_mask=encoded_input["attention_mask"],
    )
    logits = output.logits
    # sigmoid関数を適用して確率に変換する
    probs = F.sigmoid(logits)
    print(probs)
    max_prob_index = torch.argmax(probs, dim=1)
    print(max_prob_index)
    print(id2label[max_prob_index.item()])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--test-txt", help="test dataset", required=True)
    parser.add_argument("--model-dir", help="model dir", required=True)

    args = parser.parse_args()
    predict(Path(args.model_dir), Path(args.test_txt))
