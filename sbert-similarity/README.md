# 【日本語モデル付き】2020年に自然言語処理をする人にお勧めしたい文ベクトルモデル

[【日本語モデル付き】2020年に自然言語処理をする人にお勧めしたい文ベクトルモデル](https://qiita.com/sonoisa/items/1df94d0a98cd4f209051)
の実装です.

## setup

[あなたの文章に合った「いらすとや」画像をレコメンド♪（アルゴリズム解説編）](https://qiita.com/sonoisa/items/775ac4c7871ced6ed4c3)
から「いらすとや」さんの画像メタデータをダウンロード(ライセンスを必ず確認すること)

```bash
gdown 1DZjgYCda82IYAUhbmsJin5A8y9Y-lNyw
```

## run

venvは必要に応じて.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 similarity.py
```
