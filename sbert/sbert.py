# はじめての自然言語処理
# 第9回 Sentence BERT による類似文章検索の検証
# https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part9.html
# https://www.subcul-science.com/post/20210203sbert/ (fixed version)

import transformers
from transformers import BertJapaneseTokenizer, BertModel

BertTokenizer = transformers.BertJapaneseTokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers import models

transformer = models.Transformer("cl-tohoku/bert-base-japanese-whole-word-masking")

pooling = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)
model = SentenceTransformer(modules=[transformer, pooling])

sentences = ["吾輩は猫である", "本日は晴天なり"]
embeddings = model.encode(sentences)

for i, embedding in enumerate(embeddings):
    print(
        "[%d] : %s"
        % (
            i,
            embedding.shape,
        )
    )

# [0] : (768,)
# [1] : (768,)
