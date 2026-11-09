"""
Truy hồi câu hỏi giống nhau (cho bước 3: ví dụ mẫu khi LLM viết đặc tả, và đoán dạng câu).

- Có bge-m3 (thư mục model offline, cần sentence-transformers): vector dày, cosine.
- Không có: TF-IDF ký tự 2–4 gram trên câu đã bỏ dấu — chạy CPU, không cần tải gì.
Kho câu CHỈ lấy từ TRAIN (lexicon/spec_bank.jsonl — câu train mà máy tính đã trả lời KHỚP đáp án train).
"""
import json
import os
import re
import unicodedata

import numpy as np


def key(s):
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d"))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"\d+([/.,]\d+)*", "0", s)          # số / ngày -> 0: câu giống nhau về DẠNG, không về con số
    return re.sub(r"\s+", " ", s).strip()


class Retriever:
    def __init__(self, texts, meta=None, model_path=None, device=None):
        self.texts = list(texts)
        self.meta = list(meta) if meta is not None else [{} for _ in self.texts]
        self.kind = "tfidf"
        self.model = None
        if model_path and os.path.isdir(model_path):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_path, device=device)
                self.M = self._enc(self.texts)
                self.kind = "bge-m3"
            except Exception as e:                      # thiếu thư viện / model hỏng -> TF-IDF
                print(f"  [retriever] không nạp được {model_path}: {e} -> dùng TF-IDF", flush=True)
                self.model = None
        if self.model is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
            self.M = self.vec.fit_transform([key(t) for t in self.texts])

    def _enc(self, xs):
        return np.asarray(self.model.encode(list(xs), batch_size=64, normalize_embeddings=True, show_progress_bar=False))

    def scores(self, query):
        if self.model is not None:
            return self.M @ self._enc([query])[0]
        q = self.vec.transform([key(query)])
        return (self.M @ q.T).toarray().ravel()

    def topk(self, query, k=4, exclude=None):
        """[(điểm, chỉ số)] giảm dần; exclude(meta) -> True thì bỏ (vd cùng shop khi đánh giá)."""
        s = self.scores(query)
        order = np.argsort(-s)
        out = []
        for i in order:
            if exclude is not None and exclude(self.meta[i]):
                continue
            out.append((float(s[i]), int(i)))
            if len(out) >= k:
                break
        return out


def load_bank(path):
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


_RET = {}


def bank_retriever(bank_path, model_path=None):
    """Retriever dùng chung (nạp một lần)."""
    k = (bank_path, model_path)
    if k not in _RET:
        bank = load_bank(bank_path)
        _RET[k] = Retriever([b["question"] for b in bank], bank, model_path=model_path) if bank else None
    return _RET[k]


def few_shot_text(ret, query, k=4, min_score=0.0):
    """Các ví dụ (câu train + đặc tả JSON đã kiểm) gần nhất để đưa vào prompt LLM viết đặc tả."""
    if ret is None:
        return ""
    lines = []
    for sc, i in ret.topk(query, k):
        if sc < min_score:
            continue
        b = ret.meta[i]
        lines.append(f"- Câu: {b['question']}\n  Hôm nay: {b['as_of']}\n  Đặc tả: {json.dumps(b['spec'], ensure_ascii=False)}")
    return "\n".join(lines)
