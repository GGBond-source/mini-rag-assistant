import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 让 Python 能找到 data/documents.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
if DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)

from documents import DOCUMENTS


class Retriever:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.documents = DOCUMENTS
        self.model = SentenceTransformer(model_name)

        # 编码文档
        self.doc_embeddings = self.model.encode(
            self.documents,
            normalize_embeddings=True
        )
        self.doc_embeddings = np.array(self.doc_embeddings, dtype=np.float32)

        # 建立 FAISS 索引（内积 = 归一化后相当于余弦相似度）
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embeddings)

    def search_documents(self, question, top_k=5):
        """
        向量检索
        返回：
        [
            {"id": 0, "text": "...", "score": 0.91},
            ...
        ]
        """
        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "id": int(idx),
                "text": self.documents[idx],
                "score": float(score)
            })

        return results

    def rerank_documents(self, question, recalled_docs):
        """
        Day1 的轻量 rerank：
        在原始检索分数基础上，加一点关键词匹配分
        """
        question_chars = set(question.replace("？", "").replace("?", "").strip())

        reranked = []
        for doc in recalled_docs:
            text = doc["text"]

            overlap = 0
            for ch in question_chars:
                if ch and ch in text:
                    overlap += 1

            rerank_score = doc["score"] + 0.01 * overlap

            item = dict(doc)
            item["rerank_score"] = float(rerank_score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked