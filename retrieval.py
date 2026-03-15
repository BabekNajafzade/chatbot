import pickle
import numpy as np
import faiss
import pandas as pd
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_excel("data/faq_clean.xlsx", engine='openpyxl')

with open("data/faq_bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)


faiss_index = faiss.read_index("data/faq_faiss.index")
embeddings = np.load("data/faq_embeddings.npy")


def refine_query_llm(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sən e-gov.az FAQ axtarış sistemi üçün query rewriting mütəxəssisisən. "
                    "İstifadəçinin sualını FAQ bazasında daha yaxşı nəticə verəcək formaya çevir. "
                    "Qısa, aydın, tam cümlə şəklində yaz. "
                    "Yalnız yenidən yazılmış sualı qaytar, heç bir izahat əlavə etmə."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def get_query_embedding(query: str) -> np.ndarray:
    resp = client.embeddings.create(input=query, model="text-embedding-3-small")
    vec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def hybrid_retrieval(query: str, top_k: int = 5, faiss_weight: float = 0.6, bm25_weight: float = 0.4):
    refined_query = refine_query_llm(query)

    tokenized_query = refined_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(-bm25_scores)[:top_k]

    query_vec = get_query_embedding(refined_query)
    _, faiss_indices_result = faiss_index.search(query_vec, top_k)
    faiss_indices = faiss_indices_result[0]

    faiss_repeats = int(faiss_weight * 10)
    bm25_repeats = int(bm25_weight * 10)

    ensemble_list = list(faiss_indices) * faiss_repeats + list(bm25_indices) * bm25_repeats
    ensemble_indices = list(dict.fromkeys(ensemble_list))

    results = []
    for i in ensemble_indices[:top_k]:
        if 0 <= i < len(df):
            results.append({
                "question": df.iloc[i]['question'],
                "answer": df.iloc[i]['answer']
            })
    return results


if __name__ == "__main__":
    test_query = "ödəniş"
    print(f"Orijinal sorğu: {test_query}")
    refined = refine_query_llm(test_query)
    print(f"Yaxşılaşdırılmış sorğu: {refined}")
    results = hybrid_retrieval(test_query, top_k=5)
    for i, res in enumerate(results, 1):
        print(f"{i}. Q: {res['question']}")
        print(f"   A: {res['answer']}\n")