import pickle
import pandas as pd
from rank_bm25 import BM25Okapi

df = pd.read_excel("data/faq_clean.xlsx", engine='openpyxl')

tokenized_corpus = [
    (row['question'] + " " + row['answer']).lower().split()
    for _, row in df.iterrows()
]

bm25 = BM25Okapi(tokenized_corpus)

with open("data/faq_bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("BM25 index yaradıldı: data/faq_bm25.pkl")