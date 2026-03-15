import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_excel("data/faq_clean.xlsx", engine='openpyxl')
texts = (df['question'] + " " + df['answer']).tolist()

print(f"Cəmi {len(texts)} sətir tapıldı. Embeddings yaradılır...")

embeddings = []
for i, text in enumerate(texts):
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    embeddings.append(resp.data[0].embedding)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(texts)} hazırdır...")

np.save("data/faq_embeddings.npy", np.array(embeddings, dtype=np.float32))
print("Embeddings saxlanıldı: data/faq_embeddings.npy")