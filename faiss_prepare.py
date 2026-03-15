import numpy as np
import faiss

embeddings = np.load("data/faq_embeddings.npy")
embedding_dim = embeddings.shape[1]

faiss_index = faiss.IndexFlatIP(embedding_dim)

faiss.normalize_L2(embeddings)

faiss_index.add(embeddings)

faiss.write_index(faiss_index, "data/faq_faiss.index")
print("FAISS index yaradıldı: data/faq_faiss.index")