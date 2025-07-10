# rag_retriever.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer

EMBED_MODEL = 'all-MiniLM-L6-v2'
INDEX_PATH = 'rag_index.index'
META_PATH = 'rag_meta.pkl'

model = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)

# all_chunks: List[str]
# metadata: List[str]，如 "xxx.pdf 第3段"
with open(META_PATH, 'rb') as f:
    all_chunks, metadata = pickle.load(f)

def retrieve_top_k(question, k=5):
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, k)

    top_chunks = []
    for i in indices[0]:
        source = metadata[i]
        content = all_chunks[i]
        formatted = f"[{source}] {content.strip()}"
        top_chunks.append(formatted)

    return top_chunks
