from collections import Counter
from pathlib import Path

from haystack_integrations.document_stores.chroma import ChromaDocumentStore

store = ChromaDocumentStore(collection_name="tesco_car_insurance",
                            persist_path="chroma_db", distance_function="cosine")
docs = store.filter_documents()

counts = Counter(Path(d.meta.get("file_path", "unknown")).name for d in docs)
for name, n in sorted(counts.items()):
    print(f"{n:3} chunks  {name}")
print(f"\n{len(counts)} files, {len(docs)} chunks")