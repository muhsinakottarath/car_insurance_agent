from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# ---------- Settings (must match indexing.py) ----------
db_path = "chroma_db"
collection_name = "tesco_car_insurance"
embedding_model = "nomic-embed-text"
chat_model = "llama3.2"
top_k = 3   # how many chunks to retrieve

SYSTEM_PROMPT = """You are a customer support assistant for a car insurance company.
Answer ONLY using the information in the documents provided.
If the documents do not contain the answer, reply exactly:
"I'm not sure about that. I'll pass your question to a member of our team."
Keep answers short, friendly and accurate. Do not make up phone numbers, prices or policy details."""

USER_TEMPLATE = """Documents:
{% for doc in documents %}
---
{{ doc.content }}
{% endfor %}

Customer question: {{ question }}"""


def build_pipeline():
    document_store = ChromaDocumentStore(
        collection_name=collection_name,
        persist_path=db_path,
        distance_function="cosine",
    )

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", OllamaTextEmbedder(model=embedding_model))
    pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store, top_k=top_k))
    pipeline.add_component("prompt_builder", ChatPromptBuilder(
        template=[ChatMessage.from_system(SYSTEM_PROMPT), ChatMessage.from_user(USER_TEMPLATE)],
        required_variables=["documents", "question"],
    ))
    pipeline.add_component("llm", OllamaChatGenerator(model=chat_model))

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    return pipeline


def ask(pipeline, question):
    result = pipeline.run(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}},
        include_outputs_from={"retriever"},   # also return the retrieved chunks
    )
    answer = result["llm"]["replies"][0].text
    documents = result["retriever"]["documents"]
    return answer, documents


def main():
    pipeline = build_pipeline()
    print("Ask a car insurance question (type 'quit' to stop)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        answer, documents = ask(pipeline, question)
        print(f"\nBot: {answer}\n")
        print("Sources used:")
        for doc in documents:
            name = doc.meta.get("file_path", "unknown").split("/")[-1]
            print(f"  score={doc.score:.3f}  {name}")
        print()


if __name__ == "__main__":
    main()