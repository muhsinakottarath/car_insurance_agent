from pathlib import Path

from haystack import Pipeline
from haystack.components.routers import FileTypeRouter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, HTMLToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# Raw files will be embedded and stored to output path
raw_docs = Path("Knowledge_base/tesco_car_insurance_kb")
output_path = Path("./kn.json")
db_path = "chroma_db"
collection_name = "tesco_car_insurance"
embedding_model = "nomic-embed-text"
chunk_size = 150
chunk_overlap = 30


def find_files(folder):
    """Return all supported files, skipping the README."""
    supported = {".md", ".pdf", ".html"}
    files = [f for f in sorted(folder.iterdir())
             if f.suffix in supported and f.name != "README.md"]
    if not files:
        raise FileNotFoundError(f"No .md, .pdf or .html files found in {folder}")
    return files


def get_document_store():
    """Connect to the Chroma database on disk, creating it if needed."""
    return ChromaDocumentStore(
        collection_name=collection_name,
        persist_path=db_path,
        distance_function="cosine",   # standard choice for text embeddings
    )

def clear_store(document_store):
    """Remove old chunks so a re-run doesn't leave out-of-date content behind."""
    old_ids = [doc.id for doc in document_store.filter_documents()]
    if old_ids:
        document_store.delete_documents(old_ids)
        print(f"Removed {len(old_ids)} old chunks")


def build_pipeline(document_store):
    """Router -> converters -> joiner -> cleaner -> splitter -> embedder -> writer."""
    pipeline = Pipeline()

    pipeline.add_component("router", FileTypeRouter(
        mime_types=["text/markdown", "application/pdf", "text/html"],
        additional_mimetypes={"text/markdown": ".md"},
    ))
    pipeline.add_component("markdown_converter", MarkdownToDocument())
    pipeline.add_component("pdf_converter", PyPDFToDocument())
    pipeline.add_component("html_converter", HTMLToDocument())
    pipeline.add_component("joiner", DocumentJoiner())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(
        split_by="word", split_length=chunk_size, split_overlap=chunk_overlap))
    pipeline.add_component("embedder", OllamaDocumentEmbedder(model=embedding_model))
    pipeline.add_component("writer", DocumentWriter(
        document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

    pipeline.connect("router.text/markdown", "markdown_converter.sources")
    pipeline.connect("router.application/pdf", "pdf_converter.sources")
    pipeline.connect("router.text/html", "html_converter.sources")
    pipeline.connect("markdown_converter", "joiner")
    pipeline.connect("pdf_converter", "joiner")
    pipeline.connect("html_converter", "joiner")
    pipeline.connect("joiner", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    return pipeline


def main():
    files = find_files(raw_docs)
    print(f"Found {len(files)} files in {raw_docs}")

    document_store = get_document_store()
    clear_store(document_store)

    pipeline = build_pipeline(document_store)
    print("Indexing... (embedding can take a minute)")
    pipeline.run({"router": {"sources": files}})

    print(f"Stored {document_store.count_documents()} chunks in {db_path}/")


if __name__ == "__main__":
    main()