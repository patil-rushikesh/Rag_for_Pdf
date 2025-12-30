import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

print("PDF path:", pdf_path)
print("Exists?", pdf_path.exists())

if not pdf_path.exists():
    raise FileNotFoundError("PDF file not found!")

loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)

chunks = text_splitter.split_documents(docs)

key = os.getenv("GOOGLE_API_KEY")


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag",
)

print("✅ Indexing completed successfully")
