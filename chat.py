import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 🔑 API key (Gemini OpenAI-compatible)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")

# 🔹 Embeddings (must match indexing)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔹 Load existing Qdrant collection
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model,
)

# 🔹 Gemini client (OpenAI-compatible)
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 🔹 User query
user_query = input("Ask something: ")

# 🔹 Retrieve context
search_results = vector_db.similarity_search(query=user_query, k=4)

context = "\n\n".join(
    [
        f"""
Source: {res.metadata.get('source')}
Page: {res.metadata.get('page_label')}
Content:
{res.page_content}
"""
        for res in search_results
    ]
)

# 🔹 System prompt WITH injected context
SYSTEM_PROMPT = f"""
You are a helpful AI assistant.
Answer the user's question strictly using the provided context.
If the answer is not in the context, say "I don't know based on the document".

Guide the user to the correct page number for more details.

Context:
{context}
"""

# 🔹 Call Gemini
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)

print("\n[BOT]:", response.choices[0].message.content)
