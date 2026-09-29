from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("API key loaded", api_key is not None)

from langchain_community.document_loaders import TextLoader
loader = TextLoader("knowledge.txt")
documents = loader.load()
print("documents loaded",len(documents))

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)
chunks = splitter.split_documents(documents)
print("numebr of chunks:", len(chunks))

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model launched")

from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory="./chroma_db"
)
print("Vector database created")
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever created")

question = "when is customer support available?"
results = retriever.invoke(question)
print("retireved info:")
for result in results:
    print(result.page_content)

from langchain_groq import ChatGroq
llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature=0

)
print("LLM created")

context = "\n\n".join(
    result.page_content for result in results
)

print("Context created")

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer is not in the context, say you don't know.
"""
response = llm.invoke(prompt)

print("\nFinal Answer:")
print(response.content)