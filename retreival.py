import os
import re
import json
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI

# Paths
DATA_PATH = "data/"
VECTOR_DB_PATH = "vector_db/"

# Load PDF documents
def load_documents() -> List[Document]:
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_file"] = file
            docs.extend(documents)
    print(f"✅ Loaded {len(docs)} pages from PDF files.")
    return docs

# Semantic chunking
def semantic_split_documents(documents: List[Document]) -> List[Document]:
    split_docs = []
    for doc in documents:
        content = doc.page_content.strip()
        chunks = re.split(r'\n(?=[A-Z0-9][^\n]{1,100})', content)
        for chunk_text in chunks:
            if not chunk_text.strip():
                continue
            new_doc = Document(page_content=chunk_text, metadata=dict(doc.metadata))
            first_line = chunk_text.split("\n")[0]
            new_doc.metadata["section"] = first_line if len(first_line) < 100 else "General"
            words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", chunk_text)
            new_doc.metadata["keywords"] = ", ".join(list(set(words))[:10])
            split_docs.append(new_doc)
    print(f"✅ Semantic split into {len(split_docs)} chunks.")
    return split_docs

# Store embeddings in Chroma
def store_embeddings(split_docs: List[Document]):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    for doc in split_docs:
        for key, value in list(doc.metadata.items()):
            if isinstance(value, (list, dict)):
                doc.metadata[key] = json.dumps(value)
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )
    vectordb.persist()
    print("✅ Embeddings stored in ChromaDB.")
    return vectordb

# Build BM25 index
def build_bm25_index(split_docs: List[Document]):
    corpus = [doc.page_content for doc in split_docs]
    tokenized_corpus = [text.split() for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus

# Hybrid retrieval
def retrieve_hybrid(query: str, split_docs: List[Document], bm25: BM25Okapi, corpus: List[str], top_k: int = 10):
    tokenized_query = query.split()
    bm25_results = bm25.get_top_n(tokenized_query, corpus, n=top_k)
    candidate_docs = [doc for doc in split_docs if doc.page_content in bm25_results]
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs)
    ranked_docs = [doc for _, doc in sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)]
    return ranked_docs

# LLM answer generation
def answer_query_with_llm(query: str, top_docs: List[Document], max_tokens: int = 500):
    context = "\n\n".join([doc.page_content for doc in top_docs])
    prompt = f"""
You are a helpful assistant. Answer the question based on the following context:

Context:
{context}

Question:
{query}
"""
    client = Groq(api_key = "enter your API key")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# Build full pipeline
def build_pipeline():
    documents = load_documents()
    split_docs = semantic_split_documents(documents)
    vectordb = store_embeddings(split_docs)
    bm25, corpus = build_bm25_index(split_docs)
    print("🚀 Hybrid RAG Pipeline ready!")
    return split_docs, bm25, corpus

# Interactive Q&A loop
def interactive_qa(split_docs, bm25, corpus):
    print("🟢 Enter 'exit' to quit the interactive session.")
    while True:
        query = input("\n💬 Your query: ").strip()
        if query.lower() == "exit":
            print("🛑 Exiting Q&A session.")
            break
        top_docs = retrieve_hybrid(query, split_docs, bm25, corpus, top_k=5)
        answer = answer_query_with_llm(query, top_docs)
        print("\n📝 Answer:\n", answer)

# Main
if __name__ == "__main__":
    split_docs, bm25, corpus = build_pipeline()
    interactive_qa(split_docs, bm25, corpus)

