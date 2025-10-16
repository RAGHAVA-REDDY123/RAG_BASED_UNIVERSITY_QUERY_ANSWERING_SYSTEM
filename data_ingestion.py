# # 🧩 Step 1: Imports
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
# import os
# import re
# import json

# # 🧩 Step 2: Define data folder and database folder
# DATA_PATH = "data/"
# VECTOR_DB_PATH = "vector_db/"

# # 🧩 Step 3: Load PDF documents
# def load_documents():
#     docs = []
#     for file in os.listdir(DATA_PATH):
#         if file.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(DATA_PATH, file))
#             documents = loader.load()

#             # Add metadata
#             for doc in documents:
#                 doc.metadata["source_file"] = file

#             docs.extend(documents)
#     print(f"✅ Loaded {len(docs)} pages from PDF files.")
#     return docs

# # 🧩 Step 4: Improved split documents into chunks with metadata
# def split_documents(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=250,
#         separators=["\n\n", "\n", ".", "!", "?"]
#     )

#     split_docs = text_splitter.split_documents(documents)

#     for doc in split_docs:
#         content = doc.page_content.strip()

#         # Section detection
#         first_line = content.split("\n")[0]
#         doc.metadata["section"] = first_line if len(first_line) < 100 else "General"

#         # Keywords extraction
#         words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", content)
#         keywords = list(set(words))[:10]
#         doc.metadata["keywords"] = ", ".join(keywords)  # ✅ convert to string

#         # Category detection
#         if "admission" in doc.metadata["source_file"].lower():
#             doc.metadata["category"] = "Admissions"
#         elif "guidelines" in doc.metadata["source_file"].lower():
#             doc.metadata["category"] = "Rules"
#         else:
#             doc.metadata["category"] = "General"

#         # Page number fallback
#         doc.metadata["page_number"] = doc.metadata.get("page", "Unknown")

#     print(f"✅ Split into {len(split_docs)} chunks with enriched metadata.")
#     return split_docs

# # 🧩 Step 5: Generate embeddings and store in ChromaDB
# def store_embeddings(split_docs):
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Ensure all metadata values are valid types
#     for doc in split_docs:
#         for key, value in list(doc.metadata.items()):
#             if isinstance(value, (list, dict)):
#                 doc.metadata[key] = json.dumps(value)  # safe fallback to string

#     vectordb = Chroma.from_documents(
#         documents=split_docs,
#         embedding=embedding_model,
#         persist_directory=VECTOR_DB_PATH
#     )
#     vectordb.persist()
#     print("✅ Embeddings generated and stored successfully!")


# # 🧩 Step 6: Main pipeline
# def build_ingestion_pipeline():
#     documents = load_documents()
#     split_docs = split_documents(documents)
#     store_embeddings(split_docs)
#     print("🚀 Data Ingestion Pipeline Completed!")

# # 🧩 Step 7: Execute the pipeline
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectordb = Chroma(persist_directory="vector_db", embedding_function=embedding_model)

# print("✅ Vector DB contains:", vectordb._collection.count(), "documents.")

# if __name__ == "__main__":
#     build_ingestion_pipeline()


# 🧩 Step 1: Imports
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

# 🧩 Step 2: Paths
DATA_PATH = "data/"
VECTOR_DB_PATH = "vector_db/"

# 🧩 Step 3: Load PDF documents
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

# 🧩 Step 4: Semantic chunking
def semantic_split_documents(documents: List[Document]) -> List[Document]:
    split_docs = []
    
    for doc in documents:
        content = doc.page_content.strip()
        chunks = re.split(r'\n(?=[A-Z0-9][^\n]{1,100})', content)
        
        for chunk_text in chunks:
            if not chunk_text.strip():
                continue
            
            new_doc = Document(page_content=chunk_text, metadata=dict(doc.metadata))
            
            # Section
            first_line = chunk_text.split("\n")[0]
            new_doc.metadata["section"] = first_line if len(first_line) < 100 else "General"
            
            # Keywords
            words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", chunk_text)
            new_doc.metadata["keywords"] = ", ".join(list(set(words))[:10])
            
            split_docs.append(new_doc)
    
    print(f"✅ Semantic split into {len(split_docs)} chunks.")
    return split_docs

# 🧩 Step 5: Store embeddings in Chroma
def store_embeddings(split_docs: List[Document]):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Ensure metadata is serializable
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

# 🧩 Step 6: Build BM25 index
def build_bm25_index(split_docs: List[Document]):
    corpus = [doc.page_content for doc in split_docs]
    tokenized_corpus = [text.split() for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus

# 🧩 Step 7: Hybrid retrieval with BM25 + embeddings reranking
def retrieve_hybrid(query: str, split_docs: List[Document], bm25: BM25Okapi, corpus: List[str],
                    top_k: int = 10):
    # Step 1: BM25 sparse retrieval
    tokenized_query = query.split()
    bm25_results = bm25.get_top_n(tokenized_query, corpus, n=top_k)
    
    # Map BM25 text results back to Document objects
    candidate_docs = [doc for doc in split_docs if doc.page_content in bm25_results]

    # Step 2: Rerank using CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs)
    
    # Sort candidates by reranker score
    ranked_docs = [doc for _, doc in sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)]
    return ranked_docs

# 🧩 Step 8: Main pipeline
def build_pipeline():
    # Load + chunk PDFs
    documents = load_documents()
    split_docs = semantic_split_documents(documents)
    
    # Store embeddings in ChromaDB
    vectordb = store_embeddings(split_docs)
    
    # Build BM25 index
    bm25, corpus = build_bm25_index(split_docs)
    
    print("🚀 Hybrid RAG Pipeline ready!")
    return split_docs, bm25, corpus

# 🧩 Step 9: Run pipeline and query
if __name__ == "__main__":
    split_docs, bm25, corpus = build_pipeline()
    
    query = "any contact number for admissions queries"
    top_docs = retrieve_hybrid(query, split_docs, bm25, corpus, top_k=5)
    
    print("🔎 Top 5 relevant chunks:")
    for i, doc in enumerate(top_docs, 1):
        print(f"{i}. Section: {doc.metadata['section']}, Keywords: {doc.metadata['keywords']}")
        print(f"Content: {doc.page_content[:300]}...\n")

