from flask import Flask, render_template_string, jsonify, request, send_file
import sys
import json
import os
import threading
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
import uuid

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import re

# Document processing imports
from pypdf import PdfReader
try:
    from docx import Document as DocxDocument
except ImportError:
    # Fallback if python-docx is not installed
    DocxDocument = None
import io

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

load_dotenv(override=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploaded_documents'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# LLM Configuration - Using Ollama
env_path = r"C:\Users\Cem\Desktop\projeler\YapayZekaUygulamalari\.env"
load_dotenv(env_path)

# Ollama Configuration
ollama_base_url = "http://localhost:11434/v1"
ollama_model = "llama3.2"
ollama_llm = None

try:
    ollama_llm = ChatOllama(temperature=0, model=ollama_model)
    print("✅ Ollama client ready")
except Exception as e:
    print(f"⚠️  Ollama not available: {e}")

# RAG Configuration
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
device = 'cpu'
RETRIEVAL_K = 10
db_name = "vector_db_uploaded_faiss"

HOLY_PROMPT = """You are a knowledgeable, friendly assistant.
You are chatting with a user about documents they have uploaded.
If relevant, use the given context from the uploaded documents to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""
print("=" * 60)
print("💬 Upload RAG Chatbot - Initializing")
print("=" * 60)

# Initialize embeddings
print(f"\n⏳ Loading embedding model: {EMBEDDING_MODEL}...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embedding model loaded")

# Load or create vectorstore
print(f"\n💾 Loading vectorstore: {db_name}...")
vector_store = None
if os.path.exists(db_name) and os.path.exists(os.path.join(db_name, "index.faiss")):
    try:
        vector_store = FAISS.load_local(
            folder_path=db_name,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        doc_count = len(vector_store.docstore._dict)
        print(f"✅ Vectorstore loaded ({doc_count} documents)")
    except Exception as e:
        print(f"⚠️  Error loading vectorstore: {e}. Creating new one.")
        vector_store = FAISS.from_texts([""], embeddings)
        vector_store.save_local(db_name)
        print("✅ New vectorstore created")
else:
    print("📝 Creating new vectorstore...")
    vector_store = FAISS.from_texts([""], embeddings)
    vector_store.save_local(db_name)
    print("✅ New vectorstore created")

# Thread lock for vectorstore operations
_vectorstore_lock = threading.Lock()

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

print("\n✅ Upload RAG System ready!")
print("=" * 60)

def process_pdf(file_content: bytes, filename: str) -> list[Document]:
    """Extract text from PDF file"""
    documents = []
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
        
        if full_text.strip():
            # Split into chunks
            chunks = text_splitter.split_text(full_text)
            
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "file_type": "pdf",
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "upload_date": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
        
        print(f"✅ Processed PDF: {filename} - {len(documents)} chunks")
    except Exception as e:
        print(f"❌ Error processing PDF {filename}: {e}")
        raise
    
    return documents

def process_docx(file_content: bytes, filename: str) -> list[Document]:
    """Extract text from Word document"""
    if DocxDocument is None:
        raise ImportError("python-docx is not installed. Please install it with: pip install python-docx")
    
    documents = []
    try:
        docx_file = io.BytesIO(file_content)
        doc = DocxDocument(docx_file)
        
        full_text = ""
        for para in doc.paragraphs:
            if para.text.strip():
                full_text += para.text + "\n"
        
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                if row_text.strip():
                    full_text += row_text + "\n"
        
        if full_text.strip():
            # Split into chunks
            chunks = text_splitter.split_text(full_text)
            
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "file_type": "docx",
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "upload_date": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
        
        print(f"✅ Processed DOCX: {filename} - {len(documents)} chunks")
    except Exception as e:
        print(f"❌ Error processing DOCX {filename}: {e}")
        raise
    
    return documents

def process_txt(file_content: bytes, filename: str) -> list[Document]:
    """Extract text from plain text file"""
    documents = []
    try:
        text = file_content.decode('utf-8', errors='ignore')
        
        if text.strip():
            # Split into chunks
            chunks = text_splitter.split_text(text)
            
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "file_type": "txt",
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "upload_date": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
        
        print(f"✅ Processed TXT: {filename} - {len(documents)} chunks")
    except Exception as e:
        print(f"❌ Error processing TXT {filename}: {e}")
        raise
    
    return documents

def process_uploaded_file(file) -> list[Document]:
    """Process uploaded file based on its extension"""
    filename = file.filename
    file_content = file.read()
    
    # Determine file type
    file_ext = Path(filename).suffix.lower()
    
    if file_ext == '.pdf':
        return process_pdf(file_content, filename)
    elif file_ext in ['.docx', '.doc']:
        return process_docx(file_content, filename)
    elif file_ext == '.txt':
        return process_txt(file_content, filename)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

def add_documents_to_vectorstore(documents: list[Document]):
    """Add documents to the vector store"""
    if not documents:
        return
    
    try:
        with _vectorstore_lock:
            # Add documents to existing vectorstore
            vector_store.add_documents(documents)
            # Save the updated vectorstore
            vector_store.save_local(db_name)
        
        print(f"✅ Added {len(documents)} document chunks to vectorstore")
    except Exception as e:
        print(f"❌ Error adding documents to vectorstore: {e}")
        raise