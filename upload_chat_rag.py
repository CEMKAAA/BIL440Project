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