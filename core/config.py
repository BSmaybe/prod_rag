import os

# Настройки Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_k_m"  # Твоя модель

# Настройки Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "bank_tickets"

# Настройки эмбеддингов (локально на CPU)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
