import os

# Настройки Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_k_m")

# Настройки Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "bank_tickets"

# Настройки эмбеддингов (локально на CPU)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Опциональный API ключ для вызова /process_ticket
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")

# Настройки Qdrant (вектор и метрика)
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")
