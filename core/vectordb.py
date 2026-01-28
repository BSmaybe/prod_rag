from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from .config import QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL

# Инициализация (Singleton)
client = QdrantClient(url=QDRANT_URL)
# Модель грузим в память один раз (она легкая, ~300Мб)
embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

def search_tickets(query: str, top_k: int = 5):
    """
    Превращает текст в вектор и ищет в Qdrant.
    """
    # 1. Генерируем вектор запроса
    # Добавляем префикс "query: ", так как модель e5-small этого требует
    query_vec = embedder.encode(f"query: {query}", normalize_embeddings=True).tolist()

    # 2. Поиск
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k
    )

    # 3. Упрощаем формат для LLM
    # Возвращаем список строк: "Ticket-123: Текст..."
    context_parts = []
    for hit in results:
        payload = hit.payload
        # Формируем сниппет как в RGA-SD
        part = f"Тикет {payload.get('issue_key', 'N/A')}:\n{payload.get('text_chunk', '')}"
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)