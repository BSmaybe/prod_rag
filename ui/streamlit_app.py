from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8080")


def call_ask(api_url: str, issue_text: str, context_count: int, service: str) -> Dict[str, Any]:
    data = {"issue_text": issue_text, "context_count": str(context_count)}
    if service.strip():
        data["service"] = service.strip()

    r = requests.post(f"{api_url.rstrip('/')}/ask", data=data, timeout=(10, 180))
    r.raise_for_status()
    return r.json()


def ensure_api_ready(api_url: str) -> None:
    """Проверяем доступность API перед отправкой запроса.

    Это позволяет отлавливать ситуацию, когда контейнер ещё не стартовал,
    PostgreSQL недоступен или API зависает при инициализации, вместо того
    чтобы ждать таймаута в 180 секунд.
    """

    url = f"{api_url.rstrip('/')}/readyz"
    try:
        requests.get(url, timeout=5).raise_for_status()
    except requests.RequestException as exc:
        raise requests.ConnectionError(
            f"API недоступно по адресу {url}. Убедитесь, что сервис rag-api запущен и готов."
        ) from exc


def call_feedback(
    api_url: str,
    query: str,
    answer_full_text: str,
    is_helpful: bool,
    comment: Optional[str],
    used_issue_keys: Optional[list[str]],
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "answer_full_text": answer_full_text,
        "is_helpful": is_helpful,
        "comment": comment,
        "used_issue_keys": used_issue_keys or [],
    }

    # ВАЖНО: endpoint у тебя /feedback/ (со слэшем)
    r = requests.post(f"{api_url.rstrip('/')}/feedback/", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def _as_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    return [str(val)]


def extract_sections(resp: Dict[str, Any]) -> Dict[str, List[str]]:
    base = resp.get("structured") if isinstance(resp.get("structured"), dict) else resp

    desc = _as_list(base.get("description") or base.get("desc"))
    causes = _as_list(base.get("causes") or base.get("root_causes") or base.get("cause"))
    actions = _as_list(base.get("actions") or base.get("steps") or base.get("recommendations"))
    next_steps = _as_list(base.get("next_steps") or base.get("next") or base.get("followups"))

    if not any([desc, causes, actions, next_steps]):
        full = resp.get("full_text") or resp.get("answer") or resp.get("result") or ""
        actions = _as_list(full)

    return {
        "Описание": desc,
        "Причины": causes,
        "Действия": actions,
        "Следующие шаги": next_steps,
    }


def format_comment(sections: Dict[str, List[str]]) -> str:
    parts: List[str] = []
    for title, items in sections.items():
        if not items:
            continue
        parts.append(f"**{title}:**")
        for i, it in enumerate(items, 1):
            parts.append(f"{i}. {it}")
        parts.append("")
    return "\n".join(parts).strip()


def init_state() -> None:
    defaults = {
        "last_resp": None,
        "last_sections": None,
        "last_answer_full_text": "",
        "last_used_issue_keys": [],
        "last_used_chunks": [],
        "last_feedback_id": None,
        "last_feedback_msg": "",
        "last_feedback_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    st.set_page_config(page_title="RAG Agent UI", layout="wide")
    st.title("RAG Agent — помощник инженера поддержки")

    init_state()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Настройки")

        st.text_input(
            "API URL",
            value=DEFAULT_API_URL,
            key="api_url",
            help="Пример: http://localhost:8080 или http://rag-api:8080",
        )
        st.text_input("Service (опционально)", value="", key="service")
        st.slider("Контекст (top_k)", min_value=1, max_value=50, value=20, key="context_count")
        st.caption("UI не индексирует тикеты. Все данные берутся из твоей БД через API.")

        # маленький DEBUG индикатор, чтобы понимать теряется ли state
        st.caption("DEBUG: session_state сохранён?")
        st.write(
            {
                "has_last_resp": st.session_state.last_resp is not None,
                "last_feedback_id": st.session_state.last_feedback_id,
            }
        )

    st.subheader("Поиск и генерация ответа")

    # --- Form для ASK (чтобы состояние было стабильным) ---
    with st.form("ask_form", clear_on_submit=False):
        st.text_area(
            "Текст заявки / вопрос инженера",
            placeholder="Например: Клиент не может пройти биометрию, аккаунт заблокирован...",
            height=180,
            key="query_text",
        )
        submitted = st.form_submit_button("Запустить")

    if submitted:
        query_text = (st.session_state.query_text or "").strip()
        if not query_text:
            st.error("Введите текст заявки / вопрос.")
            st.stop()

        api_url = st.session_state.api_url
        service = st.session_state.service
        context_count = int(st.session_state.context_count)

        try:
            ensure_api_ready(api_url)
            with st.spinner("Ищу похожие кейсы и формирую ответ..."):
                resp = call_ask(api_url=api_url, issue_text=query_text, context_count=context_count, service=service)
        except requests.RequestException as exc:
            st.error(f"Ошибка вызова API: {exc}")
            if hasattr(exc, "response") and exc.response is not None:
                st.code(exc.response.text)
            st.stop()

        sections = extract_sections(resp)
        answer_full_text = resp.get("full_text") or format_comment(sections) or "N/A"

        used_issue_keys = resp.get("used_issue_keys") or resp.get("issue_keys") or []
        used_chunks = resp.get("used_chunks") or resp.get("snippets") or resp.get("chunks") or []

        # сохраняем всё
        st.session_state.last_resp = resp
        st.session_state.last_sections = sections
        st.session_state.last_answer_full_text = answer_full_text
        st.session_state.last_used_issue_keys = used_issue_keys
        st.session_state.last_used_chunks = used_chunks

        # сбрасываем сообщения фидбека при новом ответе
        st.session_state.last_feedback_id = None
        st.session_state.last_feedback_msg = ""
        st.session_state.last_feedback_error = ""

    # --- Render сохранённого ответа ---
    if st.session_state.last_resp is None:
        st.info("Сначала нажмите «Запустить», чтобы получить ответ.")
        return

    resp = st.session_state.last_resp
    sections = st.session_state.last_sections or extract_sections(resp)
    answer_full_text = st.session_state.last_answer_full_text or (resp.get("full_text") or format_comment(sections) or "N/A")
    used_issue_keys = st.session_state.last_used_issue_keys or []
    used_chunks = st.session_state.last_used_chunks or []

    st.success("Готово")

    tabs = st.tabs(["Описание", "Причины", "Действия", "Следующие шаги", "Комментарий", "Источники", "RAW JSON"])

    def render_items(items: List[str]):
        if not items:
            st.info("Пусто")
            return
        for i, it in enumerate(items, 1):
            st.markdown(f"{i}. {it}")

    with tabs[0]:
        render_items(sections.get("Описание", []))

    with tabs[1]:
        render_items(sections.get("Причины", []))

    with tabs[2]:
        render_items(sections.get("Действия", []))

    with tabs[3]:
        render_items(sections.get("Следующие шаги", []))

    with tabs[4]:
        st.caption("Сформированный комментарий (можно копировать в Service Desk):")
        st.code(answer_full_text, language="markdown")

        st.download_button(
            "Скачать как .txt",
            data=answer_full_text,
            file_name="rag_comment.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Оценка полезности (feedback)")

        # показываем результат прошлого клика (не пропадает после rerun)
        if st.session_state.last_feedback_msg:
            st.success(st.session_state.last_feedback_msg)
        if st.session_state.last_feedback_error:
            st.error("Ошибка feedback")
            st.code(st.session_state.last_feedback_error)

        feedback_comment = st.text_area(
            "Комментарий к оценке (опционально)",
            placeholder="Например: не хватило конкретики / источники нерелевантны / нужно больше шагов…",
            height=90,
            key="feedback_comment",
        )

        c1, c2 = st.columns(2)

        def do_send_feedback(is_helpful: bool):
            st.session_state.last_feedback_error = ""
            st.session_state.last_feedback_msg = ""

            api_url = st.session_state.api_url
            query_text = (st.session_state.query_text or "").strip()

            # важно: query в feedback должен соответствовать последнему запросу
            if not query_text:
                query_text = query_text or "N/A"

            try:
                out = call_feedback(
                    api_url=api_url,
                    query=query_text,
                    answer_full_text=answer_full_text or "N/A",
                    is_helpful=is_helpful,
                    comment=(feedback_comment.strip() or None),
                    used_issue_keys=used_issue_keys,
                )
            except requests.RequestException as exc:
                if hasattr(exc, "response") and exc.response is not None:
                    st.session_state.last_feedback_error = exc.response.text
                else:
                    st.session_state.last_feedback_error = str(exc)
                return

            st.session_state.last_feedback_id = out.get("id")
            st.session_state.last_feedback_msg = f"Feedback сохранён ✅ (id={out.get('id')})"

        # НЕ отключаем кнопки полностью (иногда нужно пережать), но можно выключить после success:
        disabled_after_success = st.session_state.last_feedback_id is not None

        if c1.button("👍 Полезно", use_container_width=True, disabled=disabled_after_success, key="btn_helpful"):
            do_send_feedback(True)
            st.rerun()

        if c2.button("👎 Не полезно", use_container_width=True, disabled=disabled_after_success, key="btn_not_helpful"):
            do_send_feedback(False)
            st.rerun()

        st.caption("Feedback пишется в таблицу Postgres: public.feedback")

    with tabs[5]:
        st.markdown("### Used issue keys")
        st.write(used_issue_keys)

        st.markdown("### Used chunks / snippets")
        st.write(used_chunks)

    with tabs[6]:
        st.json(resp)


if __name__ == "__main__":
    main()
