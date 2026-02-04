FROM python:3.11-slim

WORKDIR /app

# зависимости UI
COPY ui/requirements_ui.txt /app/requirements_ui.txt
RUN pip install --no-cache-dir -r /app/requirements_ui.txt

# сам UI
COPY ui/streamlit_app.py /app/streamlit_app.py

EXPOSE 8501

# Streamlit должен слушать 0.0.0.0 внутри контейнера
CMD ["streamlit", "run", "/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
