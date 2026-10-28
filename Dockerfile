FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MAINTAI_HOME=/app/runtime
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 maintai && mkdir -p /app/runtime && chown maintai:maintai /app/runtime
COPY --chown=maintai:maintai . .
USER maintai
CMD ["python", "-m", "uvicorn", "maintai.api:app", "--host", "0.0.0.0", "--port", "8000"]
