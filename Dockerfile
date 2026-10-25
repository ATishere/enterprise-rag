FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu \
    && python -m pip install .

EXPOSE 8000
CMD ["uvicorn", "enterprise_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
