FROM python:3.12-slim

ARG WITH_VIT=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt requirements-vit.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_VIT" = "true" ]; then pip install --no-cache-dir -r requirements-vit.txt; fi

COPY app ./app
COPY scripts ./scripts
COPY pyproject.toml ./

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
