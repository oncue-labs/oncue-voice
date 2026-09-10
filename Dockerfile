FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.4.3

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-interaction --no-ansi

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "oncue_voice.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
