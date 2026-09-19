FROM python:3.13-slim-bookworm

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000
ENV INFER_MODEL=Qwen/Qwen2.5-0.5B-Instruct

ENTRYPOINT ["infer"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
