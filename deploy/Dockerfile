# Gateway image. CPU torch is enough for encoder students; LLM students can be served
# by the vllm service instead (serving.backend: vllm).
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[gateway,train,security,postgres]"
ENV DISTILLER_CONFIG=/config/distiller.yaml
EXPOSE 8000
CMD ["sh", "-c", "distiller gateway -c $DISTILLER_CONFIG --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2}"]
