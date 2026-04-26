FROM python:3.10-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY model.pth .
RUN mkdir -p plant-disease-classifier && \
    cp src/models/model-settings.json plant-disease-classifier/model-settings.json && \
    rm src/models/model-settings.json

ENV MLSERVER_PARALLEL_WORKERS=0

EXPOSE 8080

CMD ["mlserver", "start", "."]