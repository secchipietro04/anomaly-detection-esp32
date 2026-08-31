# Stage 1: CDDL Schema Builder (cached by Docker, runs only if cbor_schema/ changes)
FROM node:20-alpine AS schema-builder
WORKDIR /build

RUN npm install -g cddl2py@0.3.1
RUN apk add --no-cache python3

COPY cbor_schema/ ./cbor_schema/
COPY backend/scripts/generate_schemas.py ./generate_schemas.py

RUN python3 ./generate_schemas.py --cddl-dir ./cbor_schema --output /build/schemas.py

# Stage 2: Runtime Python Image
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY --from=schema-builder /build/schemas.py ./app/cbor/schemas.py

ENV PYTHONPATH=/app
CMD ["python", "app/worker_mqtt.py"]
