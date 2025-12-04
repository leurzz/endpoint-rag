# MiniRAG FastAPI

Backend ligero para generación aumentada por recuperación (RAG) con funcionalidades básicas. Permite la indexación y consulta eficiente de documentos.

## Requisitos

- Python 3.11+
- Dependencias: `pip install -r requirements.txt`

## Ejecutar local

```bash
uvicorn app.main:app --reload --port 8000
```

## Docker Compose

```bash
docker compose up --build
```

## Configuración principal (`app/settings.py`)

- `languages`: lista de idiomas aceptados (por defecto `["va"]`).
- `domains`: dominios disponibles (`parliament`, `news`, `bulletins`).
- `prompt_*`: prompt base por dominio.
- `chunk_size` / `chunk_overlap`: control de troceo de textos.
- `top_k`: número de fragmentos devueltos en el contexto.
- `llm_server` / `llm_api_key`: URL y API key si se quiere llamar a un servidor LLM tipo OpenAI compatible.

Los ficheros `.jsonl` deben tener al menos `text` y `language`. El dominio se infiere del nombre del archivo.
