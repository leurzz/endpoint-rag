# Documents

Coloca aquí los ficheros `.jsonl` a indexar. Cada línea debe ser un JSON con al menos:

```json
{"text": "Contenido del párrafo", "language": "va", "title": "opcional"}
```

El dominio se deduce del nombre del archivo (por ejemplo `news.jsonl`, `bulletins.jsonl`, `parliament.jsonl`). Los textos se trocean con solapamiento (`chunk_size` y `chunk_overlap` en ajustes) y se generan embeddings ligeros para recuperar contexto en `/predict`.
