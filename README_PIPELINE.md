**Pipeline: Extracción de features por usuario**

- **Input**: Uno o más archivos CSV (columnas mínimas: `user_id`, `text` o `comment`, `timestamp`; opcionales: `likes`, `replies`, `shares`). Los múltiples archivos se combinan en un solo dataset.
- **Output**: Un único archivo CSV `data/user_features.csv` (cada fila = un `user_id` con features agregados)

Pasos del pipeline:
- Preprocesamiento: limpieza básica (URLs, mentions, exceso de espacios), normalización y tokenización.
- Ingeniería de rasgos:
  - Comportamiento: `n_posts`, `posts_per_day`, `total_likes`, `avg_likes`, `total_replies`, `avg_replies`, `total_shares`, `avg_shares`.
  - Léxicos: `avg_char_len`, `avg_tokens`, `ttr` (type-token ratio), `dup_post_ratio`, `trigram_repeat_ratio`.
  - Semánticos: embeddings por post (modelo `all-MiniLM-L6-v2`) agregados por usuario (`emb_mean_0..N`, `embedding_std_norm`).

Cómo ejecutar:

1. Instalar dependencias (recomendado en un virtualenv):

```powershell
pip install -r requirements.txt
```

2. Ejecutar el pipeline:

Con un solo archivo de entrada:
```powershell
python scripts/pipeline.py --input data/comments_clean.csv --output data/user_features.csv
```

Con múltiples archivos de entrada (se combinarán en un solo dataset):
```powershell
python scripts/pipeline.py --input data/export_20251120-175026.csv data/export_20251120-235652.csv data/export_20251121-000802.csv --output data/user_features.csv
```

Notas:
- El script descargará recursos NLTK si no están presentes.
- El modelo de embeddings se descargará automáticamente la primera vez.
- Guardará un CSV con todas las columnas de features; las columnas de embedding son muchas (dimensión del modelo, normalmente 384 para MiniLM).

Si quieres, puedo ejecutar el pipeline localmente aquí y generar `data/user_features.csv` para que lo revises. Si prefieres, puedo adaptar el script para producir solo resumenes más compactos (por ejemplo, PCA sobre embeddings) o para exportar un dataset listo para modelado.