# 📰 News Sentiment Analyzer (ES)

Aplicación **Streamlit** para analizar el sentimiento de comentarios en noticias.
Usa `transformers` (modelo multilingüe de estrellas) y visualiza resultados con **Plotly**.

## 🚀 Cómo ejecutar

```bash
cd news_sentiment_app
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 📥 Modos de entrada
- **Pegar texto:** comentarios uno por línea.
- **Cargar CSV:** archivo con columna obligatoria `comment` y opcional `author`.
- **Demo:** usa `data/sample_comments.csv` ya incluido.

## 🧠 Modelo
- `nlptown/bert-base-multilingual-uncased-sentiment` (1-5 estrellas).
- Mapeo a polaridad: 1-2 ⇒ Negativo; 3 ⇒ Neutro; 4-5 ⇒ Positivo.

## 📊 Gráficos
- Barras: distribución de polaridad
- Pie: proporción por polaridad
- Histograma: confianza del modelo
- Boxplot: longitud del comentario por polaridad

## ⚠️ Notas
- La primera ejecución descargará pesos del modelo (requiere internet).
- Si deseas un modelo específico para **español**, puedes cambiar a:
  - `pysentimiento/robertuito-sentiment-analysis` (labels: POS, NEG, NEU).
- Para producción, considera moderación de contenido y anonimización.