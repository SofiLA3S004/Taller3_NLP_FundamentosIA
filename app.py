
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import pipeline

st.set_page_config(page_title="News Sentiment Analyzer", page_icon="📰", layout="wide")

st.title("📰 News Sentiment Analyzer (ES)")
st.caption("Pegue comentarios de una noticia o cargue un CSV para analizar sentimiento con un modelo preentrenado y visualizar resultados con Plotly.")

@st.cache_resource(show_spinner=False)
def load_model():
    # Modelo multilingüe de 1-5 estrellas; lo convertimos a Pos/Neu/Neg
    clf = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return clf

def map_label_to_polarity(label: str):
    # Mapea '1 star'..'5 stars' a Negativo / Neutro / Positivo + score normalizado.
    stars = int(label.split()[0])
    if stars <= 2:
        return "Negativo", (stars - 1) / 1.0  # 0..1 dentro de negativo
    elif stars == 3:
        return "Neutro", 0.5
    else:
        return "Positivo", (stars - 3) / 2.0  # 0..1 dentro de positivo


def validate_dataset_columns(df: pd.DataFrame):
    # Requeridos mínimos: 'text' (o 'comment') y 'user_id' y 'timestamp'
    cols = set([c.lower() for c in df.columns])
    has_text = ('text' in cols) or ('comment' in cols)
    has_user = 'user_id' in cols
    has_time = 'timestamp' in cols
    return has_text and has_user and has_time

st.sidebar.header("Entrada de datos")
mode = st.sidebar.radio("Seleccione modo de entrada", ["Pegar texto", "Cargar CSV", "Demo"], index=2)

if mode == "Pegar texto":
    st.subheader("Pegar comentarios (uno por línea)")
    text = st.text_area("Comentarios", height=200, placeholder="Escribe o pega comentarios aquí, uno por línea...")
    if st.button("Analizar"):
        comments = [t.strip() for t in text.splitlines() if t.strip()]
        if comments:
            clf = load_model()
            preds = clf(comments)
            rows = []
            for c, p in zip(comments, preds):
                pol, strength = map_label_to_polarity(p["label"])
                rows.append({"comment": c, "label_raw": p["label"], "score_raw": p["score"], "polarity": pol, "strength": strength})
            df = pd.DataFrame(rows)
            st.session_state["df"] = df
        else:
            st.warning("Ingrese al menos un comentario.")

elif mode == "Cargar CSV":
    st.subheader("Cargar CSV con columnas 'user_id','text' (o 'comment'), 'timestamp' y métricas opcionales")
    file = st.file_uploader("Seleccionar CSV", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        if not validate_dataset_columns(data):
            st.error("El CSV debe contener al menos las columnas 'user_id', 'timestamp' y 'text' (o 'comment').")
        else:
            if st.button("Analizar CSV"):
                # Normalizar columna 'comment' -> 'text'
                if 'comment' in data.columns and 'text' not in data.columns:
                    data = data.rename(columns={'comment': 'text'})
                texts = data['text'].astype(str).tolist()
                clf = load_model()
                preds = clf(texts)
                rows = []
                for idx, (c, p) in enumerate(zip(texts, preds)):
                    pol, strength = map_label_to_polarity(p["label"])
                    row = {
                        "user_id": data.get('user_id', [None]*len(data))[idx],
                        "text": c,
                        "timestamp": data.get('timestamp', [None]*len(data))[idx],
                        "label_raw": p["label"],
                        "score_raw": p["score"],
                        "polarity": pol,
                        "strength": strength
                    }
                    for m in ('likes','replies','shares'):
                        if m in data.columns:
                            row[m] = data[m].iloc[idx]
                    rows.append(row)
                df = pd.DataFrame(rows)
                st.session_state["df"] = df

else:
    st.subheader("Demo con datos de ejemplo")
    try:
        demo_df = pd.read_csv("data/comments_clean.csv")
    except Exception as e:
        st.error(f"No se pudo leer demo CSV: {e}")
        demo_df = None

    if demo_df is not None and st.button("Analizar demo"):
        if 'comment' in demo_df.columns and 'text' not in demo_df.columns:
            demo_df = demo_df.rename(columns={'comment': 'text'})
        texts = demo_df['text'].astype(str).tolist()
        clf = load_model()
        preds = clf(texts)
        rows = []
        for idx, (c, p) in enumerate(zip(texts, preds)):
            pol, strength = map_label_to_polarity(p["label"])
            row = {
                "user_id": demo_df.get('user_id', [None]*len(demo_df))[idx],
                "text": c,
                "timestamp": demo_df.get('timestamp', [None]*len(demo_df))[idx],
                "label_raw": p["label"],
                "score_raw": p["score"],
                "polarity": pol,
                "strength": strength
            }
            for m in ('likes','replies','shares'):
                if m in demo_df.columns:
                    row[m] = demo_df[m].iloc[idx]
            rows.append(row)
        st.session_state["df"] = pd.DataFrame(rows)

df = st.session_state.get("df")
if df is not None and not df.empty:
    st.success(f"Se analizaron {len(df)} comentarios.")
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Positivos", int((df["polarity"]=="Positivo").sum()))
    col2.metric("Neutros", int((df["polarity"]=="Neutro").sum()))
    col3.metric("Negativos", int((df["polarity"]=="Negativo").sum()))
    col4.metric("Promedio score raw", round(float(df["score_raw"].mean()), 3))

    with st.expander("Ver tabla de resultados"):
        st.dataframe(df, use_container_width=True)

    # Normalizar: si tenemos 'text' pero no 'comment', crear 'comment' para compatibilidad con visualizaciones previas
    if 'text' in df.columns and 'comment' not in df.columns:
        df['comment'] = df['text']

    # Métricas de engagement si están disponibles
    metric_cols = [c for c in ('likes','replies','shares') if c in df.columns]
    if metric_cols:
        mcol1, mcol2, mcol3 = st.columns(3)
        if 'likes' in df.columns:
            mcol1.metric("Total likes", int(df['likes'].fillna(0).sum()))
            mcol1.metric("Likes promedio", round(float(df['likes'].fillna(0).mean()),1))
        if 'replies' in df.columns:
            mcol2.metric("Total replies", int(df['replies'].fillna(0).sum()))
            mcol2.metric("Replies promedio", round(float(df['replies'].fillna(0).mean()),1))
        if 'shares' in df.columns:
            mcol3.metric("Total shares", int(df['shares'].fillna(0).sum()))
            mcol3.metric("Shares promedio", round(float(df['shares'].fillna(0).mean()),1))

    # Gráficos Plotly
    counts = df["polarity"].value_counts().reset_index()
    counts.columns = ["polarity", "count"]
    fig_bar = px.bar(counts, x="polarity", y="count", title="Distribución de polaridad", text="count")
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(counts, names="polarity", values="count", title="Proporción por polaridad", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_hist = px.histogram(df, x="score_raw", nbins=10, title="Histograma de confianza del modelo")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Nube simple por longitud (proxy de engagement)
    df["len"] = df["comment"].str.len()
    fig_len = px.box(df, x="polarity", y="len", points="all", title="Longitud de comentario por polaridad")
    st.plotly_chart(fig_len, use_container_width=True)

    # Descargar resultados
    st.download_button("Descargar CSV de resultados", data=df.to_csv(index=False).encode("utf-8"), file_name="sentiment_results.csv", mime="text/csv")

st.markdown("---")
st.caption("Nota: El modelo produce etiquetas de 1 a 5 estrellas. Se mapean a Negativo (1-2), Neutro (3), Positivo (4-5).")
