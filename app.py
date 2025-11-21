
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import pipeline
import sys
from pathlib import Path

# Agregar scripts al path para importar detectar_bots
sys.path.append(str(Path(__file__).parent / "scripts"))
from detectar_bots import detect_bots

st.set_page_config(page_title="News Sentiment Analyzer", page_icon="📰", layout="wide")

st.title("📰 News Sentiment Analyzer (ES)")
st.caption("Pegue comentarios de una noticia o cargue un CSV para analizar sentimiento con un modelo preentrenado y visualizar resultados con Plotly.")

@st.cache_resource(show_spinner=False)
def load_model():
    # Modelo multilingüe de 1-5 estrellas; lo convertimos a Pos/Neu/Neg
    clf = pipeline("sentiment-analysis", model="dccuchile/tulio-chilean-spanish-bert")
    return clf

def analyze_sentiment(clf, texts):
    """
    Analiza el sentimiento de una lista de textos, truncando automáticamente
    los que excedan 512 tokens (límite de BERT).
    """
    return clf(texts, truncation=True, max_length=512)

def map_label_to_polarity(label: str):
    """
    Mapea etiquetas del modelo a polaridad (Negativo/Neutro/Positivo).
    Soporta dos formatos:
    - '1 star', '2 stars', etc. (modelos de clasificación de estrellas)
    - 'LABEL_0', 'LABEL_1', etc. (modelos genéricos, donde LABEL_N = N+1 estrellas)
    """
    # Intentar extraer el número de estrellas
    stars = None
    
    # Formato 1: '1 star', '2 stars', etc.
    if 'star' in label.lower():
        try:
            stars = int(label.split()[0])
        except (ValueError, IndexError):
            pass
    
    # Formato 2: 'LABEL_0', 'LABEL_1', etc.
    elif label.startswith('LABEL_'):
        try:
            label_num = int(label.split('_')[1])
            # Asumir que LABEL_0 = 1 estrella, LABEL_1 = 2 estrellas, etc.
            # O si son 5 clases: LABEL_0=1, LABEL_1=2, LABEL_2=3, LABEL_3=4, LABEL_4=5
            stars = label_num + 1
        except (ValueError, IndexError):
            pass
    
    # Si no se pudo extraer, intentar convertir directamente
    if stars is None:
        try:
            stars = int(label)
        except ValueError:
            # Si falla todo, asumir neutro
            return "Neutro", 0.5
    
    # Mapear estrellas a polaridad
    if stars <= 2:
        return "Negativo", (stars - 1) / 1.0  # 0..1 dentro de negativo
    elif stars == 3:
        return "Neutro", 0.5
    else:
        return "Positivo", (stars - 3) / 2.0  # 0..1 dentro de positivo


def validate_dataset_columns(df: pd.DataFrame):
    # Requeridos mínimos: 'text' (o 'comment') y 'user_id' (o 'Name') y 'timestamp' (o 'Date')
    cols = set([c.lower() for c in df.columns])
    has_text = ('text' in cols) or ('comment' in cols)
    has_user = ('user_id' in cols) or ('name' in cols)
    has_time = ('timestamp' in cols) or ('date' in cols)
    return has_text and has_user and has_time


def run_pipeline_if_needed(df: pd.DataFrame) -> Path:
    """Ejecuta el pipeline para generar features de usuarios si no existen."""
    import subprocess
    from pathlib import Path
    import tempfile
    import os
    
    # Verificar que tenemos las columnas necesarias
    if 'user_id' not in df.columns:
        st.error("Se requiere la columna 'user_id' para ejecutar el pipeline de detección de bots.")
        return None
    
    if 'text' not in df.columns and 'comment' not in df.columns:
        st.error("Se requiere la columna 'text' o 'comment' para ejecutar el pipeline.")
        return None
    
    user_features_path = Path("data/user_features.csv")
    
    # Guardar datos temporalmente para el pipeline
    temp_path = Path("data/temp_for_pipeline.csv")
    # Asegurar que las columnas estén normalizadas
    df_pipeline = df.copy()
    if 'comment' in df_pipeline.columns and 'text' not in df_pipeline.columns:
        df_pipeline = df_pipeline.rename(columns={'comment': 'text'})
    if 'Name' in df_pipeline.columns and 'user_id' not in df_pipeline.columns:
        df_pipeline = df_pipeline.rename(columns={'Name': 'user_id'})
    if 'Date' in df_pipeline.columns and 'timestamp' not in df_pipeline.columns:
        df_pipeline = df_pipeline.rename(columns={'Date': 'timestamp'})
    
    df_pipeline.to_csv(temp_path, index=False)
    
    # Ejecutar pipeline
    try:
        result = subprocess.run([
            sys.executable, "scripts/pipeline.py",
            "--input", str(temp_path),
            "--output", str(user_features_path)
        ], check=True, capture_output=True, text=True, timeout=300)
        
        if user_features_path.exists():
            return user_features_path
        else:
            st.error(f"El pipeline no generó el archivo esperado. Error: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        st.error("El pipeline tardó demasiado tiempo. Intente con un dataset más pequeño.")
        return None
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
        st.error(f"Error ejecutando pipeline: {error_msg}")
        return None
    except Exception as e:
        st.error(f"Error inesperado ejecutando pipeline: {e}")
        return None

st.sidebar.header("Entrada de datos")
mode = st.sidebar.radio("Seleccione modo de entrada", ["Pegar texto", "Cargar CSV", "Demo"], index=2)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Detección de Bots")
enable_bot_detection = st.sidebar.checkbox("Habilitar detección de bots", value=False)
bot_threshold = st.sidebar.slider("Umbral de detección", 0.0, 1.0, 0.5, 0.05, 
                                   help="Score mínimo para clasificar como bot (0-1)")

if mode == "Pegar texto":
    st.subheader("Pegar comentarios (uno por línea)")
    text = st.text_area("Comentarios", height=200, placeholder="Escribe o pega comentarios aquí, uno por línea...")
    if st.button("Analizar"):
        comments = [t.strip() for t in text.splitlines() if t.strip()]
        if comments:
            clf = load_model()
            preds = analyze_sentiment(clf, comments)
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
            st.error("El CSV debe contener al menos las columnas 'user_id' (o 'Name'), 'timestamp' (o 'Date') y 'text' (o 'comment').")
        else:
            if st.button("Analizar CSV"):
                # Normalizar columnas: Comment -> text, Name -> user_id, Date -> timestamp
                if 'Comment' in data.columns and 'text' not in data.columns:
                    data = data.rename(columns={'Comment': 'text'})
                elif 'comment' in data.columns and 'text' not in data.columns:
                    data = data.rename(columns={'comment': 'text'})
                
                if 'Name' in data.columns and 'user_id' not in data.columns:
                    data = data.rename(columns={'Name': 'user_id'})
                
                if 'Date' in data.columns and 'timestamp' not in data.columns:
                    data = data.rename(columns={'Date': 'timestamp'})
                
                if 'Likes' in data.columns and 'likes' not in data.columns:
                    data = data.rename(columns={'Likes': 'likes'})
                
                texts = data['text'].astype(str).tolist()
                clf = load_model()
                preds = analyze_sentiment(clf, texts)
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
        # Intentar leer comentarios_limpios.csv primero, luego comments_clean.csv
        demo_df = None
        for filename in ["data/comentarios_limpios.csv", "data/comments_clean.csv"]:
            try:
                demo_df = pd.read_csv(filename)
                break
            except FileNotFoundError:
                continue
        if demo_df is None:
            raise FileNotFoundError("No se encontró archivo de demo")
    except Exception as e:
        st.error(f"No se pudo leer demo CSV: {e}")
        demo_df = None

    if demo_df is not None and st.button("Analizar demo"):
        # Normalizar columnas
        if 'Comment' in demo_df.columns and 'text' not in demo_df.columns:
            demo_df = demo_df.rename(columns={'Comment': 'text'})
        elif 'comment' in demo_df.columns and 'text' not in demo_df.columns:
            demo_df = demo_df.rename(columns={'comment': 'text'})
        
        if 'Name' in demo_df.columns and 'user_id' not in demo_df.columns:
            demo_df = demo_df.rename(columns={'Name': 'user_id'})
        
        if 'Date' in demo_df.columns and 'timestamp' not in demo_df.columns:
            demo_df = demo_df.rename(columns={'Date': 'timestamp'})
        
        if 'Likes' in demo_df.columns and 'likes' not in demo_df.columns:
            demo_df = demo_df.rename(columns={'Likes': 'likes'})
        
        texts = demo_df['text'].astype(str).tolist()
        clf = load_model()
        preds = analyze_sentiment(clf, texts)
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
    
    # Detección de bots si está habilitada y tenemos user_id
    bot_results_df = None
    if enable_bot_detection and 'user_id' in df.columns:
        st.markdown("---")
        st.subheader("🤖 Detección de Bots")
        
        with st.spinner("Generando features de usuarios y detectando bots..."):
            try:
                # Ejecutar pipeline si es necesario
                user_features_path = run_pipeline_if_needed(df)
                
                if user_features_path and user_features_path.exists():
                    # Guardar datos temporalmente para detección
                    temp_comments_path = Path("data/temp_comments_for_detection.csv")
                    df.to_csv(temp_comments_path, index=False)
                    
                    # Ejecutar detección
                    bot_results_df = detect_bots(
                        user_features_path=str(user_features_path),
                        comments_path=str(temp_comments_path),
                        threshold=bot_threshold,
                        output_path=None  # No guardar, solo retornar
                    )
                    
                    # Guardar en session state
                    st.session_state["bot_results"] = bot_results_df
                    
                    # Agregar información de bot a cada comentario en el dataframe principal
                    if 'user_id' in df.columns:
                        # Crear un diccionario de mapeo user_id -> (is_bot, bot_probability)
                        bot_info_map = bot_results_df.set_index('user_id')[['is_bot', 'bot_probability']].to_dict('index')
                        
                        # Agregar columnas al dataframe principal
                        df['is_bot'] = df['user_id'].map(lambda x: bot_info_map.get(x, {}).get('is_bot', 0))
                        df['bot_probability'] = df['user_id'].map(lambda x: bot_info_map.get(x, {}).get('bot_probability', 0.0))
                        # Asegurar que los valores sean del tipo correcto
                        if 'is_bot' in df.columns:
                            df['is_bot'] = df['is_bot'].fillna(0).astype(int)
                        if 'bot_probability' in df.columns:
                            df['bot_probability'] = df['bot_probability'].fillna(0.0)
                        
                        # Actualizar session state con el dataframe actualizado
                        st.session_state["df"] = df
                    
                    # Mostrar KPIs de bots
                    n_bots = bot_results_df['is_bot'].sum()
                    n_humans = len(bot_results_df) - n_bots
                    pct_bots = (n_bots / len(bot_results_df)) * 100 if len(bot_results_df) > 0 else 0
                    
                    bot_col1, bot_col2, bot_col3 = st.columns(3)
                    bot_col1.metric("🤖 Bots detectados", n_bots, f"{pct_bots:.1f}%")
                    bot_col2.metric("👤 Humanos", n_humans, f"{100-pct_bots:.1f}%")
                    bot_col3.metric("📊 Score promedio", f"{bot_results_df['bot_probability'].mean():.3f}")
                    
                    # Visualizaciones avanzadas de bots
                    st.markdown("#### 📈 Análisis Detallado de Bots")
                    
                    # Histograma de scores de bot
                    fig_bot_hist = px.histogram(
                        bot_results_df, 
                        x="bot_probability", 
                        nbins=20,
                        title="Distribución de Scores de Bot",
                        labels={"bot_probability": "Score de Bot", "count": "Número de Usuarios"},
                        color_discrete_sequence=['#FF6B6B']
                    )
                    fig_bot_hist.add_vline(x=bot_threshold, line_dash="dash", line_color="red", 
                                          annotation_text=f"Umbral ({bot_threshold})")
                    st.plotly_chart(fig_bot_hist, use_container_width=True)
                    
                    # Scatter plots de características
                    col_scatter1, col_scatter2 = st.columns(2)
                    
                    with col_scatter1:
                        # Posts por día vs Ratio de duplicados
                        if 'posts_per_day' in bot_results_df.columns and 'dup_post_ratio' in bot_results_df.columns:
                            fig_scatter1 = px.scatter(
                                bot_results_df,
                                x="posts_per_day",
                                y="dup_post_ratio",
                                color="is_bot",
                                size="n_posts",
                                hover_data=["user_id", "bot_probability"],
                                title="Frecuencia vs Repetición",
                                labels={
                                    "posts_per_day": "Posts por Día",
                                    "dup_post_ratio": "Ratio de Posts Duplicados",
                                    "is_bot": "Es Bot"
                                },
                                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
                            )
                            st.plotly_chart(fig_scatter1, use_container_width=True)
                    
                    with col_scatter2:
                        # TTR vs Variación Semántica
                        if 'ttr' in bot_results_df.columns and 'embedding_std_norm' in bot_results_df.columns:
                            fig_scatter2 = px.scatter(
                                bot_results_df,
                                x="ttr",
                                y="embedding_std_norm",
                                color="is_bot",
                                size="n_posts",
                                hover_data=["user_id", "bot_probability"],
                                title="Diversidad Léxica vs Variación Semántica",
                                labels={
                                    "ttr": "Type-Token Ratio (Diversidad)",
                                    "embedding_std_norm": "Variación Semántica",
                                    "is_bot": "Es Bot"
                                },
                                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'}
                            )
                            st.plotly_chart(fig_scatter2, use_container_width=True)
                    
                    # Comparación de características: Bots vs Humanos
                    st.markdown("#### 🔍 Comparación: Bots vs Humanos")
                    comparison_cols = ['posts_per_day', 'dup_post_ratio', 'ttr', 'embedding_std_norm', 
                                      'avg_likes', 'n_posts']
                    available_cols = [c for c in comparison_cols if c in bot_results_df.columns]
                    
                    if available_cols:
                        comparison_data = []
                        for col in available_cols:
                            bots_mean = bot_results_df[bot_results_df['is_bot'] == 1][col].mean()
                            humans_mean = bot_results_df[bot_results_df['is_bot'] == 0][col].mean()
                            comparison_data.append({
                                'Característica': col,
                                'Bots': bots_mean,
                                'Humanos': humans_mean
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        comparison_df_melted = comparison_df.melt(
                            id_vars=['Característica'],
                            value_vars=['Bots', 'Humanos'],
                            var_name='Tipo',
                            value_name='Valor Promedio'
                        )
                        
                        fig_comparison = px.bar(
                            comparison_df_melted,
                            x='Característica',
                            y='Valor Promedio',
                            color='Tipo',
                            barmode='group',
                            title="Comparación de Características Promedio",
                            color_discrete_map={'Bots': '#FF6B6B', 'Humanos': '#4ECDC4'}
                        )
                        fig_comparison.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Tabla de usuarios sospechosos (top bots)
                    st.markdown("#### 🎯 Usuarios Más Sospechosos (Top Bots)")
                    top_bots = bot_results_df.nlargest(20, 'bot_probability')[
                        ['user_id', 'bot_probability', 'n_posts', 'posts_per_day', 
                         'dup_post_ratio', 'ttr', 'is_bot']
                    ].copy()
                    top_bots = top_bots.round(3)
                    st.dataframe(top_bots, use_container_width=True, hide_index=True)
                    
                    # Descargar resultados de detección de bots
                    st.download_button(
                        "📥 Descargar Resultados de Detección de Bots",
                        data=bot_results_df.to_csv(index=False).encode("utf-8"),
                        file_name="bot_detection_results.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("No se pudieron generar las features de usuarios. Verifique que el pipeline funcione correctamente.")
            except Exception as e:
                st.error(f"Error en detección de bots: {e}")
                import traceback
                with st.expander("Detalles del error"):
                    st.code(traceback.format_exc())
    
    # Mostrar sección de bots detectados si está disponible
    if 'is_bot' in df.columns and df['is_bot'].sum() > 0:
        st.markdown("---")
        st.subheader("🤖 Comentarios de Bots Detectados")
        
        bot_comments_df = df[df['is_bot'] == 1].copy()
        n_bots_detected = len(bot_comments_df)
        n_unique_bots = bot_comments_df['user_id'].nunique() if 'user_id' in bot_comments_df.columns else 0
        
        col_bot1, col_bot2, col_bot3 = st.columns(3)
        col_bot1.metric("🤖 Comentarios de Bots", n_bots_detected)
        col_bot2.metric("👤 Usuarios Bot Únicos", n_unique_bots)
        if 'bot_probability' in bot_comments_df.columns:
            col_bot3.metric("📊 Probabilidad Promedio", f"{bot_comments_df['bot_probability'].mean():.3f}")
        
        # Mostrar tabla de comentarios de bots
        with st.expander(f"Ver {n_bots_detected} comentarios de bots", expanded=False):
            bot_display_cols = ['user_id', 'text', 'bot_probability', 'polarity', 'timestamp']
            if 'comment' in bot_comments_df.columns:
                bot_display_cols.insert(1, 'comment')
            available_cols = [c for c in bot_display_cols if c in bot_comments_df.columns]
            st.dataframe(bot_comments_df[available_cols], use_container_width=True, hide_index=True)
    
    # KPIs de sentimiento
    st.markdown("---")
    st.subheader("📊 Análisis de Sentimiento")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Positivos", int((df["polarity"]=="Positivo").sum()))
    col2.metric("Neutros", int((df["polarity"]=="Neutro").sum()))
    col3.metric("Negativos", int((df["polarity"]=="Negativo").sum()))
    col4.metric("Promedio score raw", round(float(df["score_raw"].mean()), 3))
    
    # Filtrar bots si está habilitado
    df_for_sentiment = df.copy()
    if enable_bot_detection and bot_results_df is not None and 'user_id' in df.columns:
        # Obtener lista de bots
        bot_users = bot_results_df[bot_results_df['is_bot'] == 1]['user_id'].tolist()
        n_bots_filtered = df_for_sentiment[df_for_sentiment['user_id'].isin(bot_users)].shape[0]
        df_for_sentiment = df_for_sentiment[~df_for_sentiment['user_id'].isin(bot_users)]
        
        if len(bot_users) > 0:
            st.info(f"⚠️ Se filtraron {n_bots_filtered} comentarios de {len(bot_users)} bot(s) del análisis de sentimiento.")

    with st.expander("Ver tabla de resultados"):
        # Si tenemos información de bots, mostrar con colores
        display_df = df.copy()
        
        # Agregar columna de estado de bot si existe
        if 'is_bot' in display_df.columns:
            # Crear una columna formateada para mostrar
            display_df['Estado'] = display_df['is_bot'].apply(lambda x: '🤖 Bot' if x == 1 else '👤 Humano')
            # Reordenar columnas para que Estado y bot_probability estén visibles
            cols = ['Estado', 'bot_probability'] if 'bot_probability' in display_df.columns else ['Estado']
            other_cols = [c for c in display_df.columns if c not in cols]
            display_df = display_df[cols + other_cols]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Mostrar resumen de bots si está disponible
        if 'is_bot' in df.columns:
            n_bots_comments = df['is_bot'].sum()
            n_humans_comments = len(df) - n_bots_comments
            st.caption(f"📊 Resumen: {n_bots_comments} comentarios de bots, {n_humans_comments} comentarios de humanos")

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

    # Gráficos Plotly de sentimiento (usando datos filtrados)
    counts = df_for_sentiment["polarity"].value_counts().reset_index()
    counts.columns = ["polarity", "count"]
    fig_bar = px.bar(counts, x="polarity", y="count", title="Distribución de polaridad (bots excluidos)", text="count")
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(counts, names="polarity", values="count", title="Proporción por polaridad (bots excluidos)", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_hist = px.histogram(df_for_sentiment, x="score_raw", nbins=10, title="Histograma de confianza del modelo")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Nube simple por longitud (proxy de engagement)
    if 'comment' in df_for_sentiment.columns:
        df_for_sentiment["len"] = df_for_sentiment["comment"].str.len()
        fig_len = px.box(df_for_sentiment, x="polarity", y="len", points="all", title="Longitud de comentario por polaridad")
        st.plotly_chart(fig_len, use_container_width=True)

    # Descargar resultados (incluir etiquetas de bot si están disponibles)
    df_export = df.copy()
    if bot_results_df is not None and 'user_id' in df.columns:
        # Agregar información de bots al dataframe de exportación
        bot_info = bot_results_df[['user_id', 'bot_probability', 'is_bot']].copy()
        df_export = df_export.merge(bot_info, on='user_id', how='left')
        # Verificar que las columnas existan después del merge antes de procesarlas
        if 'is_bot' in df_export.columns:
            df_export['is_bot'] = df_export['is_bot'].fillna(0).astype(int)
        if 'bot_probability' in df_export.columns:
            df_export['bot_probability'] = df_export['bot_probability'].fillna(0.0)
    
    st.download_button("Descargar CSV de resultados", data=df_export.to_csv(index=False).encode("utf-8"), file_name="sentiment_results.csv", mime="text/csv")

st.markdown("---")
st.caption("Nota: El modelo produce etiquetas de 1 a 5 estrellas. Se mapean a Negativo (1-2), Neutro (3), Positivo (4-5).")
