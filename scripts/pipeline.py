#!/usr/bin/env python3
"""
Pipeline de extracción de features por usuario.

Funciones:
- Preprocesamiento: limpieza básica, normalización y agrupamiento por usuario.
- Ingeniería de rasgos:
  - Comportamiento: número de publicaciones, frecuencia (posts/día), total y promedio de interacciones.
  - Léxicos: longitud promedio, tokens promedio, Type-Token Ratio (TTR), repetición de frases (duplicados y n-grams repetidos).
  - Semánticos: embeddings por publicación con Sentence-BERT (all-MiniLM-L6-v2) y agregación por usuario (media, norma std).

Salida: Un único archivo CSV con características agregadas por `user_id` combinando todos los inputs.

Uso:
    python scripts/pipeline.py --input data/comments_clean.csv --output data/user_features.csv
    python scripts/pipeline.py --input data/file1.csv data/file2.csv data/file3.csv --output data/user_features.csv

Requiere: sentence-transformers, nltk, pandas, scikit-learn, tqdm
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# NLP
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
from collections import Counter


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")


def basic_clean(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text)
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    # normalizar comillas y guiones
    t = t.replace('\"', '"').replace("\r", " ").replace("\n", " ")
    # quitar exceso de espacios
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokens(text: str):
    # requiere nltk punkt/tokenizers descargado
    return [w.lower() for w in word_tokenize(text) if any(c.isalnum() for c in w)]


def compute_lexical_features(texts):
    # texts: list of strings for a single user
    toks = [tokens(t) for t in texts]
    total_tokens = sum(len(t) for t in toks)
    unique_tokens = len(set([w for t in toks for w in t]))
    ttr = (unique_tokens / total_tokens) if total_tokens > 0 else 0.0
    avg_tokens = (total_tokens / len(texts)) if texts else 0.0
    avg_chars = np.mean([len(t) for t in texts]) if texts else 0.0
    # duplicated posts ratio
    dup_ratio = 1.0 - (len(set(texts)) / len(texts)) if texts else 0.0
    # repeated 3-grams ratio: proportion of posts that share at least one trigram in common with another post
    def trigrams_of(text):
        w = tokens(text)
        return [tuple(w[i:i+3]) for i in range(max(0, len(w)-2))]
    trigram_counts = Counter([tg for t in texts for tg in trigrams_of(t)])
    # count posts that contain a trigram repeated elsewhere
    repeated_trigram_posts = 0
    for t in texts:
        if any(trigram_counts[tg] > 1 for tg in trigrams_of(t)):
            repeated_trigram_posts += 1
    trigram_repeat_ratio = (repeated_trigram_posts / len(texts)) if texts else 0.0

    return {
        'avg_char_len': float(avg_chars),
        'avg_tokens': float(avg_tokens),
        'ttr': float(ttr),
        'dup_post_ratio': float(dup_ratio),
        'trigram_repeat_ratio': float(trigram_repeat_ratio)
    }


def aggregate_user_features(df: pd.DataFrame, embeddings, embed_dim: int):
    # df: cleaned dataframe with columns ['user_id','text','timestamp', optional metrics]
    users = []
    rows = []
    for user_id, g in df.groupby('user_id'):
        texts = g['text'].astype(str).tolist()
        # behavioral
        n_posts = len(g)
        # timestamps -> posts per day
        try:
            # Usar 'timestamp' que ya fue mapeado desde 'Date' si existe
            times = pd.to_datetime(g['timestamp'])
            days_span = (times.max() - times.min()).days
            days_span = max(days_span, 1)
            posts_per_day = n_posts / days_span
        except Exception:
            posts_per_day = float(n_posts)

        # interactions
        def sum_or_zero(col):
            if col in g.columns:
                return int(g[col].fillna(0).sum()), float(g[col].fillna(0).mean())
            return 0, 0.0

        total_likes, avg_likes = sum_or_zero('likes')
        total_replies, avg_replies = sum_or_zero('replies')
        total_shares, avg_shares = sum_or_zero('shares')

        lex = compute_lexical_features(texts)

        # semantic: embeddings slice corresponding to this user's rows
        idxs = g.index.tolist()
        user_embs = embeddings[idxs]
        emb_mean = np.mean(user_embs, axis=0)
        emb_std = np.std(user_embs, axis=0)
        emb_std_norm = float(np.linalg.norm(emb_std))

        row = {
            'user_id': user_id,  # Mantener user_id en output (que viene de Name)
            'n_posts': n_posts,
            'posts_per_day': float(posts_per_day),
            'total_likes': int(total_likes),
            'avg_likes': float(avg_likes),
            'total_replies': int(total_replies),
            'avg_replies': float(avg_replies),
            'total_shares': int(total_shares),
            'avg_shares': float(avg_shares),
            **lex,
            'embedding_std_norm': emb_std_norm
        }
        # include embedding mean dims
        for i in range(embed_dim):
            row[f'emb_mean_{i}'] = float(emb_mean[i])

        rows.append(row)
    return pd.DataFrame(rows)


def run(input_paths: list[Path], output_path: Path, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 64):
    # Cargar y combinar múltiples archivos de entrada
    print(f"Cargando datos desde {len(input_paths)} archivo(s)...")
    dfs = []
    for input_path in input_paths:
        print(f"  - Leyendo {input_path}")
        df = pd.read_csv(input_path)
        # Eliminar columnas completamente vacías (Unnamed)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        dfs.append(df)
    
    # Combinar todos los DataFrames
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total de registros combinados: {len(df)}")
    
    # Preprocesamiento básico
    print("Aplicando limpieza básica de texto...")
    # Mapear columnas: Comment -> text, Name -> user_id (para compatibilidad interna)
    if 'Comment' in df.columns and 'text' not in df.columns:
        df['text'] = df['Comment']
    elif 'comment' in df.columns and 'text' not in df.columns:
        df['text'] = df['comment']
    elif 'text' not in df.columns:
        raise ValueError('El CSV debe contener una columna de texto: "Comment", "comment" o "text"')
    
    df['text'] = df['text'].astype(str).apply(basic_clean)

    # Mapear columnas de usuario y fecha
    if 'Name' in df.columns:
        df['user_id'] = df['Name']  # Usar Name como user_id internamente
    elif 'user_id' not in df.columns:
        raise ValueError('El CSV debe contener la columna "Name" o "user_id"')
    
    # Mapear fecha
    if 'Date' in df.columns:
        df['timestamp'] = df['Date']
    elif 'timestamp' not in df.columns:
        print('Advertencia: no se encontró "Date" ni "timestamp"; ciertas métricas de frecuencia se basarán en conteos simples')
    
    # Mapear Likes (con mayúscula)
    if 'Likes' in df.columns and 'likes' not in df.columns:
        df['likes'] = df['Likes']

    # Embeddings
    print(f"Cargando modelo de embeddings {model_name}...")
    model = SentenceTransformer(model_name)
    embed_dim = model.get_sentence_embedding_dimension()

    texts = df['text'].astype(str).tolist()
    embeddings = np.zeros((len(texts), embed_dim), dtype=float)
    print(f"Generando embeddings para {len(texts)} textos (batch_size={batch_size})...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings[i:i+len(batch), :] = emb

    # Agregar features por usuario
    print("Agregando features por usuario...")
    user_feats = aggregate_user_features(df.reset_index(drop=True), embeddings, embed_dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    user_feats.to_csv(output_path, index=False)
    print(f"Features generados y guardados en {output_path}")


def ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print('Descargando recursos NLTK...')
        nltk.download('punkt')


def main():
    parser = argparse.ArgumentParser(description='Pipeline de features por usuario')
    parser.add_argument('--input', type=str, nargs='+', default=['data/cleaned_comments.csv'],
                        help='Uno o más archivos CSV de entrada. Se combinarán en un solo dataset.')
    parser.add_argument('--output', type=str, default='data/user_features.csv',
                        help='Archivo CSV de salida con features agregadas por user_id')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Nombre del modelo de Sentence-BERT a usar')
    args = parser.parse_args()

    ensure_nltk()
    input_paths = [Path(p) for p in args.input]
    run(input_paths, Path(args.output), model_name=args.model)


if __name__ == '__main__':
    main()
