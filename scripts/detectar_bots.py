#!/usr/bin/env python3
"""
Script para detectar bots en comentarios basándose en características de comportamiento,
léxicas y semánticas.

Utiliza las features calculadas por pipeline.py y agrega features adicionales específicas
para detección de bots, luego aplica un sistema de scoring heurístico.

Uso:
    python scripts/detectar_bots.py --input data/user_features.csv --output data/bot_detection_results.csv
    python scripts/detectar_bots.py --input data/user_features.csv --output data/bot_detection_results.csv --threshold 0.5
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


def analyze_username_pattern(username: str) -> dict:
    """
    Analiza patrones sospechosos en nombres de usuario que pueden indicar bots.
    
    Args:
        username: Nombre de usuario a analizar
        
    Returns:
        Diccionario con features del nombre de usuario
    """
    if pd.isna(username):
        username = ""
    username = str(username).strip()
    
    # Remover @ si está presente
    username = username.lstrip('@')
    
    # Longitud del nombre
    username_len = len(username)
    
    # Contar números
    num_digits = sum(c.isdigit() for c in username)
    digit_ratio = num_digits / username_len if username_len > 0 else 0.0
    
    # Contar caracteres especiales (guiones, puntos, guiones bajos)
    special_chars = sum(c in '-._' for c in username)
    special_ratio = special_chars / username_len if username_len > 0 else 0.0
    
    # Contar letras
    num_letters = sum(c.isalpha() for c in username)
    letter_ratio = num_letters / username_len if username_len > 0 else 0.0
    
    # Patrones sospechosos comunes
    has_random_numbers = bool(re.search(r'\d{4,}', username))  # 4+ dígitos consecutivos
    has_mixed_random = bool(re.search(r'[a-z]\d{3,}|\d{3,}[a-z]', username.lower()))  # letra+3+digitos o viceversa
    is_very_short = username_len < 5
    is_very_long = username_len > 30
    
    # Score heurístico de sospecha del nombre (0-1)
    name_suspicion = 0.0
    if digit_ratio > 0.5:
        name_suspicion += 0.3
    if has_random_numbers:
        name_suspicion += 0.2
    if has_mixed_random:
        name_suspicion += 0.2
    if special_ratio > 0.3:
        name_suspicion += 0.1
    if is_very_short or is_very_long:
        name_suspicion += 0.1
    if letter_ratio < 0.3 and username_len > 0:
        name_suspicion += 0.1
    
    name_suspicion = min(1.0, name_suspicion)
    
    return {
        'username_len': username_len,
        'username_digit_ratio': digit_ratio,
        'username_special_ratio': special_ratio,
        'username_letter_ratio': letter_ratio,
        'username_has_random_numbers': int(has_random_numbers),
        'username_suspicion_score': name_suspicion
    }


def compute_temporal_regularity(df_user: pd.DataFrame) -> dict:
    """
    Calcula métricas de regularidad temporal de los posts de un usuario.
    Los bots suelen tener intervalos muy regulares entre posts.
    
    Args:
        df_user: DataFrame con los posts de un usuario (debe tener columna 'timestamp')
        
    Returns:
        Diccionario con métricas de regularidad temporal
    """
    try:
        times = pd.to_datetime(df_user['timestamp'])
        times = times.sort_values()
        
        if len(times) < 2:
            return {
                'temporal_std_seconds': 0.0,
                'temporal_cv': 0.0,  # coefficient of variation
                'temporal_regularity_score': 0.0
            }
        
        # Calcular intervalos entre posts (en segundos)
        intervals = times.diff().dropna().dt.total_seconds()
        intervals = intervals[intervals > 0]  # Filtrar intervalos válidos
        
        if len(intervals) == 0:
            return {
                'temporal_std_seconds': 0.0,
                'temporal_cv': 0.0,
                'temporal_regularity_score': 0.0
            }
        
        mean_interval = intervals.mean()
        std_interval = intervals.std()
        
        # Coefficient of variation (CV): std/mean
        # CV bajo = muy regular (sospechoso de bot)
        cv = std_interval / mean_interval if mean_interval > 0 else 0.0
        
        # Score de regularidad: CV bajo indica alta regularidad (más sospechoso)
        # Normalizar: CV < 0.5 = muy regular (score alto), CV > 2.0 = muy irregular (score bajo)
        if cv < 0.5:
            regularity_score = 1.0 - (cv / 0.5)  # 1.0 cuando CV=0, 0.0 cuando CV=0.5
        elif cv < 2.0:
            regularity_score = max(0.0, 1.0 - ((cv - 0.5) / 1.5))  # Decrece gradualmente
        else:
            regularity_score = 0.0
        
        return {
            'temporal_std_seconds': float(std_interval),
            'temporal_cv': float(cv),
            'temporal_regularity_score': float(regularity_score)
        }
    except Exception:
        return {
            'temporal_std_seconds': 0.0,
            'temporal_cv': 0.0,
            'temporal_regularity_score': 0.0
        }


def compute_engagement_ratio(user_features: dict) -> float:
    """
    Calcula el ratio de engagement (interacciones por post).
    Los bots suelen tener bajo engagement relativo a su frecuencia de posting.
    
    Args:
        user_features: Diccionario con features del usuario
        
    Returns:
        Ratio de engagement normalizado
    """
    n_posts = user_features.get('n_posts', 1)
    posts_per_day = user_features.get('posts_per_day', 0)
    avg_likes = user_features.get('avg_likes', 0)
    avg_replies = user_features.get('avg_replies', 0)
    avg_shares = user_features.get('avg_shares', 0)
    
    # Engagement total promedio por post
    total_engagement = avg_likes + avg_replies + avg_shares
    
    # Si postea mucho pero tiene poco engagement, es sospechoso
    # Normalizar: engagement bajo + alta frecuencia = score alto de bot
    if posts_per_day > 0:
        engagement_per_frequency = total_engagement / posts_per_day
        # Si engagement_per_frequency < 0.1, es muy sospechoso
        if engagement_per_frequency < 0.1:
            return 1.0
        elif engagement_per_frequency < 0.5:
            return 0.7
        elif engagement_per_frequency < 1.0:
            return 0.4
        else:
            return 0.1
    else:
        return 0.0


def calculate_bot_score(user_features: dict, username: str, df_user_posts: pd.DataFrame = None,
                       weights: dict = None) -> dict:
    """
    Calcula un score de probabilidad de bot (0-1) basado en múltiples features.
    
    Args:
        user_features: Diccionario con features del usuario (del pipeline)
        username: Nombre de usuario
        df_user_posts: DataFrame con posts del usuario (opcional, para regularidad temporal)
        weights: Diccionario con pesos para cada componente del score (opcional)
        
    Returns:
        Diccionario con scores individuales y score final
    """
    if weights is None:
        weights = {
            'frequency': 0.20,      # posts_per_day
            'repetition': 0.25,      # dup_post_ratio, trigram_repeat_ratio
            'diversity': 0.15,        # ttr
            'semantic': 0.15,        # embedding_std_norm
            'username': 0.10,        # username_suspicion_score
            'temporal': 0.10,        # temporal_regularity_score
            'engagement': 0.05       # engagement_ratio
        }
    
    # 1. Score de frecuencia (alta frecuencia = más sospechoso)
    posts_per_day = user_features.get('posts_per_day', 0)
    if posts_per_day > 10:
        freq_score = 1.0
    elif posts_per_day > 5:
        freq_score = 0.8
    elif posts_per_day > 2:
        freq_score = 0.5
    elif posts_per_day > 1:
        freq_score = 0.3
    else:
        freq_score = 0.0
    
    # 2. Score de repetición (alta repetición = más sospechoso)
    dup_ratio = user_features.get('dup_post_ratio', 0)
    trigram_ratio = user_features.get('trigram_repeat_ratio', 0)
    repetition_score = max(dup_ratio, trigram_ratio * 0.8)  # Trigramas son menos definitivos
    
    # 3. Score de diversidad léxica (baja diversidad = más sospechoso)
    ttr = user_features.get('ttr', 1.0)
    if ttr < 0.3:
        diversity_score = 1.0
    elif ttr < 0.5:
        diversity_score = 0.7
    elif ttr < 0.7:
        diversity_score = 0.4
    else:
        diversity_score = 0.0
    
    # 4. Score de variación semántica (baja variación = más sospechoso)
    emb_std_norm = user_features.get('embedding_std_norm', 1.0)
    # Normalizar: embedding_std_norm típicamente está entre 0 y ~2 para MiniLM
    if emb_std_norm < 0.3:
        semantic_score = 1.0
    elif emb_std_norm < 0.5:
        semantic_score = 0.7
    elif emb_std_norm < 0.8:
        semantic_score = 0.4
    else:
        semantic_score = 0.0
    
    # 5. Score de nombre de usuario
    username_features = analyze_username_pattern(username)
    username_score = username_features['username_suspicion_score']
    
    # 6. Score de regularidad temporal
    if df_user_posts is not None:
        temporal_features = compute_temporal_regularity(df_user_posts)
        temporal_score = temporal_features['temporal_regularity_score']
    else:
        temporal_score = 0.0
    
    # 7. Score de engagement
    engagement_score = compute_engagement_ratio(user_features)
    
    # Calcular score final ponderado
    bot_score = (
        weights['frequency'] * freq_score +
        weights['repetition'] * repetition_score +
        weights['diversity'] * diversity_score +
        weights['semantic'] * semantic_score +
        weights['username'] * username_score +
        weights['temporal'] * temporal_score +
        weights['engagement'] * engagement_score
    )
    
    # Asegurar que esté en [0, 1]
    bot_score = max(0.0, min(1.0, bot_score))
    
    return {
        'bot_score': bot_score,
        'freq_score': freq_score,
        'repetition_score': repetition_score,
        'diversity_score': diversity_score,
        'semantic_score': semantic_score,
        'username_score': username_score,
        'temporal_score': temporal_score,
        'engagement_score': engagement_score,
        **username_features,
        **({'temporal_std_seconds': temporal_features.get('temporal_std_seconds', 0.0),
            'temporal_cv': temporal_features.get('temporal_cv', 0.0)} if df_user_posts is not None else {})
    }


def detect_bots(user_features_path: str, comments_path: str = None,
                threshold: float = 0.5, output_path: str = None) -> pd.DataFrame:
    """
    Detecta bots en un dataset de features de usuarios.
    
    Args:
        user_features_path: Ruta al CSV con features por usuario (generado por pipeline.py)
        comments_path: Ruta opcional al CSV con comentarios originales (para calcular regularidad temporal)
        threshold: Umbral para clasificar como bot (default: 0.5)
        output_path: Ruta opcional para guardar resultados
        
    Returns:
        DataFrame con usuarios y sus scores de bot, etiquetados como bot/humano
    """
    print(f"Cargando features de usuarios desde {user_features_path}...")
    user_features_df = pd.read_csv(user_features_path)
    
    # Cargar comentarios originales si se proporciona (para regularidad temporal)
    df_comments = None
    if comments_path and Path(comments_path).exists():
        print(f"Cargando comentarios desde {comments_path}...")
        df_comments = pd.read_csv(comments_path)
        # Normalizar columnas
        if 'Name' in df_comments.columns:
            df_comments['user_id'] = df_comments['Name']
        if 'Date' in df_comments.columns:
            df_comments['timestamp'] = df_comments['Date']
        if 'Comment' in df_comments.columns:
            df_comments['text'] = df_comments['Comment']
    
    print(f"Calculando scores de bot para {len(user_features_df)} usuarios...")
    results = []
    
    for idx, row in user_features_df.iterrows():
        user_id = row['user_id']
        user_features = row.to_dict()
        
        # Obtener posts del usuario si tenemos los comentarios
        df_user_posts = None
        if df_comments is not None:
            df_user_posts = df_comments[df_comments['user_id'] == user_id].copy()
            if len(df_user_posts) == 0:
                df_user_posts = None
        
        # Calcular score de bot
        bot_scores = calculate_bot_score(user_features, user_id, df_user_posts)
        
        # Combinar features originales con scores de bot
        result_row = {**user_features, **bot_scores}
        result_row['is_bot'] = 1 if bot_scores['bot_score'] >= threshold else 0
        result_row['bot_probability'] = bot_scores['bot_score']
        
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    
    # Estadísticas
    n_bots = results_df['is_bot'].sum()
    pct_bots = (n_bots / len(results_df)) * 100 if len(results_df) > 0 else 0
    
    print(f"\nDetección completada:")
    print(f"  Total usuarios: {len(results_df)}")
    print(f"  Bots detectados: {n_bots} ({pct_bots:.1f}%)")
    print(f"  Humanos: {len(results_df) - n_bots} ({100 - pct_bots:.1f}%)")
    
    # Guardar si se especifica output
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResultados guardados en {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Detectar bots en comentarios')
    parser.add_argument('--input', type=str, required=True,
                        help='Archivo CSV con features de usuarios (generado por pipeline.py)')
    parser.add_argument('--comments', type=str, default=None,
                        help='Archivo CSV opcional con comentarios originales (para calcular regularidad temporal)')
    parser.add_argument('--output', type=str, default='data/bot_detection_results.csv',
                        help='Archivo CSV de salida con resultados de detección')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Umbral para clasificar como bot (0-1, default: 0.5)')
    args = parser.parse_args()
    
    detect_bots(
        user_features_path=args.input,
        comments_path=args.comments,
        threshold=args.threshold,
        output_path=args.output
    )


if __name__ == '__main__':
    main()

