#!/usr/bin/env python3
"""
Script para limpiar archivos CSV de comentarios.

Extrae solo las columnas: Name, Comment, Date, Likes
de los archivos CSV en la carpeta data/, limpia emojis y combina todo en un único CSV.

Uso:
    python scripts/limpiar_comentarios.py
    python scripts/limpiar_comentarios.py --output data/comentarios_limpios.csv
"""

import argparse
import pandas as pd
import re
from pathlib import Path


def limpiar_emojis(texto: str) -> str:
    """
    Elimina emojis y otros símbolos Unicode no ASCII comunes de un texto.
    
    Args:
        texto: Texto a limpiar
        
    Returns:
        Texto sin emojis
    """
    if pd.isna(texto):
        return ""
    
    texto = str(texto)
    
    # Patrón para emojis y símbolos Unicode comunes
    # Incluye rangos de emojis, símbolos emoticonos, símbolos pictográficos, etc.
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "]+",
        flags=re.UNICODE
    )
    
    # Eliminar emojis
    texto = emoji_pattern.sub('', texto)
    
    # Limpiar espacios múltiples que puedan quedar
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto


def procesar_archivo(input_file: Path):
    """
    Procesa un archivo CSV extrayendo solo las columnas necesarias y limpiando emojis.
    
    Args:
        input_file: Ruta al archivo CSV de entrada
        
    Returns:
        DataFrame con las columnas limpias o None si hay error
    """
    print(f"  Procesando {input_file.name}...")
    
    try:
        # Leer el archivo CSV
        df = pd.read_csv(input_file)
        
        # Eliminar columnas completamente vacías (Unnamed)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Verificar que existan las columnas necesarias
        columnas_requeridas = ['Name', 'Comment', 'Date', 'Likes']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        
        if columnas_faltantes:
            print(f"    [ADVERTENCIA] Faltan columnas {columnas_faltantes} en {input_file.name}")
            return None
        
        # Extraer solo las columnas necesarias
        df_limpio = df[columnas_requeridas].copy()
        
        # Eliminar filas completamente vacías
        df_limpio = df_limpio.dropna(how='all')
        
        # Limpiar emojis de la columna Comment
        print(f"    Limpiando emojis de {len(df_limpio)} comentarios...")
        df_limpio['Comment'] = df_limpio['Comment'].apply(limpiar_emojis)
        
        print(f"    [OK] {len(df_limpio)} filas procesadas")
        return df_limpio
        
    except Exception as e:
        print(f"    [ERROR] Error procesando {input_file.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Limpiar y combinar archivos CSV de comentarios')
    parser.add_argument('--input-dir', type=str, default='data',
                        help='Directorio con los archivos CSV a limpiar')
    parser.add_argument('--output', type=str, default='data/comentarios_limpios.csv',
                        help='Archivo CSV de salida único con todos los comentarios combinados')
    parser.add_argument('--pattern', type=str, default='export_*.csv',
                        help='Patrón para buscar archivos (default: export_*.csv)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    
    if not input_dir.exists():
        print(f"[ERROR] El directorio {input_dir} no existe")
        return
    
    # Buscar todos los archivos CSV que coincidan con el patrón
    archivos = list(input_dir.glob(args.pattern))
    
    if not archivos:
        print(f"[ERROR] No se encontraron archivos que coincidan con '{args.pattern}' en {input_dir}")
        return
    
    print(f"Encontrados {len(archivos)} archivo(s) para procesar\n")
    
    # Procesar cada archivo y combinar los resultados
    dataframes = []
    exitosos = 0
    
    for archivo in archivos:
        df_limpio = procesar_archivo(archivo)
        if df_limpio is not None:
            dataframes.append(df_limpio)
            exitosos += 1
        print()
    
    if not dataframes:
        print("[ERROR] No se pudo procesar ningún archivo")
        return
    
    # Combinar todos los DataFrames en uno solo
    print("Combinando todos los archivos...")
    df_combinado = pd.concat(dataframes, ignore_index=True)
    
    # Eliminar duplicados si los hay (opcional, basado en todas las columnas)
    filas_antes = len(df_combinado)
    df_combinado = df_combinado.drop_duplicates()
    filas_despues = len(df_combinado)
    
    if filas_antes != filas_despues:
        print(f"  Eliminados {filas_antes - filas_despues} duplicados")
    
    # Guardar el archivo combinado
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_combinado.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n[COMPLETADO] {exitosos}/{len(archivos)} archivos procesados exitosamente")
    print(f"Total de comentarios: {len(df_combinado)}")
    print(f"Archivo guardado en: {output_file}")


if __name__ == '__main__':
    main()

