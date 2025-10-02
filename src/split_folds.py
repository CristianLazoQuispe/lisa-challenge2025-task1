"""
Split dataset into stratified folds by volume for LISA 2025 challenge.
"""

import argparse

import pandas as pd

from utils import LABELS, assign_volume_stratified_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera folds estratificados por volumen para LISA 2025")
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Ruta al CSV de entrada con cortes 2D y etiquetas')
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Ruta al CSV de salida con la columna fold')
    parser.add_argument(
        '--volume_id',
        type=str,
        default='patient_id',
        help='Columna que identifica el volumen (p.ej. patient_id)')
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='Número de folds de cross‑validation')
    parser.add_argument(
        '--top_k',
        type=int,
        default=2,
        help='Número de etiquetas raras para la estratificación')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Leer datos
    df = pd.read_csv(args.input_csv)
    # Verificar que las etiquetas existen
    missing = [lbl for lbl in LABELS if lbl not in df.columns]
    if missing:
        raise ValueError(
            f"Las etiquetas {missing} no se encuentran en el CSV de entrada")
    # Asignar folds
    df_folds = assign_volume_stratified_folds(
        df,
        volume_id_col=args.volume_id,
        label_cols=LABELS,
        n_splits=args.n_splits,
        top_k=args.top_k,
        seed=args.seed,
    )
    # Guardar
    df_folds.to_csv(args.output_csv, index=False)
    print(f"Archivo de salida con folds guardado en {args.output_csv}")


if __name__ == '__main__':
    main()
