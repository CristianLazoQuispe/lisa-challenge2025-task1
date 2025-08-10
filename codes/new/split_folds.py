"""
CLI para dividir un conjunto de datos de cortes 2D en folds estratificados
por volumen para el desafío LISA 2025.

Cuando los datos consisten en múltiples cortes por volumen (p. ej., un
archivo NIfTI 3D convertido a varias imágenes 2D), es importante
asignar todos los cortes del mismo volumen al mismo fold para evitar
fugas de información.  Además, se desea mantener la distribución de
etiquetas raras entre los folds.  Este script utiliza la función
``assign_volume_stratified_folds`` para lograr ambos objetivos.

Uso de ejemplo
--------------

.. code-block:: bash

    python split_folds.py \
        --input_csv /ruta/al/train_imgs.csv \
        --output_csv /ruta/al/train_imgs_folds.csv \
        --volume_id patient_id \
        --n_splits 5 --top_k 2 --seed 42

Argumentos
---------

* ``--input_csv``: CSV de entrada con las columnas de etiquetas y la
  columna que identifica el volumen (por defecto, ``patient_id``).
* ``--output_csv``: CSV de salida donde se escribirá la columna ``fold``.
* ``--volume_id``: Nombre de la columna que agrupa los cortes por
  volumen.  Por ejemplo, ``patient_id`` o ``filename`` (sin sufijo de
  corte).
* ``--n_splits``: Número de folds para la cross‑validación.
* ``--top_k``: Número de etiquetas raras (según severidad 2) a usar
  para la estratificación.
* ``--seed``: Semilla aleatoria.

La lista de etiquetas se toma de ``new_repo.utils.LABELS``.  Asegúrese
de que su CSV tenga estas columnas con valores enteros 0/1/2.
"""

import argparse
import pandas as pd

from utils import assign_volume_stratified_folds, LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera folds estratificados por volumen para LISA 2025")
    parser.add_argument('--input_csv', type=str, required=True, help='Ruta al CSV de entrada con cortes 2D y etiquetas')
    parser.add_argument('--output_csv', type=str, required=True, help='Ruta al CSV de salida con la columna fold')
    parser.add_argument('--volume_id', type=str, default='patient_id', help='Columna que identifica el volumen (p.ej. patient_id)')
    parser.add_argument('--n_splits', type=int, default=5, help='Número de folds de cross‑validation')
    parser.add_argument('--top_k', type=int, default=2, help='Número de etiquetas raras para la estratificación')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Leer datos
    df = pd.read_csv(args.input_csv)
    # Verificar que las etiquetas existen
    missing = [lbl for lbl in LABELS if lbl not in df.columns]
    if missing:
        raise ValueError(f"Las etiquetas {missing} no se encuentran en el CSV de entrada")
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