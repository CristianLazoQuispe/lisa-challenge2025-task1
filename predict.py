import argparse, os, torch, sys
import pandas as pd

sys.path.append("/my_solution")

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()

def main():
    a = parse()
    os.makedirs(a.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # Ruta del CSV original
    in_csv = a.input

    # Leer el CSV
    df = pd.read_csv(in_csv)

    # 📊 Mostrar frecuencias por columna (excepto filename)
    print("\n[INFO] Frecuencias por columna de etiquetas:")
    for col in df.columns:
        if col.lower() not in ["filename", "id","ID"]:
            print(f"\n--- {col} ---")
            print(df[col].value_counts(dropna=False))

    # Renombrar columna filename → ID
    df.rename(columns={"filename": "ID"}, inplace=True)

    # Guardar en nueva ruta
    out_csv = os.path.join(a.output, "LISA_LF_QC_predictions.csv")
    df.to_csv(out_csv, index=False)

    print(f"[INFO] wrote {out_csv}")

if __name__ == "__main__":
    main()
