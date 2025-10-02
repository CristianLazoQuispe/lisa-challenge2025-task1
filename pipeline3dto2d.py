import argparse
import os

import pandas as pd

from src import utils
from src.utils3D import PreProcessing3Dto2D


def main(args):
    utils.set_seed(42)

    labels = [
        "Noise",
        "Zipper",
        "Positioning",
        "Banding",
        "Motion",
        "Contrast",
        "Distortion"]

    # Preprocesador
    preprocessor3D = PreProcessing3Dto2D(binarize_threshold=0.15)

    # 📂 Cargar CSVs
    df_test = pd.read_csv(os.path.join(args.results_dir, 'df_test.csv'))

    # Procesar TEST
    results_test = preprocessor3D.pipeline_3Dto2D(
        df_test, args.destination_dir, folder="test", labels=labels)
    df_results_test = pd.DataFrame(results_test)
    df_test = df_test.merge(df_results_test, on=["filename"], how="left")
    filepath_test = os.path.join(args.results_dir, "df_test_imgs.csv")
    df_test.to_csv(filepath_test, index=False)
    print(f"Test file saved in : {filepath_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert 3D medical images to 2D projections.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/preprocessed_data/",
        help="Directory containing df_train.csv and df_test.csv.")
    parser.add_argument(
        "--destination_dir",
        type=str,
        default="/data/cristian/projects/med_data/rise-miccai/task-1-val/2d_images_all/",
        help="Directory to save 2D converted images.")

    args = parser.parse_args()
    main(args)

    # python 3.pipeline3dto2d.py --results_dir "./results/preprocessed_data" --destination_dir "./results/2d_images_all"
    # python 3.pipeline3dto2d.py --results_dir
    # "E:\Datathon\LISA\results\preprocessed_data" --destination_dir
    # "E:\Datathon\LISA\results\preprocessed_data\2d_images_all"
