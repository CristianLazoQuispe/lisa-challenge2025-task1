import argparse
import os
from glob import glob

import pandas as pd

from src.metadata import adding_metadata
from src.utils import set_seed


def main(args):
    set_seed(seed=42)

    val_list_paths = glob(
        os.path.join(
            args.val_path_dir,
            '**',
            '*.nii.gz'),
        recursive=True)

    print("Number of validation files     :", len(val_list_paths))

    # Create initial DataFrame
    df_test = pd.DataFrame(val_list_paths, columns=['path'])
    df_test['filename'] = df_test['path'].apply(lambda x: os.path.basename(x))
    df_test["patient_id"] = df_test["filename"].str.extract(
        r"(LISA_(?:TESTING|VALIDATION)_\d+)")
    df_test = df_test[df_test['filename'].str.endswith(
        '.nii.gz')].reset_index(drop=True)

    print("Shape of test file:", df_test.shape)

    df_test = adding_metadata(df_test)
    print("Shape of test file after metadata:", df_test.shape)

    # Save CSV
    results_dir = os.path.join(args.path_results, 'preprocessed_data/')
    os.makedirs(results_dir, exist_ok=True)
    df_test.to_csv(os.path.join(results_dir, 'df_test.csv'), index=False)
    print(
        f"Saved preprocessed CSV to {
            os.path.join(
                results_dir,
                'df_test.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process validation dataset paths and metadata.")
    parser.add_argument("--val_path_dir", type=str, default="./input/",
                        help="Directory containing validation files.")
    parser.add_argument("--path_results", type=str, default="./results/",
                        help="Directory to save results.")

    args = parser.parse_args()
    main(args)

    # python 2.csv_creation.py --val_path_dir "/input/" --path_results "./results/"
    # python 2.csv_creation.py --val_path_dir
    # "E:\Datathon\LISA\input\task-1-val" --path_results
    # "E:\Datathon\LISA\results"
