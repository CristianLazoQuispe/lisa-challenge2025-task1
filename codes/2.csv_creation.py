from glob import glob
import pandas as pd
import os
import sys
import os

sys.path.append("../")
from src.metadata import adding_metadata
from src.utils import set_seed

set_seed(seed=42)

train_path_dir = '/data/cristian/projects/med_data/rise-miccai/task-1/'
val_path_dir   = '/data/cristian/projects/med_data/rise-miccai/task-1-val/'
path_results   = "../results/"

val_list_paths = glob(os.path.join(val_path_dir,'*/**/*.gz'),recursive=True)
train_list_paths = glob(os.path.join(train_path_dir,'*/**/*.gz'),recursive=True)
train_list_paths_target = glob(os.path.join(train_path_dir,'*/**/*.csv'),recursive=True)

print("Number of training files       :", len(train_list_paths))
print("Number of validation files     :", len(val_list_paths))
print("Number of training train files:", len(train_list_paths_target))
print("Sample training file path      :", train_list_paths[0])
print("Sample training file train    :", train_list_paths_target[0])

df_train          = pd.read_csv(train_list_paths_target[0])
df_train_path_aux = pd.DataFrame(train_list_paths, columns=['path'])
df_train_path_aux['filename'] = df_train_path_aux['path'].apply(lambda x: x.split('/')[-1])
df_train = df_train.merge(df_train_path_aux, on='filename', how='left')
del df_train_path_aux


df_test = pd.DataFrame(val_list_paths, columns=['path'])
df_test['filename']     = df_test['path'].apply(lambda x: x.split('/')[-1])
df_train["patient_id"]  = df_train["filename"].str.extract(r"(LISA_\d+)")
df_test["patient_id"]   = df_test["filename"].str.extract(r"(LISA_VALIDATION_\d+)")

print("shape of train file:", df_train.shape)
print("shape of test  file:", df_test.shape)
df_train.head()

df_train = adding_metadata(df_train)
df_test  = adding_metadata(df_test)
print("shape of train file:", df_train.shape)
print("shape of test  file:", df_test.shape)
df_train.head()

# save datasets preprocessing ../results/preprocessed_data/
results_dir = os.path.join(path_results,'preprocessed_data/')
os.makedirs(results_dir, exist_ok=True)
df_train.to_csv(os.path.join(results_dir, 'df_train.csv'), index=False)
df_test.to_csv(os.path.join(results_dir, 'df_test.csv'), index=False)