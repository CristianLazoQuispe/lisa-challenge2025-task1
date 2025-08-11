import pandas as pd
import cv2
import sys
import os

sys.path.append("../")
from src.utils3D  import PreProcessing3Dto2D
from src import utils

utils.set_seed(42)

results_dir = '../results/preprocessed_data/'
destination_dir = '/data/cristian/projects/med_data/rise-miccai/task-1-val/2d_images_all/'
labels=["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]


preprocessor3D = PreProcessing3Dto2D(binarize_threshold=0.15)
# 📂 Carga de CSV

df_train = pd.read_csv(os.path.join(results_dir, 'df_train.csv'))
df_test = pd.read_csv(os.path.join(results_dir, 'df_test.csv'))
df_train.head(2)

results_train = preprocessor3D.pipeline_3Dto2D(df_train,destination_dir,folder="train",labels=labels)
df_results_train = pd.DataFrame(results_train)
df_train = df_train.merge(df_results_train,on=["filename"],how="left")
filepath = os.path.join(results_dir,"df_train_imgs.csv")
df_train.to_csv(filepath,index=False)
print(f"Train file saved in : {filepath}")

results_test = preprocessor3D.pipeline_3Dto2D(df_test,destination_dir,folder="test",labels=labels)
df_results_test = pd.DataFrame(results_test)
df_test = df_test.merge(df_results_test,on=["filename"],how="left")
filepath = os.path.join(results_dir,"df_test_imgs.csv")
df_test.to_csv(filepath,index=False)
print(f"Test file saved in : {filepath}")
