import pandas as pd
import nibabel as nib
from tqdm import tqdm
import numpy as np

def get_nii_metadata(path,view_axis):
    img = nib.load(path)
    spacing = img.header.get_zooms()
    spacing = np.array(spacing)
    img = img.get_fdata()    
    if view_axis == "sag":
        pass
    elif view_axis == "cor":
        img = np.transpose(img, (1, 0, 2))
        spacing = spacing[[1,0,2]]
    elif view_axis == "axi":
        img = np.transpose(img, (2, 0, 1))
        spacing = spacing[[2,0,1]]

    shape = img.shape  # (X, Y, Z) or any order; often (Z, Y, X)
    return shape, spacing

def adding_metadata(df_data):
  # Inicializar listas
  dim_x, dim_y, dim_z = [], [], []
  spacing_x, spacing_y, spacing_z = [], [], []
  views = []

  for path in tqdm(df_data["path"]):
      # Detectar vista
      fname = path.split("/")[-1].lower()
      view = ""
      if "axi" in fname:
          view = "axi"
          views.append("axi")
      elif "cor" in fname:
          view = "cor"
          views.append("cor")
      elif "sag" in fname:
          view = "sag"
          views.append("sag")
      else:
          views.append("unknown")

      shape, spacing = get_nii_metadata(path,view)


      # Asegurar que sean 3D
      if len(shape) != 3:
          raise ValueError(f"Volumen no 3D en: {path}")

      # Guardar dimensiones
      dim_x.append(shape[0])
      dim_y.append(shape[1])
      dim_z.append(shape[2])

      # Guardar spacing
      spacing_x.append(spacing[0])
      spacing_y.append(spacing[1])
      spacing_z.append(spacing[2])



  # Agregar al dataframe
  df_data["dim_x"] = dim_x
  df_data["dim_y"] = dim_y
  df_data["dim_z"] = dim_z

  df_data["spacing_x"] = spacing_x
  df_data["spacing_y"] = spacing_y
  df_data["spacing_z"] = spacing_z

  df_data["view"] = views

  return df_data