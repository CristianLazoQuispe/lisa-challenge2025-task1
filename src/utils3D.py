from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
from .dataset3D import MRIDataset3D
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy
import os

class PreProcessing3Dto2D:
    def __init__(self,binarize_threshold=0.15):
        self.binarize_threshold = binarize_threshold
   
    def morphological_operations(self,original):
        mask = original >= self.binarize_threshold

        H, W = original.shape
        final  = binary_dilation(mask,  structure=np.ones((2,2), dtype=bool))
        final  = binary_erosion(final,  structure=np.ones((2,2), dtype=bool))
        final  = binary_dilation(final, structure=np.ones((2,2), dtype=bool))
        final  = binary_erosion(final,  structure=np.ones((3,3), dtype=bool))
        final  = binary_dilation(final, structure=np.ones((3,3), dtype=bool))
        final  = binary_dilation(final, structure=np.ones((5,5), dtype=bool))
        final  = binary_erosion(final,  structure=np.ones((2,2), dtype=bool))
        ratio  = np.count_nonzero(mask)/(H*W)
        return final, ratio

    def reorientation(self,slices, view_axis):
        # Reorient slices según vista
        if view_axis == "sag":
            slices = slices
        elif view_axis == "cor":
            slices = np.transpose(slices, (1, 0, 2))
        elif view_axis == "axi":
            slices = np.transpose(slices, (2, 0, 1))
        else:
            raise ValueError(f"Invalid view_axis: {view_axis}")
        return slices

    def pipeline_3Dto2D(self,df_data,destination_dir,folder="",
                        labels=["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]):
        destination_dir = os.path.join(destination_dir,folder)
        os.makedirs(destination_dir,exist_ok=True)

        results = []
        for idx in tqdm(range(0,df_data.shape[0])):

            nifti_path = df_data.loc[idx,"path"]
            filename = df_data.loc[idx,"filename"]
            df = pd.DataFrame({"path": [nifti_path], "filename": [filename]})
            dataset = MRIDataset3D(df, is_train=False,spatial_size=(40,120,120),labels=labels)
            volume,_,filename,view = dataset[0]
            volume = volume[0].cpu().numpy()


            for idx_slice in range(volume.shape[0]):
                original = copy.deepcopy(volume[idx_slice])
                mask, ratio     = self.morphological_operations(original)
                mask = np.array(mask, dtype=np.float32)
                img_path = os.path.join(destination_dir,filename.replace(".nii.gz",f"_{idx_slice}.png"))
                npy_path = os.path.join(destination_dir,filename.replace(".nii.gz",f"_{idx_slice}.npy"))
                #print(img_path)

                np.save(npy_path, volume[idx_slice])
                plt.imsave(img_path, volume[idx_slice], cmap="gray")

                values= {"filename":filename,"img_path":img_path,"npy_path":npy_path,"ratio":ratio}
                results.append(values)
        return results