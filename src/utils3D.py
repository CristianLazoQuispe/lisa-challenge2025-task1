from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
import numpy as np

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
