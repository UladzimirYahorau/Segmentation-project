import numpy as np
import nibabel as nib
from brain_data import *




x = np.ones([153,198,198,2])
y = brain_mask(x)
save_segmentation_to_file(y, 1)
