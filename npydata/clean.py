import numpy as np
import os
from tqdm import tqdm

#------------------
window_level = True
window = 0.5
level = 0.5

crop = True
x_min = 105
x_max = 405
y_min = 105
y_max = 405
#------------------

ct_dir = './cts/'
mask_dir = './contours/'

for file in tqdm(os.listdir(ct_dir)):
    cts = np.load(ct_dir + file) / 2000.
    masks = np.load(mask_dir + file) #file names should be identical in different dirs

    if crop:
        cts = cts[:,x_min:x_max, y_min:y_max]
        masks = masks[:,:,x_min:x_max, y_min:y_max]
    
    if window_level:
        cts[np.where(cts > ((window/2) + level))] = ((window/2) + level)
        cts[np.where(cts < (level - (window/2)))] = (level - (window/2))
        if level - (window/2) <= 0:
            cts += np.abs(level - (window/2))
        else:
            cts -= np.abs(level - (window/2))
        cts *= 1./(window)

    np.save(ct_dir + file, cts)
    np.save(mask_dir + file, masks)
