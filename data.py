from torch.utils.data import Dataset
import torch
import numpy as np
import random
import gryds

class AutosegData(Dataset):
    def __init__(self, mode, patient, rotate=False, b_spline=False, device = 'cuda', test=False):
        
        if test:
            self.inputs = np.load('./npydata/test/' + mode  + '/cts/' + str(patient) + '.npy')
            self.masks = np.load('./npydata/test/' + mode + '/contours/' + str(patient) + '.npy')
        else:
            self.inputs = np.load('./npydata/train/' + mode  + '/cts/' + str(patient) + '.npy')
            self.masks = np.load('./npydata/train/' + mode + '/contours/' + str(patient) + '.npy')
        self.n = 3
        self.stretch = 0.1
        self.max_angle = 5. #degrees
        self.b_spline = b_spline
        self.rotate = rotate
        self.device = device
                
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, sub_idx):
        image = self.inputs[sub_idx]
        mask = self.masks[sub_idx]
        if self.b_spline:
            grid = np.random.rand(2, self.n, self.n) - 0.5
            grid *= self.stretch
            bspline = gryds.BSplineTransformationCuda(grid)

            slice_interpolator = gryds.BSplineInterpolatorCuda(image, cval=0)
            image = slice_interpolator.transform(bspline)
            mask_interpolator = gryds.BSplineInterpolatorCuda(mask)
            mask = mask_interpolator.transform(bspline)
            
        if self.rotate:
            angle = (random.random() - 0.5)*2*(np.pi/180)*self.max_angle
            rot = gryds.AffineTransformation(ndim=2,angles=[angle], center=[0.5,0.5])
            im_interp = gryds.Interpolator(image)
            image = im_interp.transform(rot)
            mask_interp = gryds.Interpolator(mask)
            mask = mask_interp.transform(rot)
            
        image = torch.tensor(image).unsqueeze(0).to(self.device).float()
        mask = torch.tensor(mask).unsqueeze(0).to(self.device).float()
        return image, mask

