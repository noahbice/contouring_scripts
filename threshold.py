import numpy as np
import torch
import matplotlib.pyplot as plt
from deploy import get_dice, predict_contours
from unet import Unet
from tqdm import tqdm
from os import listdir


def get_frac(maskdir):
    total = 0.
    num = 0.
    for file in listdir(maskdir):
        mask = np.load(maskdir + file).astype(np.int)
        total += np.prod(mask.shape)
        num += np.sum(mask)
    return num / total
    
def get_dice(model, ctdir, maskdir, steps=100, eps=1e-5):
    dices = []
    threshes = np.linspace(0., 1., steps)
    for thresh in tqdm(threshes):
        int_total = 0
        union_total = 0
        for file in listdir(ctdir):
            ct = np.load(ctdir + file)
            contours = np.load(maskdir + file)
            preds = predict_contours(model.cuda(), ct, size=(1,1,300,300))
            bw = preds.copy()
            bw[np.where(bw >= thresh)] = 1.
            bw[np.where(bw != 1.)] = 0.
            int_total += (bw*contours).sum()
            union_total += (bw.sum() + contours.sum() + eps)
            
        dice = (2.*int_total) / union_total
        dices.append(dice)
    return dices

if __name__ == '__main__':


    #---------------------------------
    count = True
    dice = True

    #model options
    depth = 4
    f=3
    dropout = 0.2
    batchnorm = True
    padding = True
    #---------------------------------
    
    structs = ['Spinal-Cord', 'Neck-Right', 'Neck-Left', \
        'Submandibular-Gland-Right', 'Submandibular-Gland-Left', 'Parotid-Right', \
        'Parotid-Left', 'Oral-Cavity', 'Medulla-Oblongata', 'Brain']
    struct_dict = {structs[i]: i for i in range(len(structs))}
    dice_data = []
    counts = []    
    
    for struct in tqdm(structs):

        if count:
            trainMaskDir = './npydata/train/' + struct + '/contours/'
            frac = get_frac(trainMaskDir)
            counts.append(frac)
        
        if dice:
            ctdir = './npydata/test/' + struct + '/cts/'
            maskdir = './npydata/test/' + struct + '/contours/'
            model = Unet(depth=depth, f=f, padding=padding, batchnorm=batchnorm, dropout=dropout)
            checkpoint = torch.load('./pthdata/' + struct + '.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            dice_data.append(get_dice(model, ctdir, maskdir))
            
    np.save('./npydata/dice.npy', np.array(dice_data))
    np.save('./npydata/count.npy', np.array(counts))
     
    idx = np.linspace(0.,1., 100)
    for item in range(dice_data.shape[0]):
        plt.plot(idx, dice_data[item], label=structs[item])
    plt.legend()
    plt.show()

