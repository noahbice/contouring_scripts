import torch
import numpy as np
import matplotlib.pyplot as plt
from data_check import display_contours
from unet import Unet
from tqdm import tqdm

def plot_loss(loss_file='./npydata/loss.npy', plot_val=False, val_loss_file='./npydata/val_loss.npy'):
    l = np.load(loss_file)/131
    if plot_val:
        val_loss = np.load(val_loss_file)/6
    num = l.shape[0]
    idx = np.linspace(1, num, num)
#   plt.scatter(idx, l)
    plt.plot(idx, l, label='Training Loss')
    if plot_val:
        plt.plot(idx, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

def predict_contours(model, ct, size=(1,1,512,512)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    z_length = ct.shape[0]
    prediction = np.zeros(ct.shape, dtype='float32')
    for i in tqdm(range(z_length)):
        prediction[i] = model(torch.from_numpy(ct[i].reshape(size)).float().to(device)).detach().cpu().numpy()
    return prediction

def get_roc(preds, contours, plot=True, steps=30, eps=1e-5):
    #pixel classification TPF by threshold
    threshes = np.linspace(0., 0.2, steps)
    TPFs = []
    FPFs = []
    where_mask_zero = np.zeros(contours.shape)
    where_mask_zero[np.where(contours==0.)] = 1.
    for thresh in tqdm(threshes):
        bw = preds.copy()
        bw[np.where(bw >= thresh)] = 1.
        bw[np.where(bw != 1.)] = 0.
        where_bw_zero = np.zeros(bw.shape)
        where_bw_zero[np.where(bw==0)] = 1.
        
        TP = (bw*contours).sum()
        FN = (where_bw_zero*contours).sum()
        TPF = TP / (TP + FN)
        TPFs.append(TPF)
        
        FP = (bw*where_mask_zero).sum()
        TN = (where_mask_zero*where_bw_zero).sum()
        FPF = FP / (FP + TN)
        FPFs.append(FPF)
     
    if plot:
        plt.plot(FPFs, TPFs)
        plt.scatter(FPFs, TPFs)
        plt.xlabel('False Positive Fraction')
        plt.ylabel('True Positive Fraction')
        plt.title('ROC Curve')
        plt.show()
    
    return FPFs, TPFs

def get_dice(preds, contours, steps=30, eps=1e-5, plot=True, mode='Dice by threshold'):
    dices = []
    threshes = np.linspace(0., 1., steps)
    print('Plotting dice by treshold...')
    for thresh in tqdm(threshes):
        bw = preds.copy()
        bw[np.where(bw >= thresh)] = 1.
        bw[np.where(bw != 1.)] = 0.
        intersection = (bw*contours).sum()
        union = (bw.sum() + contours.sum() + eps)
        dice = (2.*intersection) / union
        dices.append(dice)
    
    if plot:
        plt.plot(threshes, dices)
        #plt.scatter(threshes., dices)
        plt.xlabel('Threshold')
        plt.ylabel('Dice score')
        plt.title(mode)
        plt.show()
    return threshes, dices
    
if __name__ == '__main__':

    #----------------------
    data_mode = 'Parotid-Right'
    model_mode = 'Parotid-Right'
    plot_training_loss = True
    plot_val_loss = True
    make_roc = False #make sure test contours are correct mode
    make_dice = False #same
    plot_masks = False
    plot_contours = True
    
    #model options
    depth = 4
    f=3
    dropout = 0.25
    batchnorm = True
    padding = True
    
    ##other options
    size = (1,1,300,300)
    draw_thresh_val = 10. #0 to 256
    steps = 100
    
    #window/level
    preprocess = False
    window = 0.6
    level = 0.5
    #----------------------
    
    ct_file = './npydata/cts/136.npy'
    contour_file = './npydata/contours/136.npy'
    model_file = './pthdata/' + model_mode + '.pth'
    loss_file = './npydata/loss/' + model_mode + '_training_loss.npy'
    
    if plot_val_loss:
        val_loss_file = './npydata/loss/' + model_mode + '_validation_loss.npy'
    
    if plot_training_loss:
        plot_loss(loss_file=loss_file, plot_val=plot_val_loss, val_loss_file = val_loss_file)

    ct = np.load(ct_file)[0:500]
    contours = np.load(contour_file)[0:500]
    
    if preprocess:
        ct = ct / 2000
        print('Changing window/level...')
        ct[np.where(ct > ((window/2) + level))] = ((window/2) + level)
        ct[np.where(ct < (level - (window/2)))] = (level - (window/2))
        if level - (window/2) <= 0:
            ct += np.abs(level - (window/2))
        else:
            ct -= np.abs(level - (window/2))
        ct *= 1./(window)
   
    print('Drawing contours...')
    model = Unet(depth=depth, f=f, padding=padding, batchnorm=batchnorm, dropout=dropout)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    preds = predict_contours(model, ct, size=size)
    
    if make_roc:
        print('Creating ROC curve...')
        _, _ = get_roc(preds, contours, steps=steps)
    
    if make_dice:
        _, _ = get_dice(preds, contours, steps=steps, mode=model_mode)
        
    display = display_contours(ct, preds, fill=True, threshold=draw_thresh_val)
    
    if plot_masks:
        print('Displaying masks.')
        display.plot_masks()
    
    if plot_contours:
        print('Displaying contours.')
        display.plot_contours(thresh_val=draw_thresh_val)