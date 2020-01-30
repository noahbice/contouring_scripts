import torch
import numpy as np
from unet import Unet
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import AutosegData
from tqdm import tqdm
from os import listdir
from random import shuffle
import matplotlib.pyplot as plt

structs = ['Spinal-Cord', 'Neck-Right', 'Neck-Left', \
        'Submandibular-Gland-Right', 'Submandibular-Gland-Left', 'Parotid-Right', \
        'Parotid-Left', 'Oral-Cavity', 'Medulla-Oblongata', 'Brain']   
        
#-------------------
#training options
mode = 'Medulla-Oblongata'
epochs = 100
batch_size = 16
learning_rate = 0.005
b_spline = False
rotate = False

#model options
depth = 4
f=3
dropout = 0.2
batchnorm = True
padding = True
device = 'cuda'

#saving options
model_save_directory = './pthdata/' + mode + '.pth'
loss_save_directory = './npydata/loss/'
#-------------------

structs = {structs[i] : i for i in range(len(structs))}
structure = structs[mode]

model = Unet(depth=depth, f=f, padding=padding, batchnorm=batchnorm, dropout=dropout).to(device)
model = model.float()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

patients = listdir('./npydata/train/' + mode + '/cts/')
test_patients = listdir('./npydata/test/' + mode + '/cts/')
loss_arr = []
val_loss_arr = []
best_loss = np.inf

for epoch in tqdm(range(epochs)):
    shuffle(patients)
    epoch_loss = 0.
    for patient in patients:
        index = patient.split('.')[0]
        data = AutosegData(mode, index, b_spline=b_spline, rotate=rotate)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        for X, y in dataloader:
            optim.zero_grad()
            prediction = model(X)
            loss = F.binary_cross_entropy(prediction.reshape(-1), y.float().reshape(-1))
            l = loss.detach().cpu().numpy()
            epoch_loss += l
            loss.backward()
            optim.step()
            
    #free up GPU memory
    del data
    del dataloader
    torch.cuda.empty_cache()
    
    #get validation loss
    val_loss = 0.
    for file in test_patients:
        index = file.split('.')[0]
        data = AutosegData(mode, index, b_spline=False, rotate=False, test=True)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        for X, y in dataloader:
            prediction = model(X)
            loss = F.binary_cross_entropy(prediction.reshape(-1), y.float().reshape(-1))
            l = loss.detach().cpu().numpy()
            val_loss += l
            
    if val_loss < best_loss: #change to val_loss
        torch.save({'model_state_dict': model.state_dict()}, model_save_directory)
        best_loss = val_loss
        
    loss_arr.append(epoch_loss)
    val_loss_arr.append(val_loss)
    
    np.save(loss_save_directory + mode + '_training_loss.npy', loss_arr)
    np.save(loss_save_directory + mode + '_validation_loss.npy', val_loss_arr)  
