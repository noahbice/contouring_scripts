import torch
import numpy as np
from torch.utils.data import DataLoader
from unet import Unet
import torch.nn.functional as F
from data import AutosegData
from tqdm import tqdm
from os import listdir

structs = ['Spinal-Cord', 'Neck-Right', 'Neck-Left', \
        'Submandibular-Gland-Right', 'Submandibular-Gland-Left', 'Parotid-Right', \
        'Parotid-Left', 'Oral-Cavity', 'Medulla-Oblongata', 'Brain'] 

#-------------------
mode = 'Spinal-Cord'
#training options
epochs = 400
batch_size = 16
learning_rate = 0.002
#model options
depth = 4
f=3
dropout = 0.25
batchnorm = True
padding = True
b_spline = False
rotate = False
#saving and loading
save_over = False
#-------------------

structs = {structs[i] : i for i in range(len(structs))}
structure = structs[mode]

model_load_file = './pthdata/' + mode + '.pth'
loss_load_file = './npydata/loss/' + mode + '_loss.npy'
val_loss_load_file = './npydata/loss/' + mode + '_validation_loss.npy'

if save_over:
    model_save_file = model_load_file
else:
    model_save_file = './pthdata/' + mode + '_updated.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(depth=depth, f=f, padding=padding, batchnorm=batchnorm, dropout=dropout).to(device)
model = model.train()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint = torch.load(model_load_file)
model.load_state_dict(checkpoint['model_state_dict'])


patients = listdir('./npydata/train/' + mode + '/cts/')
test_patients = listdir('./npydata/test/' + mode + '/cts/')
loss_arr = list(np.load(loss_load_file))
val_loss_arr = list(np.load(val_loss_load_file))
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
    
    np.save(loss_load_file, loss_arr)
    np.save(val_loss_load_file, val_loss_arr)