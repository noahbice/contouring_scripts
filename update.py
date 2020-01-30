import torch
import numpy as np
from torch.utils.data import DataLoader
from unet import Unet
import torch.nn.functional as F
from data import AutosegData
from tqdm import tqdm

#-------------------
mode = 'bowel'
#training options
epochs = 2500
batch_size = 25
learning_rate = 0.001
#model options
depth = 5
f=4
dropout = 0.25
batchnorm = True
padding = True
b_spline = True
rotate = True
#saving and loading
save_over = True
#-------------------

model_load_file = './pthdata/' + mode + '.pth'
loss_load_file = './npydata/loss/' + mode + '_loss.npy'
if save_over:
    model_save_file = model_load_file
    loss_save_file = loss_load_file
else:
    model_save_file = './pthdata/' + mode + '_updated.pth'
    loss_save_file = './npydata/loss/' + mode + '_updated_loss.npy'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(depth=depth, f=f, padding=padding, batchnorm=batchnorm, dropout=dropout).to(device)
model = model.train()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint = torch.load(model_load_file)
model.load_state_dict(checkpoint['model_state_dict'])
loss_arr = np.load(loss_load_file)
epoch_loss = 0
best_loss = np.inf
dataset = AutosegData(mode, b_spline=b_spline, rotate=rotate)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for _ in tqdm(range(epochs)):
    for X, y in dataloader:
        optim.zero_grad()
        X = X.to(device)
        y = y.to(device) 
        prediction = model(X.float())
        loss = F.binary_cross_entropy(prediction, y.float())
        l = loss.detach().cpu().numpy()
        epoch_loss += l
        loss.backward()
        optim.step()
    #print('loss: ' + str(epoch_loss))
    loss_arr = np.append(loss_arr, epoch_loss)
    if epoch_loss < best_loss:
            torch.save({'model_state_dict': model.state_dict()}, model_save_file)
            best_loss = epoch_loss
    epoch_loss = 0
    np.save(loss_save_file, loss_arr)