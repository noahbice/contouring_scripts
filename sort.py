import numpy as np
import os
from tqdm import tqdm

structs = ['Spinal-Cord', 'Neck-Right', 'Neck-Left', \
        'Submandibular-Gland-Right', 'Submandibular-Gland-Left', 'Parotid-Right', \
        'Parotid-Left', 'Oral-Cavity', 'Medulla-Oblongata', 'Brain']

for name in structs:
    if name not in os.listdir('./npydata/train/'):
        os.mkdir('./npydata/train/' + name)
        os.mkdir('./npydata/train/' + name + '/cts')
        os.mkdir('./npydata/train/' + name + '/contours')
    if name not in os.listdir('./npydata/test/'):
        os.mkdir('./npydata/test/' + name)
        os.mkdir('./npydata/test/' + name + '/cts')
        os.mkdir('./npydata/test/' + name + '/contours')
        
#---------------------
structure = 'Spinal-Cord' #choose a structure to make data for
omitted_for_test_percent = 0.05 #percent omitted for validation
append_noise = True
percent_noise = 0.25 #if N slices, add N*percent_noise slices of non-structure to dataset
#TO DO -- 3d data [batch_dim, slice-1:slice+1,:,:]
#---------------------

ct_dir = './npydata/cts/'
mask_dir = './npydata/contours/'
num_patients = len(os.listdir('./npydata/cts/'))
omitted = int(num_patients*omitted_for_test_percent)
num_train = num_patients - omitted
print('{} patients omitted for validation.'.format(omitted))

structs = {i : structs[i] for i in range(len(structs))}
for (k, v) in structs.items():
    if v == structure:
        index = k


def sort_data(file_index_list, struct, index, save_dir, append_noise=True, percent_noise=0.25):
    
    for file_index in tqdm(file_index_list):
        file = '{}.npy'.format(file_index)
        
        contours = np.load(mask_dir + file)[:,index,:,:]
        N = contours.shape[0]
        slices = []
        for slice in range(N):
            if contours[slice].flatten().any():
                slices.append(slice)
        slices = np.array(slices)
        
        if append_noise:
            noise_indices = np.random.randint(N, size=int(percent_noise*len(slices)))
            slices = np.concatenate((slices, noise_indices))
            
        if not slices.size > 0:
            continue
            
        contours = contours[slices]
        
        cts = np.load(ct_dir + file)
        cts = cts[slices]

        np.save(save_dir + struct + '/cts/' + file, cts)
        np.save(save_dir + struct + '/contours/' + file, contours)

training_range = range(num_train)
print('Creating training data...')
sort_data(training_range, structs[index], index, './npydata/train/')

testing_range = range(num_train, num_patients)
print('Creating testing data...')
sort_data(testing_range, structs[index], index, './npydata/test/')